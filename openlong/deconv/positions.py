"""True variant position identification.

Statistical method for distinguishing true polymorphic positions from
sequencing errors in the corrected MSA.

The key challenge: after INDEL correction, some positions still show
variation. We need to determine which of these represent true biological
variants vs. residual sequencing errors.

Approach:
1. For each position, compute the minor allele frequency (MAF)
2. Apply minimum MAF and Shannon entropy pre-filters (platform-calibrated)
3. Apply a binomial test: is the observed MAF significantly higher
   than expected from the platform's residual error rate?
4. Apply multiple testing correction (Benjamini-Hochberg FDR)
5. Positions passing all filters are declared true variant positions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Base encoding
GAP = 0

# Platform-specific residual error rates (after INDEL correction)
# These defaults can be modified at runtime using set_error_rate()
# Platform-calibrated error rates, tuned on real sequencing data.
# CLR/ONT rates are higher than often cited because they account for
# residual INDEL artifacts that survive correction plus misalignment noise.
PLATFORM_ERROR_RATES = {
    "pacbio_clr": 0.05,    # ~5% residual after correction (real CLR data)
    "pacbio_hifi": 0.005,  # ~0.5% for HiFi reads (conservative)
    "ont": 0.06,           # ~6% for ONT R10 (real ONT data)
    "unknown": 0.08,
}


def set_error_rate(platform: str, rate: float) -> None:
    """Set the error rate for a sequencing platform.

    Allows runtime configuration of platform-specific error rates.

    Args:
        platform: Platform name (e.g., 'pacbio_clr', 'pacbio_hifi', 'ont').
        rate: Error rate as a float between 0 and 1.

    Raises:
        ValueError: If rate is not between 0 and 1.
    """
    if not 0 <= rate <= 1:
        raise ValueError(f"Error rate must be between 0 and 1, got {rate}")
    PLATFORM_ERROR_RATES[platform] = rate
    logger.info(f"Set error rate for '{platform}' to {rate}")


def get_error_rate(platform: str) -> float:
    """Get the error rate for a sequencing platform.

    Retrieves the configured error rate for the given platform,
    falling back to 'unknown' if the platform is not found.

    Args:
        platform: Platform name (e.g., 'pacbio_clr', 'pacbio_hifi', 'ont').

    Returns:
        Error rate as a float.
    """
    return PLATFORM_ERROR_RATES.get(platform, PLATFORM_ERROR_RATES["unknown"])


def strand_bias_test(
    forward_counts: dict[int, int],
    reverse_counts: dict[int, int],
    method: str = "fisher",
) -> float:
    """Test for strand bias using Fisher's exact test.

    Detects whether a variant appears predominantly on one strand,
    which may indicate a sequencing artifact rather than a true variant.
    True biological variants should appear on both forward and reverse
    strands at similar frequencies.

    Uses a 2x2 contingency table:
    - [ref_fwd, alt_fwd] vs [ref_rev, alt_rev]

    Args:
        forward_counts: Dict mapping allele (int) -> count on forward strand.
        reverse_counts: Dict mapping allele (int) -> count on reverse strand.
        method: Statistical test to use. Currently only "fisher" is supported.

    Returns:
        P-value from Fisher's exact test. Low p-value (< 0.01) indicates
        strand bias (likely artifact). High p-value indicates no bias.
    """
    if method != "fisher":
        raise ValueError(f"Unsupported strand bias test method: {method}")

    # Get all alleles present
    alleles = set(forward_counts.keys()) | set(reverse_counts.keys())
    if len(alleles) == 0:
        return 1.0

    # Identify reference (major) and alternative (minor) alleles
    # Reference is the most abundant overall
    total_fwd = sum(forward_counts.values())
    total_rev = sum(reverse_counts.values())
    all_counts = forward_counts.copy()
    for allele, count in reverse_counts.items():
        all_counts[allele] = all_counts.get(allele, 0) + count

    if not all_counts:
        return 1.0

    ref_allele = max(all_counts, key=all_counts.get)

    # Sum counts for reference and all alternatives
    ref_fwd = forward_counts.get(ref_allele, 0)
    ref_rev = reverse_counts.get(ref_allele, 0)

    alt_fwd = total_fwd - ref_fwd
    alt_rev = total_rev - ref_rev

    # Handle edge cases
    if alt_fwd + alt_rev == 0:
        return 1.0

    # Fisher's exact test on 2x2 contingency table
    # [[ref_fwd, alt_fwd], [ref_rev, alt_rev]]
    oddsratio, pvalue = sp_stats.fisher_exact(
        [[ref_fwd, alt_fwd], [ref_rev, alt_rev]], alternative="two-sided"
    )

    return float(pvalue)


class StrandBiasInfo(NamedTuple):
    """Strand bias test results for a variant position."""

    forward_count: int
    reverse_count: int
    bias_pvalue: float


@dataclass
class VariantPosition:
    """A position identified as truly polymorphic."""

    position: int
    major_allele: int  # Base encoding
    minor_alleles: list[int]
    major_freq: float
    minor_freqs: list[float]
    coverage: int
    p_value: float
    q_value: float  # FDR-adjusted
    entropy: float
    strand_bias: StrandBiasInfo | None = None  # Optional strand bias info


def identify_variant_positions(
    msa: np.ndarray,
    is_main: np.ndarray,
    platform: str = "pacbio_clr",
    fdr_threshold: float = 0.05,
    min_coverage: int = 5,
    min_minor_count: int = 2,
    min_minor_freq: float = 0.0,
    min_entropy: float = 0.0,
    custom_error_rate: float | None = None,
    strand_labels: np.ndarray | None = None,
    strand_bias_threshold: float = 0.01,
) -> list[VariantPosition]:
    """Identify true variant positions using statistical testing.

    For each main position in the corrected MSA:
    1. Compute allele frequencies
    2. Apply minimum minor allele frequency (MAF) filter
    3. Test if minor allele frequency exceeds error rate (binomial test)
    4. Correct for multiple testing (Benjamini-Hochberg FDR)
    5. Apply minimum Shannon entropy filter
    6. If strand_labels provided, apply strand bias filter (optional)

    The MAF and entropy filters are critical for real sequencing data
    where residual errors can pass the binomial test at many positions.
    True biological variant positions have higher MAF and entropy than
    noise positions — these filters exploit that separation.

    Strand bias filtering reduces false positives from sequencing artifacts
    that appear predominantly on one strand.

    Args:
        msa: Corrected MSA matrix (n_reads x n_positions).
        is_main: Boolean array indicating main positions.
        platform: Sequencing platform for error rate estimation.
        fdr_threshold: FDR threshold for significance.
        min_coverage: Minimum read depth at a position.
        min_minor_count: Minimum minor allele count.
        min_minor_freq: Minimum minor allele frequency. If 0, auto-set
            from platform: CLR=0.10, HiFi=0.02, ONT=0.10.
        min_entropy: Minimum Shannon entropy. If 0, auto-set from
            platform: CLR=0.40, HiFi=0.10, ONT=0.40.
        custom_error_rate: Override platform error rate.
        strand_labels: 1D array of length n_reads where 0=forward strand,
            1=reverse strand. If None, strand filtering is skipped.
        strand_bias_threshold: P-value threshold for strand bias test.
            Variants with bias p-value < threshold are filtered out.
            Default: 0.01 (1% significance level).

    Returns:
        List of VariantPosition objects for significant positions.
    """
    error_rate = custom_error_rate or get_error_rate(platform)
    n_reads, n_positions = msa.shape
    main_indices = np.where(is_main)[0]

    # Auto-set MAF and entropy thresholds based on platform noise floor
    if min_minor_freq == 0.0:
        _maf_defaults = {
            "pacbio_clr": 0.10,
            "pacbio_hifi": 0.02,
            "ont": 0.10,
            "unknown": 0.10,
        }
        min_minor_freq = _maf_defaults.get(platform, 0.10)

    if min_entropy == 0.0:
        _entropy_defaults = {
            "pacbio_clr": 0.40,
            "pacbio_hifi": 0.10,
            "ont": 0.40,
            "unknown": 0.40,
        }
        min_entropy = _entropy_defaults.get(platform, 0.40)

    # Validate strand_labels if provided
    if strand_labels is not None:
        if len(strand_labels) != msa.shape[0]:
            raise ValueError(
                f"strand_labels length ({len(strand_labels)}) must match "
                f"number of reads ({msa.shape[0]})"
            )

    logger.info(
        f"Testing {len(main_indices)} main positions for true variants "
        f"(error_rate={error_rate}, FDR={fdr_threshold}, "
        f"min_MAF={min_minor_freq}, min_entropy={min_entropy})"
    )
    if strand_labels is not None:
        logger.info(f"Applying strand bias filter (threshold={strand_bias_threshold})")

    candidates = []
    p_values = []
    strand_filtered_count = 0

    for pos in main_indices:
        col = msa[:, pos]
        bases = col[col > 0]  # Non-gap
        coverage = len(bases)

        if coverage < min_coverage:
            continue

        # Count each base
        counts = np.bincount(bases, minlength=6)[1:5]  # A, C, G, T
        total = counts.sum()
        if total == 0:
            continue

        # Sort by frequency
        sorted_idx = np.argsort(counts)[::-1]
        major_idx = sorted_idx[0]
        major_count = counts[major_idx]
        major_freq = major_count / total

        # Collect minor alleles
        minor_alleles = []
        minor_freqs = []
        minor_total = 0

        for idx in sorted_idx[1:]:
            if counts[idx] >= min_minor_count:
                minor_alleles.append(int(idx + 1))  # +1 for base encoding
                minor_freqs.append(counts[idx] / total)
                minor_total += counts[idx]

        if not minor_alleles:
            continue

        # --- FILTER 1: Minimum minor allele frequency ---
        overall_maf = minor_total / total
        if overall_maf < min_minor_freq:
            continue

        # Compute entropy
        freqs = counts / total
        freqs = freqs[freqs > 0]
        entropy = -np.sum(freqs * np.log2(freqs))

        # --- FILTER 2: Minimum Shannon entropy ---
        if entropy < min_entropy:
            continue

        # Binomial test: is the minor allele count significantly
        # higher than expected from sequencing error alone?
        # H0: minor_total arose from errors at rate `error_rate`
        p_val = sp_stats.binom_test(
            minor_total, total, error_rate, alternative="greater"
        ) if hasattr(sp_stats, 'binom_test') else sp_stats.binomtest(
            minor_total, total, error_rate, alternative="greater"
        ).pvalue

        # --- FILTER 3: Strand bias filter (optional) ---
        strand_bias_info = None
        if strand_labels is not None:
            # Get reads at this position
            reads_at_pos = np.where(col > 0)[0]
            strands_at_pos = strand_labels[reads_at_pos]
            bases_at_pos = col[reads_at_pos]

            # Separate forward and reverse strand counts by allele
            fwd_mask = strands_at_pos == 0
            rev_mask = strands_at_pos == 1

            fwd_counts = {}
            rev_counts = {}

            for allele in set(bases_at_pos):
                if allele > 0:
                    fwd_counts[allele] = np.sum((bases_at_pos == allele) & fwd_mask)
                    rev_counts[allele] = np.sum((bases_at_pos == allele) & rev_mask)

            # Test for strand bias
            bias_pvalue = strand_bias_test(fwd_counts, rev_counts)
            strand_bias_info = StrandBiasInfo(
                forward_count=np.sum(fwd_mask),
                reverse_count=np.sum(rev_mask),
                bias_pvalue=bias_pvalue,
            )

            # Filter out strand-biased variants
            if bias_pvalue < strand_bias_threshold:
                strand_filtered_count += 1
                continue

        candidates.append(
            VariantPosition(
                position=int(pos),
                major_allele=int(major_idx + 1),
                minor_alleles=minor_alleles,
                major_freq=float(major_freq),
                minor_freqs=minor_freqs,
                coverage=int(coverage),
                p_value=float(p_val),
                q_value=0.0,  # Filled in after FDR correction
                entropy=float(entropy),
                strand_bias=strand_bias_info,
            )
        )
        p_values.append(p_val)

    if not candidates:
        logger.info("No variant positions identified")
        return []

    # Benjamini-Hochberg FDR correction
    p_values = np.array(p_values)
    q_values = benjamini_hochberg(p_values)

    # Apply FDR threshold
    significant = []
    for cand, qval in zip(candidates, q_values):
        cand.q_value = float(qval)
        if qval <= fdr_threshold:
            significant.append(cand)

    if strand_labels is not None:
        logger.info(
            f"Identified {len(significant)} variant positions "
            f"out of {len(candidates)} candidates "
            f"(FDR <= {fdr_threshold}, strand_filtered={strand_filtered_count})"
        )
    else:
        logger.info(
            f"Identified {len(significant)} variant positions "
            f"out of {len(candidates)} candidates "
            f"(FDR <= {fdr_threshold})"
        )
    return significant


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    Benjamini & Hochberg (1995) "Controlling the false discovery rate"
    J. Roy. Stat. Soc. Ser. B, 57, 289-300.

    Args:
        p_values: Array of p-values.

    Returns:
        Array of FDR-adjusted q-values.
    """
    n = len(p_values)
    if n == 0:
        return np.array([])

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH adjustment: q_i = p_i * n / rank_i
    ranks = np.arange(1, n + 1)
    q_sorted = sorted_p * n / ranks

    # Enforce monotonicity (cumulative minimum from the end)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    # Restore original order
    q_values = np.empty(n)
    q_values[sorted_idx] = q_sorted

    return q_values


def identify_variant_positions_classic(
    msa: np.ndarray,
    is_main: np.ndarray,
    error_rate: float = 0.05,
    fdr_threshold: float = 0.2,
    min_coverage: int = 5,
) -> list[VariantPosition]:
    """Permissive variant detection for use with downstream validation.

    Uses more permissive thresholds than the standard pipeline:
    - Error rate: 5%
    - FDR threshold: q < 0.2 (more permissive than 0.05)
    - Rationale: downstream bootstrap validation (ECA) post-processes
      results, so initial variant detection can be more liberal. False
      positives are cleaned up during correction phases.

    Best used when ECA validation will subsequently filter corrections.

    Args:
        msa: Corrected MSA matrix (n_reads x n_positions).
        is_main: Boolean array indicating main positions.
        error_rate: Platform error rate (default 5%).
        fdr_threshold: FDR threshold (default 0.2).
        min_coverage: Minimum read depth at a position.

    Returns:
        List of VariantPosition objects using permissive thresholds.
    """
    return identify_variant_positions(
        msa=msa,
        is_main=is_main,
        platform="pacbio_clr",
        fdr_threshold=fdr_threshold,
        min_coverage=min_coverage,
        min_minor_count=2,
        min_minor_freq=0.0,
        min_entropy=0.0,
        custom_error_rate=error_rate,
        strand_labels=None,
    )


# Backward compatibility alias
identify_variant_positions_dilernia = identify_variant_positions_classic


def build_variant_matrix(
    msa: np.ndarray,
    variant_positions: list[VariantPosition],
) -> np.ndarray:
    """Extract the variant-only submatrix from the MSA.

    This creates a reduced matrix containing only the columns
    corresponding to true variant positions. This matrix is used
    for downstream clustering/haplotype assignment.

    Args:
        msa: Full corrected MSA matrix.
        variant_positions: List of identified variant positions.

    Returns:
        Reduced MSA matrix (n_reads x n_variant_positions).
    """
    if not variant_positions:
        return np.zeros((msa.shape[0], 0), dtype=np.uint8)

    pos_indices = [vp.position for vp in variant_positions]
    variant_matrix = msa[:, pos_indices]

    logger.info(
        f"Built variant matrix: {variant_matrix.shape[0]} reads x "
        f"{variant_matrix.shape[1]} variant positions"
    )
    return variant_matrix
