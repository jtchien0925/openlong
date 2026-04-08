"""True variant position identification.

Implements the statistical method from Dilernia et al. 2015 for
distinguishing true polymorphic positions from sequencing errors
in the corrected MSA.

The key challenge: after INDEL correction, some positions still show
variation. We need to determine which of these represent true biological
variants vs. residual sequencing errors.

Approach:
1. For each position, compute the minor allele frequency (MAF)
2. Apply a binomial test: is the observed MAF significantly higher
   than expected from the platform's residual error rate?
3. Apply multiple testing correction (Benjamini-Hochberg FDR)
4. Positions passing the threshold are declared true variant positions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Base encoding
GAP = 0

# Platform-specific residual error rates (after INDEL correction)
# These defaults can be modified at runtime using set_error_rate()
PLATFORM_ERROR_RATES = {
    "pacbio_clr": 0.02,   # ~2% residual after correction
    "pacbio_hifi": 0.001,  # ~0.1% for HiFi reads
    "ont": 0.03,           # ~3% for ONT R10
    "unknown": 0.02,
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


def identify_variant_positions(
    msa: np.ndarray,
    is_main: np.ndarray,
    platform: str = "pacbio_clr",
    fdr_threshold: float = 0.05,
    min_coverage: int = 5,
    min_minor_count: int = 2,
    custom_error_rate: float | None = None,
) -> list[VariantPosition]:
    """Identify true variant positions using statistical testing.

    For each main position in the corrected MSA:
    1. Compute allele frequencies
    2. Test if minor allele frequency exceeds error rate
    3. Correct for multiple testing

    Args:
        msa: Corrected MSA matrix (n_reads x n_positions).
        is_main: Boolean array indicating main positions.
        platform: Sequencing platform for error rate estimation.
        fdr_threshold: FDR threshold for significance.
        min_coverage: Minimum read depth at a position.
        min_minor_count: Minimum minor allele count.
        custom_error_rate: Override platform error rate.

    Returns:
        List of VariantPosition objects for significant positions.
    """
    error_rate = custom_error_rate or get_error_rate(platform)
    n_reads, n_positions = msa.shape
    main_indices = np.where(is_main)[0]

    logger.info(
        f"Testing {len(main_indices)} main positions for true variants "
        f"(error_rate={error_rate}, FDR={fdr_threshold})"
    )

    candidates = []
    p_values = []

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

        # Binomial test: is the minor allele count significantly
        # higher than expected from sequencing error alone?
        # H0: minor_total arose from errors at rate `error_rate`
        p_val = sp_stats.binom_test(
            minor_total, total, error_rate, alternative="greater"
        ) if hasattr(sp_stats, 'binom_test') else sp_stats.binomtest(
            minor_total, total, error_rate, alternative="greater"
        ).pvalue

        # Compute entropy
        freqs = counts / total
        freqs = freqs[freqs > 0]
        entropy = -np.sum(freqs * np.log2(freqs))

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

    logger.info(
        f"Identified {len(significant)} variant positions "
        f"out of {len(candidates)} candidates "
        f"(FDR <= {fdr_threshold})"
    )
    return significant


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    As referenced in Dilernia et al. 2015, citing:
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
