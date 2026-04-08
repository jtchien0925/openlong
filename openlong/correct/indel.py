"""INDEL correction algorithm.

Implements the alignment correction algorithm from:
    Dilernia et al. (2015) Nucleic Acids Research, 43(20), e129.

The core insight: PacBio CLR reads have a high INDEL error rate (~15%).
This algorithm distinguishes 'main positions' (true genomic positions)
from 'INDEL positions' (sequencing artifacts) in the MSA, then corrects
spurious insertions that would confound downstream variant detection.

Algorithm overview:
1. Build MSA from aligned reads against consensus/reference
2. Classify each position as 'main' (P) or 'INDEL' (p) based on
   occupancy across reads
3. For main positions flanked by INDEL positions, apply the correction:
   if a nucleotide at a main position is surrounded by INDEL-classified
   bases, replace it with a gap (non-informative state)
4. This dramatically reduces false positive variant calls from INDEL errors
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Base encoding: 0=gap, 1=A, 2=C, 3=G, 4=T, 5=N
GAP = 0
BASE_A = 1
BASE_C = 2
BASE_G = 3
BASE_T = 4
BASE_N = 5
BASES = {BASE_A, BASE_C, BASE_G, BASE_T}


class CorrectionStats(NamedTuple):
    """Statistics from the INDEL correction step."""

    total_positions: int
    main_positions: int
    indel_positions: int
    corrections_made: int
    occupancy_threshold: float


def classify_positions(
    msa: np.ndarray,
    occupancy_threshold: float = 0.5,
) -> np.ndarray:
    """Classify MSA positions as 'main' or 'INDEL'.

    A position is classified as 'main' (True) if the fraction of reads
    with a non-gap base at that position exceeds the occupancy threshold.
    Otherwise it is classified as an INDEL position (False).

    This implements the position classification step from Dilernia et al.:
    'main positions P' vs 'INDEL positions p'.

    Args:
        msa: MSA matrix (n_reads x n_positions), uint8 encoded.
        occupancy_threshold: Fraction of reads required to call a position
            as 'main'. Default 0.5.

    Returns:
        Boolean array of length n_positions. True = main, False = INDEL.
    """
    n_reads, n_positions = msa.shape
    if n_reads == 0:
        return np.zeros(n_positions, dtype=bool)

    # Count non-gap bases at each position
    non_gap_counts = np.sum(msa > 0, axis=0)

    # Count reads that actually cover each position (non-zero anywhere nearby)
    # A read covers a position if it has ANY non-gap base
    coverage = np.sum(msa > 0, axis=0)

    # Occupancy = fraction of reads with a base at this position
    # We normalize by the maximum coverage to handle variable read depths
    max_coverage = np.max(non_gap_counts) if np.max(non_gap_counts) > 0 else 1
    occupancy = non_gap_counts / max_coverage

    is_main = occupancy >= occupancy_threshold

    logger.debug(
        f"Position classification: {np.sum(is_main)} main, "
        f"{np.sum(~is_main)} INDEL out of {n_positions} total"
    )
    return is_main


def correct_indels(
    msa: np.ndarray,
    is_main: np.ndarray | None = None,
    occupancy_threshold: float = 0.5,
) -> tuple[np.ndarray, CorrectionStats]:
    """Apply the INDEL correction algorithm from Dilernia et al. 2015.

    For each read, at each main position P_x:
    - Let P_y be the next downstream main position
    - Let P_z be the next upstream main position
    - If any INDEL position between P_y and P_x, or between P_x and P_z,
      contains a nucleotide (not a gap), then replace the base at P_x
      with a gap (non-informative state)

    This correction removes false variants caused by INDEL errors that
    shift the alignment and create apparent substitutions at neighboring
    main positions.

    Args:
        msa: MSA matrix (n_reads x n_positions), uint8 encoded.
        is_main: Pre-computed position classification. If None, computed
            using classify_positions().
        occupancy_threshold: Threshold for position classification.

    Returns:
        Tuple of (corrected MSA, CorrectionStats).
    """
    n_reads, n_positions = msa.shape
    if n_reads == 0 or n_positions == 0:
        return msa.copy(), CorrectionStats(0, 0, 0, 0, occupancy_threshold)

    if is_main is None:
        is_main = classify_positions(msa, occupancy_threshold)

    corrected = msa.copy()
    corrections = 0

    # Get indices of main positions
    main_indices = np.where(is_main)[0]

    if len(main_indices) < 2:
        return corrected, CorrectionStats(
            n_positions, len(main_indices), n_positions - len(main_indices), 0, occupancy_threshold
        )

    # For each read, check each main position
    for read_idx in range(n_reads):
        row = corrected[read_idx]

        for mi in range(len(main_indices)):
            x = main_indices[mi]
            base_at_x = row[x]

            if base_at_x == GAP:
                continue  # Nothing to correct

            # Check downstream: between previous main position (y) and x
            has_indel_artifact = False

            if mi > 0:
                y = main_indices[mi - 1]
                # Check INDEL positions between y and x
                for p in range(y + 1, x):
                    if not is_main[p] and row[p] in BASES:
                        has_indel_artifact = True
                        break

            # Check upstream: between x and next main position (z)
            if not has_indel_artifact and mi < len(main_indices) - 1:
                z = main_indices[mi + 1]
                for p in range(x + 1, z):
                    if not is_main[p] and row[p] in BASES:
                        has_indel_artifact = True
                        break

            if has_indel_artifact:
                corrected[read_idx, x] = GAP
                corrections += 1

    stats = CorrectionStats(
        total_positions=n_positions,
        main_positions=int(np.sum(is_main)),
        indel_positions=int(np.sum(~is_main)),
        corrections_made=corrections,
        occupancy_threshold=occupancy_threshold,
    )

    logger.info(
        f"INDEL correction: {corrections} corrections across "
        f"{n_reads} reads x {n_positions} positions "
        f"({stats.main_positions} main, {stats.indel_positions} INDEL)"
    )
    return corrected, stats


def iterative_indel_correction(
    msa: np.ndarray,
    max_iterations: int = 5,
    occupancy_threshold: float = 0.5,
    convergence_threshold: float = 0.001,
) -> tuple[np.ndarray, list[CorrectionStats]]:
    """Apply INDEL correction iteratively until convergence.

    The paper describes iterative refinement where after each correction
    pass, positions are **re-classified** (occupancies change after masking)
    and correction is applied again. This is critical because:
    1. First pass corrects the most obvious artifacts
    2. Re-classification may reclassify borderline positions
    3. Second pass catches artifacts exposed by the first correction

    Convergence threshold is set tight (0.1% of elements) to ensure
    thorough correction on real data. The original 1% was too loose
    and caused premature convergence after a single iteration.

    Args:
        msa: Input MSA matrix.
        max_iterations: Maximum number of correction iterations.
        occupancy_threshold: Position classification threshold.
        convergence_threshold: Stop if fraction of new corrections
            drops below this (default 0.001 = 0.1% of elements).

    Returns:
        Tuple of (final corrected MSA, list of stats per iteration).
    """
    current = msa.copy()
    all_stats = []

    for iteration in range(max_iterations):
        # Re-classify positions each iteration — this is the key insight
        # from the paper. After corrections, occupancies shift and some
        # positions that were borderline INDEL may become clearly main
        # or vice versa.
        is_main = classify_positions(current, occupancy_threshold)
        corrected, stats = correct_indels(
            current, is_main=is_main, occupancy_threshold=occupancy_threshold
        )
        all_stats.append(stats)

        total_elements = current.shape[0] * current.shape[1]
        correction_fraction = stats.corrections_made / max(total_elements, 1)

        logger.info(
            f"Iteration {iteration + 1}: {stats.corrections_made} corrections "
            f"({correction_fraction:.4%} of elements), "
            f"{stats.main_positions} main / {stats.indel_positions} INDEL positions"
        )

        if correction_fraction < convergence_threshold:
            logger.info(f"Converged after {iteration + 1} iterations")
            break

        current = corrected

    return current, all_stats


def compute_position_entropy(
    msa: np.ndarray,
    is_main: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Shannon entropy at each position in the MSA.

    High entropy positions are candidates for true variant positions.
    Low entropy positions are likely homogeneous (same base across reads).

    Args:
        msa: MSA matrix.
        is_main: Optional position classification to restrict analysis.

    Returns:
        Array of entropy values per position.
    """
    n_reads, n_positions = msa.shape
    entropies = np.zeros(n_positions)

    for pos in range(n_positions):
        if is_main is not None and not is_main[pos]:
            continue

        col = msa[:, pos]
        # Only consider non-gap bases
        bases = col[col > 0]

        if len(bases) < 2:
            continue

        # Count each base type
        counts = np.bincount(bases, minlength=6)[1:5]  # A, C, G, T
        total = counts.sum()
        if total == 0:
            continue

        probs = counts / total
        probs = probs[probs > 0]
        entropies[pos] = -np.sum(probs * np.log2(probs))

    return entropies
