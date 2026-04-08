"""Consensus polishing module.

After INDEL correction, builds polished consensus sequences from
corrected MSA columns using majority voting and quality weighting.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Base encoding
DECODE = {0: "-", 1: "A", 2: "C", 3: "G", 4: "T", 5: "N"}
ENCODE = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5, "-": 0}


def majority_consensus(
    msa: np.ndarray,
    is_main: np.ndarray | None = None,
    min_coverage: int = 3,
    min_agreement: float = 0.6,
) -> str:
    """Build consensus sequence by majority vote at each position.

    Args:
        msa: Corrected MSA matrix (n_reads x n_positions).
        is_main: Position classification. If provided, only main
            positions contribute to consensus.
        min_coverage: Minimum number of non-gap reads at a position.
        min_agreement: Minimum fraction of reads agreeing on the
            consensus base.

    Returns:
        Consensus sequence string.
    """
    n_reads, n_positions = msa.shape
    consensus = []

    for pos in range(n_positions):
        if is_main is not None and not is_main[pos]:
            continue

        col = msa[:, pos]
        bases = col[col > 0]  # Non-gap bases

        if len(bases) < min_coverage:
            consensus.append("N")
            continue

        counts = np.bincount(bases, minlength=6)
        # Only count A, C, G, T
        base_counts = counts[1:5]
        total = base_counts.sum()

        if total == 0:
            consensus.append("N")
            continue

        best_base_idx = np.argmax(base_counts) + 1  # +1 because A=1
        agreement = base_counts[best_base_idx - 1] / total

        if agreement >= min_agreement:
            consensus.append(DECODE[best_base_idx])
        else:
            consensus.append("N")

    return "".join(consensus)


def weighted_consensus(
    msa: np.ndarray,
    quality_matrix: np.ndarray | None = None,
    is_main: np.ndarray | None = None,
    min_coverage: int = 3,
) -> str:
    """Build consensus using quality-weighted voting.

    When quality scores are available (e.g., from PacBio HiFi or ONT),
    weight each base call by its quality score for more accurate consensus.

    Args:
        msa: Corrected MSA matrix.
        quality_matrix: Quality scores matrix (same shape as msa).
            If None, falls back to majority consensus.
        is_main: Position classification.
        min_coverage: Minimum coverage at a position.

    Returns:
        Consensus sequence string.
    """
    if quality_matrix is None:
        return majority_consensus(msa, is_main, min_coverage)

    n_reads, n_positions = msa.shape
    consensus = []

    for pos in range(n_positions):
        if is_main is not None and not is_main[pos]:
            continue

        col = msa[:, pos]
        qual_col = quality_matrix[:, pos]
        mask = col > 0  # Non-gap
        bases = col[mask]
        quals = qual_col[mask]

        if len(bases) < min_coverage:
            consensus.append("N")
            continue

        # Weighted vote: sum quality scores per base type
        weighted_counts = np.zeros(5)  # Index 1-4 for A,C,G,T
        for base, qual in zip(bases, quals):
            if 1 <= base <= 4:
                weighted_counts[base] += qual

        best_base = np.argmax(weighted_counts[1:]) + 1
        if weighted_counts[best_base] > 0:
            consensus.append(DECODE[best_base])
        else:
            consensus.append("N")

    return "".join(consensus)


def compute_consensus_quality(
    msa: np.ndarray,
    consensus_seq: str,
    is_main: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-base quality scores for the consensus.

    Quality is estimated from the agreement level at each position:
    QV = -10 * log10(1 - agreement_fraction)

    This approximates the >QV50 accuracy reported in the paper.

    Args:
        msa: MSA matrix.
        consensus_seq: The consensus sequence.
        is_main: Position classification.

    Returns:
        Array of Phred quality values.
    """
    n_reads, n_positions = msa.shape
    qualities = np.zeros(len(consensus_seq))

    cons_idx = 0
    for pos in range(n_positions):
        if is_main is not None and not is_main[pos]:
            continue
        if cons_idx >= len(consensus_seq):
            break

        col = msa[:, pos]
        bases = col[col > 0]

        if len(bases) < 2:
            qualities[cons_idx] = 0
            cons_idx += 1
            continue

        consensus_base = ENCODE.get(consensus_seq[cons_idx], 0)
        if consensus_base == 0:
            qualities[cons_idx] = 0
            cons_idx += 1
            continue

        agreement = np.sum(bases == consensus_base) / len(bases)
        error_rate = max(1 - agreement, 1e-6)  # Prevent log(0)
        qv = -10 * np.log10(error_rate)
        qualities[cons_idx] = min(qv, 93)  # Cap at QV93

        cons_idx += 1

    mean_qv = np.mean(qualities[qualities > 0]) if np.any(qualities > 0) else 0
    logger.info(
        f"Consensus quality: mean QV={mean_qv:.1f}, "
        f"length={len(consensus_seq)}"
    )
    return qualities
