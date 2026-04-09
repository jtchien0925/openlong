"""Consensus polishing module.

After INDEL correction, builds polished consensus sequences from
corrected MSA columns using majority voting and quality weighting.

Includes iterative polishing and within-cluster error correction
for high-accuracy consensus building (targeting QV50+).
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


def build_quality_weighted_consensus(
    msa: np.ndarray,
    quality_matrix: np.ndarray | None = None,
    is_main: np.ndarray | None = None,
    min_coverage: int = 3,
    min_agreement: float = 0.6,
) -> tuple[str, np.ndarray]:
    """Build quality-weighted consensus and compute confidence scores.

    Instead of simple majority vote, weight each base by its quality score.
    Higher quality bases get more influence on the consensus call.

    Args:
        msa: Corrected MSA matrix (n_reads x n_positions).
        quality_matrix: Quality scores matrix (same shape as msa).
            If None, falls back to majority consensus.
        is_main: Position classification.
        min_coverage: Minimum number of non-gap reads at a position.
        min_agreement: Minimum weighted agreement threshold.

    Returns:
        Tuple of (consensus sequence, confidence array as Phred QV values).
    """
    n_reads, n_positions = msa.shape
    consensus = []
    confidence = []

    for pos in range(n_positions):
        if is_main is not None and not is_main[pos]:
            continue

        col = msa[:, pos]
        mask = col > 0  # Non-gap bases
        bases = col[mask]

        if len(bases) < min_coverage:
            consensus.append("N")
            confidence.append(0.0)
            continue

        # Get quality scores for this position
        if quality_matrix is not None:
            qual_col = quality_matrix[:, pos]
            quals = qual_col[mask]
            # Convert Phred scores to error probabilities for weighting
            weights = 10.0 ** (-quals / 10.0)
            weights = 1.0 / (weights + 1e-10)  # Invert: higher quality = higher weight
        else:
            # Uniform weights if no quality matrix
            weights = np.ones(len(bases))

        # Weighted vote: sum weights per base type
        weighted_counts = np.zeros(5)  # Index 0-4 for A,C,G,T, but we use 1-4
        for base, weight in zip(bases, weights):
            if 1 <= base <= 4:
                weighted_counts[base - 1] += weight

        # Find best base
        best_idx = np.argmax(weighted_counts)
        best_base = best_idx + 1  # Convert to 1-indexed
        best_weight = weighted_counts[best_idx]
        total_weight = np.sum(weighted_counts)

        if total_weight == 0:
            consensus.append("N")
            confidence.append(0.0)
            continue

        # Compute weighted agreement
        weighted_agreement = best_weight / total_weight if total_weight > 0 else 0

        if weighted_agreement >= min_agreement:
            consensus.append(DECODE[best_base])
            # Convert agreement to Phred quality
            error_rate = max(1 - weighted_agreement, 1e-6)
            qv = -10 * np.log10(error_rate)
            confidence.append(min(qv, 93))
        else:
            consensus.append("N")
            confidence.append(0.0)

    return "".join(consensus), np.array(confidence)


def compute_entropy(col: np.ndarray) -> float:
    """Compute Shannon entropy of base distribution at a position.

    Args:
        col: Base encoding array.

    Returns:
        Entropy value (bits).
    """
    bases = col[col > 0]
    if len(bases) == 0:
        return 0.0

    counts = np.bincount(bases[bases < 5], minlength=4)  # A,C,G,T only
    total = np.sum(counts)
    if total == 0:
        return 0.0

    probs = counts / total
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    return entropy


def correct_rare_bases(
    msa: np.ndarray,
    consensus: str,
    rare_threshold: float = 0.05,
) -> np.ndarray:
    """Within-cluster error correction: replace rare bases with consensus.

    For each position in each read, if a base appears in <rare_threshold
    of reads, replace it with the consensus base. This removes sequencing
    errors that survived earlier correction stages.

    Args:
        msa: MSA matrix.
        consensus: Consensus sequence string.
        rare_threshold: Fraction threshold below which a base is considered rare.

    Returns:
        Corrected MSA matrix.
    """
    msa_corrected = msa.copy()
    n_reads, n_positions = msa.shape

    # Map consensus sequence back to MSA positions
    cons_idx = 0
    for pos in range(n_positions):
        if cons_idx >= len(consensus):
            break

        col = msa[:, pos]
        bases = col[col > 0]

        if len(bases) == 0:
            continue

        # Count base frequencies
        counts = np.bincount(bases, minlength=6)
        total = np.sum(bases > 0)

        if total == 0:
            continue

        # Find rare bases and correct them
        consensus_base = ENCODE.get(consensus[cons_idx], 0)
        if consensus_base == 0:
            cons_idx += 1
            continue

        for base in [1, 2, 3, 4]:  # A, C, G, T
            if base != consensus_base:
                freq = counts[base] / total if total > 0 else 0
                if 0 < freq < rare_threshold:
                    # Replace rare occurrences with consensus
                    rare_mask = msa[:, pos] == base
                    msa_corrected[rare_mask, pos] = consensus_base

        cons_idx += 1

    return msa_corrected


def polish_consensus(
    msa: np.ndarray,
    initial_consensus: str,
    n_rounds: int = 3,
    rare_threshold: float = 0.05,
) -> str:
    """Iteratively polish consensus by correcting rare variants.

    After initial consensus building, realign reads to consensus and
    rebuild. Each round reduces errors because the consensus is a better
    reference than the original.

    Algorithm:
    - Round 1: Build consensus from MSA (input)
    - Round 2+: For each position, if a base is rare (<threshold),
      correct it to consensus, then rebuild consensus

    Args:
        msa: MSA matrix.
        initial_consensus: Initial consensus sequence.
        n_rounds: Number of polishing rounds.
        rare_threshold: Fraction threshold for rare bases.

    Returns:
        Polished consensus sequence.
    """
    consensus = initial_consensus
    msa_working = msa.copy()

    for round_num in range(n_rounds):
        # Correct rare bases based on current consensus
        msa_corrected = correct_rare_bases(
            msa_working, consensus, rare_threshold
        )

        # Rebuild consensus from corrected MSA
        consensus_new = majority_consensus(msa_corrected)

        logger.info(
            f"Polish round {round_num + 1}: "
            f"{len(consensus)} -> {len(consensus_new)} bp"
        )

        # Check for convergence
        if consensus_new == consensus:
            logger.info(f"Consensus converged after {round_num + 1} rounds")
            break

        consensus = consensus_new
        msa_working = msa_corrected

    return consensus


def compute_position_entropy(
    msa: np.ndarray,
) -> np.ndarray:
    """Compute entropy of base distribution at each position.

    High entropy indicates ambiguous/disagreement, low entropy indicates
    high confidence consensus positions.

    Args:
        msa: MSA matrix.

    Returns:
        Array of entropy values per position.
    """
    n_reads, n_positions = msa.shape
    entropies = np.zeros(n_positions)

    for pos in range(n_positions):
        entropies[pos] = compute_entropy(msa[:, pos])

    return entropies
