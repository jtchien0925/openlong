"""Per-cluster consensus sequence building.

After reads are clustered into haplotype groups, this module builds
the final consensus sequence for each haplotype with quality scores.
"""

from __future__ import annotations

import logging

import numpy as np

from openlong.correct.polish import (
    DECODE,
    build_quality_weighted_consensus,
    compute_consensus_quality,
    compute_position_entropy,
    majority_consensus,
    polish_consensus,
    weighted_consensus,
)
from openlong.deconv.cluster import HaplotypeCluster
from openlong.deconv.prob_consensus import (
    probabilistic_consensus,
    dirichlet_consensus,
)

logger = logging.getLogger(__name__)


def build_cluster_consensus(
    msa: np.ndarray,
    cluster: HaplotypeCluster,
    is_main: np.ndarray | None = None,
    min_coverage: int = 3,
    min_agreement: float = 0.6,
    use_quality_weighting: bool = True,
    n_polish_rounds: int = 3,
    consensus_method: str = "quality_weighted",  # 'majority', 'quality_weighted', 'probabilistic', 'dirichlet'
    error_model: str = "pacbio_clr",  # For probabilistic method
) -> HaplotypeCluster:
    """Build consensus sequence for a single cluster.

    Can use quality-weighted consensus, probabilistic (Quiver-equivalent),
    or Dirichlet Bayesian methods with iterative polishing to achieve
    high accuracy (targeting QV50+).

    Args:
        msa: Full corrected MSA matrix.
        cluster: The cluster to process.
        is_main: Position classification array.
        min_coverage: Min reads for consensus call.
        min_agreement: Min agreement fraction.
        use_quality_weighting: If True, use quality-weighted consensus (legacy).
        n_polish_rounds: Number of iterative polishing rounds.
        consensus_method: 'majority', 'quality_weighted', 'probabilistic', or 'dirichlet'.
        error_model: Platform error model for probabilistic method.

    Returns:
        Updated cluster with consensus and quality.
    """
    # Extract reads belonging to this cluster
    cluster_msa = msa[cluster.read_indices]

    # Build initial consensus using selected method
    if consensus_method == "probabilistic":
        # Quiver-equivalent probabilistic consensus
        consensus, quality_array = probabilistic_consensus(
            cluster_msa,
            quality_matrix=None,
            error_model=error_model,
            n_iterations=5,
            min_coverage=min_coverage,
        )
        confidence = quality_array

    elif consensus_method == "dirichlet":
        # Bayesian Dirichlet-multinomial (good for low coverage)
        consensus, confidence = dirichlet_consensus(
            cluster_msa,
            alpha=1.0,
            min_coverage=min_coverage,
        )

    elif consensus_method == "quality_weighted":
        # Legacy quality-weighted (falls back to majority if no quality matrix)
        consensus, confidence = build_quality_weighted_consensus(
            cluster_msa,
            quality_matrix=None,
            is_main=is_main,
            min_coverage=min_coverage,
            min_agreement=min_agreement,
        )

    else:  # 'majority' or default
        # Simple majority voting
        consensus = majority_consensus(
            cluster_msa,
            is_main=is_main,
            min_coverage=min_coverage,
            min_agreement=min_agreement,
        )
        confidence = compute_consensus_quality(cluster_msa, consensus, is_main)

    # Iteratively polish consensus (applies only to majority/quality_weighted)
    if n_polish_rounds > 0 and consensus_method in ("majority", "quality_weighted"):
        consensus = polish_consensus(
            cluster_msa,
            consensus,
            n_rounds=n_polish_rounds,
            rare_threshold=0.05,
        )
        # Recompute quality after polishing
        confidence = compute_consensus_quality(cluster_msa, consensus, is_main)

    cluster.consensus = consensus
    cluster.quality = confidence
    cluster.mean_qv = float(np.mean(confidence[confidence > 0])) if np.any(confidence > 0) else 0.0

    logger.info(
        f"Cluster {cluster.cluster_id}: {len(consensus)} bp, "
        f"{cluster.n_reads} reads, mean QV={cluster.mean_qv:.1f} "
        f"(method={consensus_method}, polish_rounds={n_polish_rounds if consensus_method in ('majority', 'quality_weighted') else 'N/A'})"
    )
    return cluster


def build_all_consensus(
    msa: np.ndarray,
    clusters: list[HaplotypeCluster],
    is_main: np.ndarray | None = None,
    min_coverage: int = 3,
    min_agreement: float = 0.6,
    use_quality_weighting: bool = True,
    n_polish_rounds: int = 3,
    consensus_method: str = "quality_weighted",
    error_model: str = "pacbio_clr",
) -> list[HaplotypeCluster]:
    """Build consensus sequences for all clusters.

    Uses specified consensus method with optional iterative polishing for high accuracy.

    Args:
        msa: Full corrected MSA matrix.
        clusters: List of clusters.
        is_main: Position classification.
        min_coverage: Min reads for consensus.
        min_agreement: Min agreement fraction.
        use_quality_weighting: Legacy parameter (deprecated).
        n_polish_rounds: Number of iterative polishing rounds.
        consensus_method: 'majority', 'quality_weighted', 'probabilistic', or 'dirichlet'.
        error_model: Platform error model for probabilistic method.

    Returns:
        List of updated clusters with consensus sequences.
    """
    logger.info(
        f"Building consensus for {len(clusters)} clusters "
        f"(method={consensus_method}, "
        f"polish_rounds={n_polish_rounds})"
    )

    for cluster in clusters:
        build_cluster_consensus(
            msa,
            cluster,
            is_main,
            min_coverage,
            min_agreement,
            use_quality_weighting,
            n_polish_rounds,
            consensus_method,
            error_model,
        )

    # Report summary
    total_bp = sum(len(c.consensus) for c in clusters)
    mean_qv = np.mean([c.mean_qv for c in clusters if c.mean_qv > 0])
    logger.info(
        f"All consensus built: {len(clusters)} haplotypes, "
        f"{total_bp:,} total bp, mean QV={mean_qv:.1f}"
    )
    return clusters


def export_haplotypes(
    clusters: list[HaplotypeCluster],
    prefix: str = "haplotype",
) -> dict[str, str]:
    """Export haplotype consensus sequences as a name->sequence dict.

    Args:
        clusters: Clusters with consensus sequences built.
        prefix: Prefix for sequence names.

    Returns:
        Dictionary of name -> sequence for FASTA output.
    """
    sequences = {}
    for cluster in clusters:
        if not cluster.consensus:
            continue
        name = (
            f"{prefix}_{cluster.cluster_id}"
            f"_reads={cluster.n_reads}"
            f"_qv={cluster.mean_qv:.0f}"
        )
        sequences[name] = cluster.consensus

    return sequences
