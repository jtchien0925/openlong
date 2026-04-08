"""Per-cluster consensus sequence building.

After reads are clustered into haplotype groups, this module builds
the final consensus sequence for each haplotype with quality scores.
"""

from __future__ import annotations

import logging

import numpy as np

from openlong.correct.polish import (
    DECODE,
    compute_consensus_quality,
    majority_consensus,
    weighted_consensus,
)
from openlong.deconv.cluster import HaplotypeCluster

logger = logging.getLogger(__name__)


def build_cluster_consensus(
    msa: np.ndarray,
    cluster: HaplotypeCluster,
    is_main: np.ndarray | None = None,
    min_coverage: int = 3,
    min_agreement: float = 0.6,
) -> HaplotypeCluster:
    """Build consensus sequence for a single cluster.

    Args:
        msa: Full corrected MSA matrix.
        cluster: The cluster to process.
        is_main: Position classification array.
        min_coverage: Min reads for consensus call.
        min_agreement: Min agreement fraction.

    Returns:
        Updated cluster with consensus and quality.
    """
    # Extract reads belonging to this cluster
    cluster_msa = msa[cluster.read_indices]

    # Build consensus
    consensus = majority_consensus(
        cluster_msa,
        is_main=is_main,
        min_coverage=min_coverage,
        min_agreement=min_agreement,
    )

    # Compute quality scores
    quality = compute_consensus_quality(cluster_msa, consensus, is_main)

    cluster.consensus = consensus
    cluster.quality = quality
    cluster.mean_qv = float(np.mean(quality[quality > 0])) if np.any(quality > 0) else 0.0

    logger.info(
        f"Cluster {cluster.cluster_id}: {len(consensus)} bp, "
        f"{cluster.n_reads} reads, mean QV={cluster.mean_qv:.1f}"
    )
    return cluster


def build_all_consensus(
    msa: np.ndarray,
    clusters: list[HaplotypeCluster],
    is_main: np.ndarray | None = None,
    min_coverage: int = 3,
    min_agreement: float = 0.6,
) -> list[HaplotypeCluster]:
    """Build consensus sequences for all clusters.

    Args:
        msa: Full corrected MSA matrix.
        clusters: List of clusters.
        is_main: Position classification.
        min_coverage: Min reads for consensus.
        min_agreement: Min agreement fraction.

    Returns:
        List of updated clusters with consensus sequences.
    """
    logger.info(f"Building consensus for {len(clusters)} clusters")

    for cluster in clusters:
        build_cluster_consensus(
            msa, cluster, is_main, min_coverage, min_agreement
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
