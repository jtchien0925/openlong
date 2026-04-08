"""Read clustering for haplotype deconvolution.

Implements the iterative clustering approach from Dilernia et al. 2015
for assigning reads to distinct haplotype groups.

The paper describes an iterative process:
1. Build variant matrix from true variant positions
2. Cluster reads based on their variant profiles
3. Build consensus for each cluster
4. Repeat: re-examine each cluster for further sub-structure
5. Continue until no more sub-clusters are found

This implements both the paper's iterative approach and modern
clustering methods (hierarchical, spectral) for comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

# Base encoding
GAP = 0


@dataclass
class HaplotypeCluster:
    """A cluster of reads assigned to a single haplotype."""

    cluster_id: int
    read_indices: list[int]
    consensus: str = ""
    quality: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_qv: float = 0.0
    n_reads: int = 0
    variant_profile: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        self.n_reads = len(self.read_indices)


def hamming_distance_matrix(
    variant_matrix: np.ndarray,
    ignore_gaps: bool = True,
) -> np.ndarray:
    """Compute pairwise Hamming distance between reads.

    Only considers positions where both reads have non-gap bases.

    Args:
        variant_matrix: Variant-only MSA (n_reads x n_variants).
        ignore_gaps: If True, only count positions where both reads
            have non-gap bases.

    Returns:
        Symmetric distance matrix (n_reads x n_reads).
    """
    n_reads = variant_matrix.shape[0]
    dist = np.zeros((n_reads, n_reads))

    for i in range(n_reads):
        for j in range(i + 1, n_reads):
            ri = variant_matrix[i]
            rj = variant_matrix[j]

            if ignore_gaps:
                # Only compare positions where both have bases
                mask = (ri > 0) & (rj > 0)
                if mask.sum() == 0:
                    dist[i, j] = 1.0
                    dist[j, i] = 1.0
                    continue
                mismatches = np.sum(ri[mask] != rj[mask])
                d = mismatches / mask.sum()
            else:
                d = np.sum(ri != rj) / len(ri)

            dist[i, j] = d
            dist[j, i] = d

    return dist


def cluster_reads_hierarchical(
    variant_matrix: np.ndarray,
    distance_threshold: float = 0.05,
    method: str = "average",
    min_cluster_size: int = 2,
) -> list[HaplotypeCluster]:
    """Cluster reads using hierarchical clustering on Hamming distance.

    This implements the core deconvolution step: reads with similar
    variant profiles are grouped together as belonging to the same
    haplotype.

    Args:
        variant_matrix: Variant-only MSA (n_reads x n_variants).
        distance_threshold: Maximum distance for merging clusters.
        method: Linkage method ('average', 'complete', 'single').
        min_cluster_size: Minimum reads per cluster.

    Returns:
        List of HaplotypeCluster objects.
    """
    n_reads, n_variants = variant_matrix.shape

    if n_reads < 2:
        return [HaplotypeCluster(cluster_id=0, read_indices=list(range(n_reads)))]

    if n_variants == 0:
        return [HaplotypeCluster(cluster_id=0, read_indices=list(range(n_reads)))]

    # Compute distance matrix
    dist_matrix = hamming_distance_matrix(variant_matrix)
    condensed = squareform(dist_matrix)

    # Handle case where all distances are 0
    if np.all(condensed == 0):
        condensed = condensed + 1e-10

    # Hierarchical clustering
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # Build cluster objects
    unique_labels = np.unique(labels)
    clusters = []

    for label in unique_labels:
        read_indices = np.where(labels == label)[0].tolist()
        if len(read_indices) >= min_cluster_size:
            cluster = HaplotypeCluster(
                cluster_id=int(label),
                read_indices=read_indices,
                variant_profile=np.median(
                    variant_matrix[read_indices], axis=0
                ).astype(np.uint8),
            )
            clusters.append(cluster)

    logger.info(
        f"Hierarchical clustering: {n_reads} reads -> {len(clusters)} clusters "
        f"(threshold={distance_threshold})"
    )
    return clusters


def iterative_deconvolution(
    msa: np.ndarray,
    variant_matrix: np.ndarray,
    max_depth: int = 5,
    distance_threshold: float = 0.05,
    min_cluster_size: int = 2,
    min_variants_for_split: int = 2,
) -> list[HaplotypeCluster]:
    """Iterative deconvolution as described in Dilernia et al. 2015.

    The paper describes a recursive process:
    1. Cluster all reads
    2. For each cluster, check if it can be further split
    3. Re-examine variant positions within each cluster
    4. If new variant positions emerge, split the cluster
    5. Repeat until convergence

    Args:
        msa: Full corrected MSA matrix.
        variant_matrix: Variant-only matrix.
        max_depth: Maximum recursion depth.
        distance_threshold: Clustering threshold.
        min_cluster_size: Minimum reads per final cluster.
        min_variants_for_split: Minimum new variants to justify a split.

    Returns:
        List of final HaplotypeCluster objects.
    """
    from openlong.correct.indel import classify_positions, compute_position_entropy
    from openlong.deconv.positions import identify_variant_positions

    def _recurse(
        read_indices: list[int],
        depth: int,
        parent_id: int,
    ) -> list[HaplotypeCluster]:
        if depth >= max_depth or len(read_indices) < 2 * min_cluster_size:
            return [
                HaplotypeCluster(
                    cluster_id=parent_id, read_indices=read_indices
                )
            ]

        # Extract sub-MSA for this cluster's reads
        sub_msa = msa[read_indices]
        sub_variant = variant_matrix[read_indices]

        # Try clustering
        sub_clusters = cluster_reads_hierarchical(
            sub_variant,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
        )

        if len(sub_clusters) <= 1:
            return [
                HaplotypeCluster(
                    cluster_id=parent_id, read_indices=read_indices
                )
            ]

        # For each sub-cluster, check for further structure
        final_clusters = []
        for i, sc in enumerate(sub_clusters):
            # Map back to original read indices
            original_indices = [read_indices[j] for j in sc.read_indices]
            child_id = parent_id * 10 + i + 1

            # Check if sub-cluster has internal structure
            sub_sub_msa = msa[original_indices]
            sub_is_main = classify_positions(sub_sub_msa)
            sub_entropy = compute_position_entropy(sub_sub_msa, sub_is_main)
            n_high_entropy = np.sum(sub_entropy > 0.5)

            if n_high_entropy >= min_variants_for_split:
                # Recurse
                child_clusters = _recurse(original_indices, depth + 1, child_id)
                final_clusters.extend(child_clusters)
            else:
                final_clusters.append(
                    HaplotypeCluster(
                        cluster_id=child_id,
                        read_indices=original_indices,
                    )
                )

        return final_clusters

    all_indices = list(range(msa.shape[0]))
    clusters = _recurse(all_indices, depth=0, parent_id=1)

    # Renumber clusters sequentially
    for i, cluster in enumerate(clusters):
        cluster.cluster_id = i + 1

    logger.info(
        f"Iterative deconvolution: {msa.shape[0]} reads -> "
        f"{len(clusters)} haplotypes"
    )
    return clusters


def estimate_haplotype_frequencies(
    clusters: list[HaplotypeCluster],
    total_reads: int | None = None,
) -> dict[int, float]:
    """Estimate the frequency of each haplotype in the mixture.

    Args:
        clusters: List of haplotype clusters.
        total_reads: Total number of reads (for normalization).

    Returns:
        Dictionary mapping cluster_id to estimated frequency.
    """
    if total_reads is None:
        total_reads = sum(c.n_reads for c in clusters)

    freqs = {}
    for cluster in clusters:
        freqs[cluster.cluster_id] = cluster.n_reads / max(total_reads, 1)

    return freqs
