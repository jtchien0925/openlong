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
from sklearn.metrics import silhouette_score

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
    min_shared_positions: int = 10,
) -> np.ndarray:
    """Compute pairwise Hamming distance between reads.

    Only considers positions where both reads have non-gap bases.
    Pairs with too few shared positions are assigned maximum distance
    to prevent spurious clustering from noisy overlap.

    Args:
        variant_matrix: Variant-only MSA (n_reads x n_variants).
        ignore_gaps: If True, only count positions where both reads
            have non-gap bases.
        min_shared_positions: Minimum number of shared non-gap positions
            for a meaningful distance. Pairs below this get distance 1.0.

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
                shared = mask.sum()
                if shared < min_shared_positions:
                    dist[i, j] = 1.0
                    dist[j, i] = 1.0
                    continue
                mismatches = np.sum(ri[mask] != rj[mask])
                d = mismatches / shared
            else:
                d = np.sum(ri != rj) / len(ri)

            dist[i, j] = d
            dist[j, i] = d

    return dist


def impute_gaps_with_consensus(
    variant_matrix: np.ndarray,
) -> np.ndarray:
    """Impute gap positions with the per-column consensus base.

    Gap-heavy variant matrices corrupt Hamming distances because
    the distance is only computed over shared non-gap positions,
    which may be a biased subset. Imputation fills gaps with the
    most common base at that position, reducing the noise introduced
    by variable read coverage.

    Only applied to the variant matrix (not the full MSA) and only
    for the purpose of distance computation.

    Args:
        variant_matrix: Variant-only MSA (n_reads x n_variants).

    Returns:
        Copy with gaps replaced by column consensus.
    """
    imputed = variant_matrix.copy()
    n_reads, n_pos = imputed.shape

    for col in range(n_pos):
        column = imputed[:, col]
        bases = column[column > 0]
        if len(bases) == 0:
            continue
        # Consensus = most common non-gap base
        counts = np.bincount(bases, minlength=6)
        consensus = np.argmax(counts[1:5]) + 1  # +1 for base encoding
        # Fill gaps with consensus
        imputed[column == 0, col] = consensus

    return imputed


def estimate_n_clusters(
    distance_matrix: np.ndarray,
    max_k: int = 20,
    min_k: int = 2,
) -> int:
    """Estimate optimal cluster count using silhouette scores.

    Tries k=min_k..max_k clusters using Ward linkage and computes
    silhouette score for each k. Returns the k with the highest score.

    Fallback behavior:
    - If all silhouette scores are negative, return 1 (all in one cluster)
    - If n_reads < min_k, return 1
    - Caps max_k at n_reads // 2 (cannot have more clusters than half the reads)

    Args:
        distance_matrix: Precomputed distance matrix (n_reads x n_reads).
        max_k: Maximum number of clusters to try.
        min_k: Minimum number of clusters to try.

    Returns:
        Optimal k (number of clusters). At least 1.
    """
    n_reads = distance_matrix.shape[0]

    # If too few reads, return 1
    if n_reads < min_k:
        logger.info(f"Only {n_reads} reads, returning k=1")
        return 1

    # Cap max_k at n_reads // 2
    max_k = min(max_k, n_reads // 2)

    # If max_k < min_k after capping, return 1
    if max_k < min_k:
        logger.info(f"max_k {max_k} < min_k {min_k} after capping, returning k=1")
        return 1

    # Convert to condensed distance matrix for linkage
    condensed = squareform(distance_matrix)
    if np.all(condensed == 0):
        condensed = condensed + 1e-10

    # Perform hierarchical clustering with Ward linkage
    Z = linkage(condensed, method="ward")

    best_k = 1
    best_score = -np.inf

    # Try each k in range [min_k, max_k]
    for k in range(min_k, max_k + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")

        # Skip if we got fewer clusters than requested (shouldn't happen)
        if len(np.unique(labels)) < k:
            continue

        try:
            # Compute silhouette score using precomputed distance
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
        except Exception as e:
            logger.debug(f"Could not compute silhouette score for k={k}: {e}")
            continue

        logger.debug(f"k={k}: silhouette_score={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    # Fallback: if all scores were negative, return 1
    if best_score < 0:
        logger.info(
            f"All silhouette scores negative (best={best_score:.4f}), returning k=1"
        )
        return 1

    logger.info(f"Auto-k estimation: selected k={best_k} (score={best_score:.4f})")
    return best_k


def cluster_reads_hierarchical(
    variant_matrix: np.ndarray,
    distance_threshold: float = 0.05,
    method: str = "average",
    min_cluster_size: int = 2,
    use_imputation: bool = True,
    use_ward: bool = False,
    n_clusters: int | None = None,
    auto_k: bool = False,
) -> list[HaplotypeCluster]:
    """Cluster reads using hierarchical clustering on Hamming distance.

    This implements the core deconvolution step: reads with similar
    variant profiles are grouped together as belonging to the same
    haplotype.

    When use_imputation=True (default), gaps in the variant matrix are
    filled with the per-column consensus before computing distances.
    This prevents gap-heavy matrices from corrupting the distance signal.

    Three clustering modes:
    - Distance threshold (default): cut dendrogram at a fixed distance.
      Best when the expected divergence is known and consistent.
    - Ward + n_clusters: use Ward's method to find exactly N clusters.
      Best when the number of haplotypes can be estimated from the data
      (e.g., from the number of distinct allele patterns).
    - Ward + auto_k: use Ward's method with automatic k estimation via
      silhouette score. Best when the cluster count is unknown. If both
      auto_k and n_clusters are set, n_clusters takes precedence.

    Args:
        variant_matrix: Variant-only MSA (n_reads x n_variants).
        distance_threshold: Maximum distance for merging clusters.
        method: Linkage method ('average', 'complete', 'single').
        min_cluster_size: Minimum reads per cluster.
        use_imputation: If True, impute gaps before distance computation.
        use_ward: If True, use Ward's method with Euclidean distance.
            Overrides `method`. Requires `n_clusters` to be set or
            auto_k to be True.
        n_clusters: If set, cut dendrogram to produce exactly this many
            clusters (using maxclust criterion). Takes precedence over auto_k.
        auto_k: If True and n_clusters is None, automatically estimate
            the optimal number of clusters using silhouette scores.

    Returns:
        List of HaplotypeCluster objects.
    """
    n_reads, n_variants = variant_matrix.shape

    if n_reads < 2:
        return [HaplotypeCluster(cluster_id=0, read_indices=list(range(n_reads)))]

    if n_variants == 0:
        return [HaplotypeCluster(cluster_id=0, read_indices=list(range(n_reads)))]

    # Optionally impute gaps to improve distance computation
    if use_imputation:
        gap_frac = np.mean(variant_matrix == 0)
        if gap_frac > 0.05:
            logger.info(
                f"Gap fraction {gap_frac:.1%} > 5%, imputing gaps with consensus"
            )
            dist_input = impute_gaps_with_consensus(variant_matrix)
        else:
            dist_input = variant_matrix
    else:
        dist_input = variant_matrix

    if use_ward or auto_k:
        # Ward's method: uses Euclidean distance, better for separating
        # clusters of similar size. Can use explicit n_clusters or auto_k.

        # Explicit n_clusters takes precedence over auto_k
        if n_clusters is not None:
            k = n_clusters
        elif auto_k:
            # Use Hamming distance to estimate k via silhouette scores
            dist_matrix = hamming_distance_matrix(dist_input)
            k = estimate_n_clusters(dist_matrix)
        else:
            raise ValueError("use_ward requires either n_clusters or auto_k=True")

        condensed = pdist(dist_input.astype(float), metric="euclidean")
        if np.all(condensed == 0):
            condensed = condensed + 1e-10
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, t=k, criterion="maxclust")
    else:
        # Standard Hamming distance + average linkage (distance threshold mode)
        dist_matrix = hamming_distance_matrix(dist_input)
        condensed = squareform(dist_matrix)
        if np.all(condensed == 0):
            condensed = condensed + 1e-10
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
