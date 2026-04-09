"""Read clustering for haplotype deconvolution.

Iterative clustering approach for assigning reads to distinct haplotype
groups based on their variant profiles.

The algorithm:
1. Build variant matrix from true variant positions
2. Cluster reads based on their variant profiles
3. Build consensus for each cluster
4. Repeat: re-examine each cluster for further sub-structure
5. Continue until no more sub-clusters are found

Supports hierarchical (average linkage, Ward's method) and spectral
clustering, with gap-aware distance computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

from openlong.deconv.positions import identify_variant_positions, build_variant_matrix

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


def _compute_wcss_from_labels(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute within-cluster sum of squares (WCSS) from distance matrix and labels.

    For each cluster, computes the sum of squared distances from each point
    to the cluster centroid (represented as the mean distance to cluster members).

    Args:
        distance_matrix: Pairwise distance matrix (n_reads x n_reads).
        labels: Cluster assignment for each point.

    Returns:
        Total within-cluster sum of squares.
    """
    wcss = 0.0
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        cluster_indices = np.where(mask)[0]

        if len(cluster_indices) <= 1:
            continue

        # Compute within-cluster distances
        sub_dist = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
        # Sum all pairwise distances (counting each pair once)
        wcss += np.sum(np.triu(sub_dist, k=1))

    return wcss


def _smooth_scores(scores: np.ndarray, window: int = 3) -> np.ndarray:
    """Smooth a score curve with a moving window to reduce noise.

    Args:
        scores: Array of scores.
        window: Window size (should be odd).

    Returns:
        Smoothed scores array.
    """
    if len(scores) < window:
        return scores

    half_window = window // 2
    smoothed = np.zeros_like(scores, dtype=float)

    for i in range(len(scores)):
        start = max(0, i - half_window)
        end = min(len(scores), i + half_window + 1)
        smoothed[i] = np.mean(scores[start:end])

    return smoothed


def _detect_elbow(wcss_values: np.ndarray, k_values: np.ndarray) -> int:
    """Detect the elbow point in WCSS curve.

    Uses the maximum curvature method: find the point where the curve
    changes direction most sharply.

    Args:
        wcss_values: Within-cluster sum of squares for each k.
        k_values: Corresponding k values.

    Returns:
        k value at the elbow (index into wcss_values).
    """
    if len(wcss_values) < 3:
        return k_values[0]

    # Normalize to [0, 1] for comparison
    wcss_norm = (wcss_values - wcss_values.min()) / (wcss_values.max() - wcss_values.min() + 1e-10)

    # First and second derivatives (approximate)
    first_deriv = np.diff(wcss_norm)
    second_deriv = np.diff(first_deriv)

    # Elbow is where second derivative is most negative (greatest decrease in improvement)
    # Skip first and last points
    if len(second_deriv) > 0:
        elbow_idx = np.argmin(second_deriv) + 1
        return k_values[elbow_idx]

    return k_values[0]


def _gap_statistic(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    n_refs: int = 10,
) -> float:
    """Compute gap statistic for cluster quality.

    Compares the within-cluster dispersion to that of a uniform reference
    distribution. Higher gap indicates better clustering.

    Args:
        distance_matrix: Pairwise distance matrix.
        labels: Cluster assignments.
        n_refs: Number of reference datasets to sample.

    Returns:
        Gap statistic value.
    """
    n_reads = distance_matrix.shape[0]

    # Observed dispersion: log(WCSS)
    wcss_obs = _compute_wcss_from_labels(distance_matrix, labels)
    log_wcss_obs = np.log(wcss_obs + 1e-10)

    # Reference dispersion: average over random uniform data
    log_wcss_refs = []
    for _ in range(n_refs):
        # Random uniform data in same dimensionality as original
        # (approximate using random uniform in [0, max_distance])
        max_dist = distance_matrix.max()
        ref_dist = np.random.uniform(0, max_dist, size=distance_matrix.shape)
        np.fill_diagonal(ref_dist, 0)

        wcss_ref = _compute_wcss_from_labels(ref_dist, labels)
        log_wcss_refs.append(np.log(wcss_ref + 1e-10))

    log_wcss_ref = np.mean(log_wcss_refs)
    gap = log_wcss_ref - log_wcss_obs

    return gap


def estimate_n_clusters(
    distance_matrix: np.ndarray,
    max_k: int = 20,
    min_k: int = 2,
    method: str = "combined",
    lambda_penalty: float = 0.1,
    smooth_window: int = 3,
) -> int:
    """Estimate optimal cluster count using multiple strategies.

    Implements robust auto-k estimation using:
    1. Silhouette score (original method)
    2. Elbow detection (WCSS curvature)
    3. BIC-like penalty criterion (prefers simpler models)
    4. Gap statistic (robust to noise)

    The combined approach (default) uses consensus across multiple signals
    with configurable penalty for complexity.

    Fallback behavior:
    - If all silhouette scores are negative, return 1 (all in one cluster)
    - If n_reads < min_k, return 1
    - Caps max_k at n_reads // 2 (cannot have more clusters than half the reads)

    Args:
        distance_matrix: Precomputed distance matrix (n_reads x n_reads).
        max_k: Maximum number of clusters to try.
        min_k: Minimum number of clusters to try.
        method: Strategy to use:
            - "silhouette": original silhouette-only method
            - "elbow": elbow detection on WCSS
            - "penalized": BIC-like penalized silhouette
            - "gap": gap statistic
            - "combined": consensus across all methods (default)
        lambda_penalty: Penalty weight for complexity (BIC-like term).
        smooth_window: Window size for smoothing score curves (reduces noise spikes).

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

    # Collect metrics for all k values
    k_values = np.arange(min_k, max_k + 1)
    silhouette_scores = []
    wcss_values = []
    gap_scores = []

    for k in k_values:
        labels = fcluster(Z, t=k, criterion="maxclust")

        # Skip if we got fewer clusters than requested
        if len(np.unique(labels)) < k:
            silhouette_scores.append(-np.inf)
            wcss_values.append(np.inf)
            gap_scores.append(-np.inf)
            continue

        try:
            # Silhouette score
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
            silhouette_scores.append(score)
        except Exception as e:
            logger.debug(f"Could not compute silhouette score for k={k}: {e}")
            silhouette_scores.append(-np.inf)

        try:
            # WCSS
            wcss = _compute_wcss_from_labels(distance_matrix, labels)
            wcss_values.append(wcss)
        except Exception as e:
            logger.debug(f"Could not compute WCSS for k={k}: {e}")
            wcss_values.append(np.inf)

        try:
            # Gap statistic
            gap = _gap_statistic(distance_matrix, labels, n_refs=10)
            gap_scores.append(gap)
        except Exception as e:
            logger.debug(f"Could not compute gap statistic for k={k}: {e}")
            gap_scores.append(-np.inf)

        logger.debug(
            f"k={k}: silhouette={silhouette_scores[-1]:.4f}, "
            f"wcss={wcss_values[-1]:.2f}, gap={gap_scores[-1]:.4f}"
        )

    silhouette_scores = np.array(silhouette_scores)
    wcss_values = np.array(wcss_values)
    gap_scores = np.array(gap_scores)

    # Determine best k based on chosen method
    if method == "silhouette":
        if np.all(np.isinf(silhouette_scores)):
            logger.info("All silhouette scores invalid, returning k=1")
            return 1
        best_k = k_values[np.argmax(silhouette_scores)]
        logger.info(f"Auto-k (silhouette): selected k={best_k}")

    elif method == "elbow":
        if np.all(np.isinf(wcss_values)):
            logger.info("All WCSS values invalid, returning k=1")
            return 1
        best_k = _detect_elbow(wcss_values, k_values)
        logger.info(f"Auto-k (elbow): selected k={best_k}")

    elif method == "penalized":
        # BIC-like criterion: maximize silhouette - penalty * log(k)
        if np.all(np.isinf(silhouette_scores)):
            logger.info("All silhouette scores invalid, returning k=1")
            return 1

        # Smooth to reduce noise
        sil_smooth = _smooth_scores(silhouette_scores, window=smooth_window)
        penalty_term = lambda_penalty * np.log(k_values) / np.log(max_k + 1)
        penalized_scores = sil_smooth - penalty_term

        best_k = k_values[np.argmax(penalized_scores)]
        logger.info(f"Auto-k (penalized): selected k={best_k}")

    elif method == "gap":
        if np.all(np.isinf(gap_scores)):
            logger.info("All gap scores invalid, returning k=1")
            return 1
        best_k = k_values[np.argmax(gap_scores)]
        logger.info(f"Auto-k (gap): selected k={best_k}")

    else:  # method == "combined" (default)
        # Consensus approach: normalize all scores and combine
        if np.all(np.isinf(silhouette_scores)):
            logger.info("All silhouette scores invalid, returning k=1")
            return 1

        # Smooth silhouette to reduce noise spikes (exclude inf values)
        valid_sil = silhouette_scores[np.isfinite(silhouette_scores)]
        if len(valid_sil) > 0:
            sil_smooth = _smooth_scores(silhouette_scores, window=smooth_window)
        else:
            sil_smooth = silhouette_scores

        # Normalize silhouette to [0, 1], handling inf values
        valid_sil_smooth = sil_smooth[np.isfinite(sil_smooth)]
        if len(valid_sil_smooth) > 0:
            sil_min, sil_max = valid_sil_smooth.min(), valid_sil_smooth.max()
            if sil_max > sil_min:
                sil_norm = np.where(
                    np.isfinite(sil_smooth),
                    (sil_smooth - sil_min) / (sil_max - sil_min),
                    0.0
                )
            else:
                sil_norm = np.where(np.isfinite(sil_smooth), 0.5, 0.0)
        else:
            sil_norm = np.zeros_like(sil_smooth)

        # Normalize WCSS (inverted: lower is better), handling inf values
        valid_wcss = wcss_values[np.isfinite(wcss_values)]
        if len(valid_wcss) > 0:
            wcss_min, wcss_max = valid_wcss.min(), valid_wcss.max()
            if wcss_max > wcss_min:
                wcss_norm = np.where(
                    np.isfinite(wcss_values),
                    (wcss_max - wcss_values) / (wcss_max - wcss_min),
                    0.0
                )
            else:
                wcss_norm = np.where(np.isfinite(wcss_values), 0.5, 0.0)
        else:
            wcss_norm = np.zeros_like(wcss_values)

        # Normalize gap (higher is better), handling inf values
        valid_gap = gap_scores[np.isfinite(gap_scores)]
        if len(valid_gap) > 0:
            gap_min, gap_max = valid_gap.min(), valid_gap.max()
            if gap_max > gap_min:
                gap_norm = np.where(
                    np.isfinite(gap_scores),
                    (gap_scores - gap_min) / (gap_max - gap_min),
                    0.0
                )
            else:
                gap_norm = np.where(np.isfinite(gap_scores), 0.5, 0.0)
        else:
            gap_norm = np.zeros_like(gap_scores)

        # Add penalty term for complexity
        penalty_term = lambda_penalty * np.log(k_values) / np.log(max_k + 1)

        # Combined score: weighted average of normalized metrics - complexity penalty
        combined_score = (sil_norm + wcss_norm + gap_norm) / 3.0 - penalty_term

        best_k = k_values[np.argmax(combined_score)]
        logger.info(f"Auto-k (combined): selected k={best_k}")

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
    """Iterative deconvolution for haplotype resolution.

    Recursive process:
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


def recursive_cluster_reads(
    msa: np.ndarray,
    is_main: np.ndarray,
    platform: str = "pacbio_clr",
    fdr_threshold: float = 0.2,
    min_cluster_size: int = 3,
    min_variants: int = 1,
    depth: int = 0,
    max_depth: int = 20,
) -> list[HaplotypeCluster]:
    """Recursive clustering for haplotype deconvolution.

    Recursive binary splitting approach:
    1. Run identify_variant_positions() on the input MSA
    2. If no variant positions found OR depth >= max_depth → return single cluster
    3. Build variant matrix from variant positions
    4. Compute Hamming distance matrix
    5. Run hierarchical clustering with complete linkage (furthest neighbor)
       and binary split (maxclust with t=2)
    6. Split reads into 2 groups based on cluster labels
    7. Recurse on each group (passing the SUB-MSA for that group)
    8. Return the flattened list of leaf clusters

    Key insight: variant positions change when you look at a subgroup, so we
    re-run variant detection at each recursion level on the FULL corrected
    MSA (not just variant matrix) for each subgroup.

    Args:
        msa: Full corrected MSA matrix (n_reads x n_positions). Must be the
            full MSA, not just variant columns, because we re-identify variants
            at each level.
        is_main: Boolean array indicating which positions are "main" (stable
            after INDEL correction) in the full MSA.
        platform: Sequencing platform for error rate estimation.
        fdr_threshold: FDR threshold for variant significance (0.2 = permissive).
        min_cluster_size: Minimum reads to form a valid cluster.
        min_variants: Minimum variant positions required to justify further splitting.
        depth: Current recursion depth (auto-incremented).
        max_depth: Maximum recursion depth to prevent infinite recursion.

    Returns:
        List of HaplotypeCluster objects representing leaf clusters.
    """
    n_reads = msa.shape[0]
    cluster_id = depth + 1

    # Base case: stop if we've reached max depth or have too few reads
    if depth >= max_depth or n_reads < min_cluster_size:
        logger.debug(
            f"Recursion depth {depth}: returning single cluster with {n_reads} reads "
            f"(max_depth={max_depth}, min_cluster_size={min_cluster_size})"
        )
        return [HaplotypeCluster(cluster_id=cluster_id, read_indices=list(range(n_reads)))]

    # Step 1: Identify variant positions in this subgroup
    variant_positions = identify_variant_positions(
        msa,
        is_main,
        platform=platform,
        fdr_threshold=fdr_threshold,
    )

    # Step 2: If no variants or too few variants, return as leaf
    if len(variant_positions) < min_variants:
        logger.debug(
            f"Recursion depth {depth}: {len(variant_positions)} variant positions < "
            f"{min_variants}, returning single cluster with {n_reads} reads"
        )
        return [HaplotypeCluster(cluster_id=cluster_id, read_indices=list(range(n_reads)))]

    # Step 3: Build variant matrix from identified positions
    variant_matrix = build_variant_matrix(msa, variant_positions)

    # Step 4: Compute Hamming distance matrix
    dist_matrix = hamming_distance_matrix(variant_matrix)

    # Step 5: Run hierarchical clustering with binary split
    condensed = squareform(dist_matrix)
    if np.all(condensed == 0):
        # All distances are zero (identical reads)
        logger.debug(
            f"Recursion depth {depth}: all distances are zero, returning single cluster"
        )
        return [HaplotypeCluster(cluster_id=cluster_id, read_indices=list(range(n_reads)))]

    # Use Ward's method for the binary split. The paper specifies complete
    # linkage, but complete linkage + maxclust=2 always peels off a single
    # outlier (the last merge joins the two most distant *points*, not the
    # two most balanced groups). Ward minimizes within-cluster variance and
    # reliably produces balanced binary partitions. We still enforce binary
    # splitting and recurse — the overall algorithm structure (recursive
    # binary splitting until no significant variants remain) matches the paper.
    Z = linkage(condensed, method="ward")

    # Binary split: always cut into exactly 2 clusters
    labels = fcluster(Z, t=2, criterion="maxclust")

    # Step 6: Check if we actually got 2 clusters with reasonable balance
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.debug(
            f"Recursion depth {depth}: could not split into 2 clusters, "
            f"returning single cluster"
        )
        return [HaplotypeCluster(cluster_id=cluster_id, read_indices=list(range(n_reads)))]

    # Check balance: if the split is extremely lopsided (one side <2% of reads),
    # treat it as a failed split — the tiny cluster is noise, not a real haplotype.
    label_counts = {l: np.sum(labels == l) for l in unique_labels}
    min_count = min(label_counts.values())
    if min_count < max(min_cluster_size, int(0.02 * n_reads)):
        logger.debug(
            f"Recursion depth {depth}: split too lopsided "
            f"({label_counts}), returning single cluster"
        )
        return [HaplotypeCluster(cluster_id=cluster_id, read_indices=list(range(n_reads)))]

    # Step 7: Split reads into 2 groups and recurse
    final_clusters = []
    for label in unique_labels:
        read_indices = np.where(labels == label)[0].tolist()

        if len(read_indices) < min_cluster_size:
            # Too small to split further, but include as leaf
            logger.debug(
                f"Recursion depth {depth}: subgroup with {len(read_indices)} reads "
                f"< min_cluster_size, including as leaf"
            )
            final_clusters.append(
                HaplotypeCluster(cluster_id=cluster_id, read_indices=read_indices)
            )
        else:
            # Extract sub-MSA for this group
            sub_msa = msa[read_indices]

            # Recurse on sub-MSA
            sub_clusters = recursive_cluster_reads(
                sub_msa,
                is_main,
                platform=platform,
                fdr_threshold=fdr_threshold,
                min_cluster_size=min_cluster_size,
                min_variants=min_variants,
                depth=depth + 1,
                max_depth=max_depth,
            )

            # Map cluster read indices back to original MSA
            for sub_cluster in sub_clusters:
                original_indices = [read_indices[i] for i in sub_cluster.read_indices]
                sub_cluster.read_indices = original_indices
                final_clusters.append(sub_cluster)

    logger.info(
        f"Recursion depth {depth}: {n_reads} reads split into "
        f"{len(final_clusters)} clusters"
    )
    return final_clusters


def _merge_similar_clusters(
    clusters: list[HaplotypeCluster],
    msa: np.ndarray,
    merge_threshold: float = 0.02,
) -> list[HaplotypeCluster]:
    """Merge clusters whose consensus profiles are very similar.

    After recursive splitting, some haplotypes may be over-fragmented
    because sequencing noise at the sub-cluster level creates spurious
    variant positions. This post-processing step merges clusters whose
    consensus variant profiles are within merge_threshold Hamming distance.

    Args:
        clusters: List of clusters from recursive splitting.
        msa: Full corrected MSA matrix.
        merge_threshold: Maximum Hamming distance between consensus
            profiles to merge two clusters. Default 0.02 = 2% divergence,
            which is below the ~5% minimum inter-haplotype distance for
            most viral quasispecies.

    Returns:
        Merged list of clusters.
    """
    if len(clusters) <= 1:
        return clusters

    # Build consensus profile for each cluster (majority vote per column)
    n_pos = msa.shape[1]
    profiles = []
    for c in clusters:
        sub_msa = msa[c.read_indices]
        consensus = np.zeros(n_pos, dtype=np.uint8)
        for col in range(n_pos):
            bases = sub_msa[:, col]
            bases = bases[bases > 0]
            if len(bases) > 0:
                counts = np.bincount(bases, minlength=6)
                consensus[col] = np.argmax(counts[1:5]) + 1
        profiles.append(consensus)

    profiles = np.array(profiles)

    # Compute pairwise distances between cluster consensuses
    n = len(clusters)
    merged = [False] * n
    merge_map = list(range(n))  # points to final cluster index

    for i in range(n):
        if merged[i]:
            continue
        for j in range(i + 1, n):
            if merged[j]:
                continue
            # Hamming distance between consensus profiles
            mask = (profiles[i] > 0) & (profiles[j] > 0)
            shared = mask.sum()
            if shared < 10:
                continue
            dist = np.sum(profiles[i][mask] != profiles[j][mask]) / shared
            if dist <= merge_threshold:
                # Merge j into i
                clusters[i].read_indices.extend(clusters[j].read_indices)
                clusters[i].n_reads = len(clusters[i].read_indices)
                merged[j] = True
                # Update consensus profile for merged cluster
                sub_msa = msa[clusters[i].read_indices]
                for col in range(n_pos):
                    bases = sub_msa[:, col]
                    bases = bases[bases > 0]
                    if len(bases) > 0:
                        counts = np.bincount(bases, minlength=6)
                        profiles[i][col] = np.argmax(counts[1:5]) + 1

    result = [c for i, c in enumerate(clusters) if not merged[i]]
    if len(result) < len(clusters):
        logger.info(
            f"Post-merge: {len(clusters)} clusters -> {len(result)} "
            f"(merged {len(clusters) - len(result)} similar pairs)"
        )
    return result


def openlong_cluster(
    msa: np.ndarray,
    is_main: np.ndarray,
    platform: str = "pacbio_clr",
    merge_threshold: float = 0.02,
    **kwargs,
) -> list[HaplotypeCluster]:
    """Full OpenLong recursive clustering with post-merge.

    Convenience function that calls recursive_cluster_reads with sensible
    defaults, then merges over-fragmented clusters that are very similar.

    Args:
        msa: Full corrected MSA matrix (n_reads x n_positions).
        is_main: Boolean array indicating main positions.
        platform: Sequencing platform for error rate estimation.
        merge_threshold: Maximum consensus Hamming distance to merge
            similar clusters (default 0.02 = 2% divergence).
        **kwargs: Additional arguments passed to recursive_cluster_reads
            (e.g., fdr_threshold, min_cluster_size, min_variants, max_depth).

    Returns:
        List of HaplotypeCluster objects representing final haplotypes.
    """
    clusters = recursive_cluster_reads(msa, is_main, platform=platform, **kwargs)

    # Post-process: merge over-fragmented clusters
    clusters = _merge_similar_clusters(clusters, msa, merge_threshold=merge_threshold)

    # Renumber clusters sequentially
    for i, cluster in enumerate(clusters):
        cluster.cluster_id = i + 1

    logger.info(f"OpenLong clustering: {msa.shape[0]} reads -> {len(clusters)} haplotypes")
    return clusters


# Backward compatibility alias
dilernia_cluster = openlong_cluster
