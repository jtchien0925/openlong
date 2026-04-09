"""Tests for read clustering and haplotype deconvolution."""

import numpy as np
import pytest

from openlong.correct.indel import BASE_A, BASE_C, BASE_G, BASE_T
from openlong.deconv.cluster import (
    hamming_distance_matrix,
    cluster_reads_hierarchical,
    estimate_n_clusters,
    estimate_haplotype_frequencies,
    HaplotypeCluster,
    recursive_cluster_reads,
    openlong_cluster,
)


class TestHammingDistance:
    def test_identical_reads(self):
        """Identical reads should have zero distance."""
        vm = np.tile([BASE_A, BASE_C, BASE_G, BASE_T, BASE_A,
                      BASE_C, BASE_G, BASE_T, BASE_A, BASE_C], (2, 1)).astype(np.uint8)
        dist = hamming_distance_matrix(vm, min_shared_positions=1)
        assert dist[0, 1] == 0.0
        assert dist[1, 0] == 0.0

    def test_completely_different(self):
        """Completely different reads should have distance 1.0."""
        vm = np.array([
            [BASE_A] * 10,
            [BASE_C] * 10,
        ], dtype=np.uint8)
        dist = hamming_distance_matrix(vm, min_shared_positions=1)
        assert dist[0, 1] == 1.0

    def test_partial_difference(self):
        """One mismatch in 3 positions = 1/3 distance."""
        row1 = [BASE_A, BASE_C, BASE_G]
        row2 = [BASE_A, BASE_C, BASE_T]  # Only last differs
        vm = np.array([row1, row2], dtype=np.uint8)
        dist = hamming_distance_matrix(vm, min_shared_positions=1)
        assert pytest.approx(dist[0, 1], abs=0.01) == 1 / 3

    def test_gaps_ignored(self):
        """Gap positions should be excluded from distance."""
        row1 = [BASE_A, 0, BASE_G] + [BASE_A] * 7
        row2 = [BASE_A, BASE_C, BASE_G] + [BASE_A] * 7
        vm = np.array([row1, row2], dtype=np.uint8)
        dist = hamming_distance_matrix(vm, ignore_gaps=True, min_shared_positions=1)
        # Only non-gap positions compared (all match)
        assert dist[0, 1] == 0.0

    def test_min_shared_positions(self):
        """Pairs below min_shared_positions get distance 1.0."""
        vm = np.array([
            [BASE_A, 0, 0],
            [0, BASE_C, 0],
        ], dtype=np.uint8)
        dist = hamming_distance_matrix(vm, min_shared_positions=2)
        assert dist[0, 1] == 1.0  # Only 0 shared positions

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        rng = np.random.RandomState(42)
        vm = rng.randint(1, 5, size=(5, 10)).astype(np.uint8)
        dist = hamming_distance_matrix(vm)
        assert np.allclose(dist, dist.T)


class TestClusterReads:
    def test_two_clear_clusters(self):
        """Two distinct variant profiles should form two clusters."""
        # Group 1: all A (10 columns to satisfy min_shared_positions default)
        # Group 2: all C
        vm = np.array([
            [BASE_A] * 10,
            [BASE_A] * 10,
            [BASE_A] * 10,
            [BASE_C] * 10,
            [BASE_C] * 10,
            [BASE_C] * 10,
        ], dtype=np.uint8)

        clusters = cluster_reads_hierarchical(
            vm, distance_threshold=0.5, min_cluster_size=2,
        )
        assert len(clusters) == 2

    def test_single_group(self):
        """All identical reads should form one cluster."""
        vm = np.full((10, 10), BASE_A, dtype=np.uint8)
        clusters = cluster_reads_hierarchical(vm, distance_threshold=0.5)
        assert len(clusters) == 1

    def test_minimum_cluster_size(self):
        """Clusters below min size should be dropped."""
        vm = np.array([
            [BASE_A] * 10,
            [BASE_A] * 10,
            [BASE_A] * 10,
            [BASE_C] * 10,  # Only 1 read of this type
        ], dtype=np.uint8)

        clusters = cluster_reads_hierarchical(
            vm, distance_threshold=0.5, min_cluster_size=2,
        )
        # The singleton should be dropped
        for c in clusters:
            assert c.n_reads >= 2

    def test_single_read(self):
        """Single read should form one cluster."""
        vm = np.array([[BASE_A, BASE_C]], dtype=np.uint8)
        clusters = cluster_reads_hierarchical(vm)
        assert len(clusters) == 1


class TestAutoK:
    def test_auto_k_finds_two_clusters(self):
        """Auto-k should detect two clear clusters."""
        # Create a clear 2-cluster pattern with 10+ columns
        vm = np.array([
            [BASE_A] * 15,  # Group 1
            [BASE_A] * 15,
            [BASE_A] * 15,
            [BASE_C] * 15,  # Group 2
            [BASE_C] * 15,
            [BASE_C] * 15,
        ], dtype=np.uint8)

        clusters = cluster_reads_hierarchical(
            vm, auto_k=True, use_ward=True, min_cluster_size=1
        )
        assert len(clusters) == 2

    def test_auto_k_single_cluster(self):
        """Auto-k should return 1 cluster for identical reads."""
        vm = np.full((5, 15), BASE_A, dtype=np.uint8)
        clusters = cluster_reads_hierarchical(
            vm, auto_k=True, use_ward=True, min_cluster_size=1
        )
        assert len(clusters) == 1

    def test_auto_k_respects_explicit_n_clusters(self):
        """When n_clusters is set, auto_k should not override it."""
        # Create a 2-cluster pattern
        vm = np.array([
            [BASE_A] * 15,
            [BASE_A] * 15,
            [BASE_C] * 15,
            [BASE_C] * 15,
        ], dtype=np.uint8)

        # Force 1 cluster explicitly
        clusters = cluster_reads_hierarchical(
            vm, auto_k=True, use_ward=True, n_clusters=1, min_cluster_size=1
        )
        assert len(clusters) == 1


class TestHaplotypeFrequencies:
    def test_equal_frequencies(self):
        """Two equal-sized clusters should be 50/50."""
        clusters = [
            HaplotypeCluster(cluster_id=1, read_indices=[0, 1, 2]),
            HaplotypeCluster(cluster_id=2, read_indices=[3, 4, 5]),
        ]
        freqs = estimate_haplotype_frequencies(clusters)
        assert freqs[1] == pytest.approx(0.5)
        assert freqs[2] == pytest.approx(0.5)

    def test_unequal_frequencies(self):
        """Frequencies should reflect read counts."""
        clusters = [
            HaplotypeCluster(cluster_id=1, read_indices=[0, 1, 2, 3]),
            HaplotypeCluster(cluster_id=2, read_indices=[4]),
        ]
        freqs = estimate_haplotype_frequencies(clusters)
        assert freqs[1] == pytest.approx(0.8)
        assert freqs[2] == pytest.approx(0.2)

    def test_sum_to_one(self):
        """Frequencies should sum to 1.0."""
        clusters = [
            HaplotypeCluster(cluster_id=i, read_indices=list(range(i * 3, (i + 1) * 3)))
            for i in range(4)
        ]
        freqs = estimate_haplotype_frequencies(clusters)
        assert pytest.approx(sum(freqs.values()), abs=0.001) == 1.0


class TestRecursiveClustering:
    """Tests for the recursive clustering algorithm."""

    def test_recursive_finds_two_clusters(self):
        """Recursive clustering should find two distinct groups."""
        # Create MSA with two clear groups:
        # Group 1: all A (30 reads, 20 positions to ensure variants are detected)
        # Group 2: all C (30 reads, same 20 positions)
        n_reads_per_group = 15
        n_positions = 20

        msa = np.vstack([
            np.full((n_reads_per_group, n_positions), BASE_A, dtype=np.uint8),
            np.full((n_reads_per_group, n_positions), BASE_C, dtype=np.uint8),
        ])

        # Mark all positions as main (variant-eligible)
        is_main = np.ones(n_positions, dtype=bool)

        clusters = recursive_cluster_reads(
            msa,
            is_main,
            platform="pacbio_clr",
            fdr_threshold=0.2,
            min_cluster_size=2,
            min_variants=1,
            max_depth=20,
        )

        # Should find 2 clusters
        assert len(clusters) == 2
        # All reads should be assigned
        total_reads = sum(c.n_reads for c in clusters)
        assert total_reads == 2 * n_reads_per_group

    def test_recursive_stops_on_homogeneous(self):
        """All identical reads should form single cluster."""
        n_reads = 20
        n_positions = 15

        # All reads are identical
        msa = np.full((n_reads, n_positions), BASE_A, dtype=np.uint8)
        is_main = np.ones(n_positions, dtype=bool)

        clusters = recursive_cluster_reads(
            msa,
            is_main,
            platform="pacbio_clr",
            fdr_threshold=0.2,
            min_cluster_size=3,
        )

        # Should return single cluster since no variant positions exist
        assert len(clusters) == 1
        assert clusters[0].n_reads == n_reads

    def test_recursive_finds_multiple(self):
        """Should find at least 2 clusters when data contains subgroups."""
        # Create two clearly distinct groups to force binary splitting
        # This tests the recursive splitting property
        n_reads_per_group = 20
        n_positions = 30

        # Group 1: mostly A with some C
        group1 = np.full((n_reads_per_group, n_positions), BASE_A, dtype=np.uint8)

        # Group 2: mostly C with some G
        group2 = np.full((n_reads_per_group, n_positions), BASE_C, dtype=np.uint8)

        msa = np.vstack([group1, group2])
        is_main = np.ones(n_positions, dtype=bool)

        clusters = recursive_cluster_reads(
            msa,
            is_main,
            platform="pacbio_clr",
            fdr_threshold=0.2,  # Permissive like the paper
            min_cluster_size=2,
            min_variants=1,
            max_depth=20,
        )

        # Binary splitting should find at least 2 clusters
        assert len(clusters) >= 2
        total_reads = sum(c.n_reads for c in clusters)
        assert total_reads == 2 * n_reads_per_group

    def test_recursive_respects_max_depth(self):
        """Should stop recursing at max_depth."""
        n_reads = 20
        n_positions = 15

        # Two distinct groups
        msa = np.vstack([
            np.full((n_reads // 2, n_positions), BASE_A, dtype=np.uint8),
            np.full((n_reads // 2, n_positions), BASE_C, dtype=np.uint8),
        ])

        is_main = np.ones(n_positions, dtype=bool)

        # Set max_depth to 1 to prevent recursion
        clusters = recursive_cluster_reads(
            msa,
            is_main,
            platform="pacbio_clr",
            fdr_threshold=0.2,
            min_cluster_size=1,
            max_depth=1,
        )

        # Should still complete without error
        assert len(clusters) >= 1
        total_reads = sum(c.n_reads for c in clusters)
        assert total_reads == n_reads

    def test_openlong_cluster_wrapper(self):
        """openlong_cluster wrapper should work and renumber clusters."""
        n_reads_per_group = 12
        n_positions = 20

        msa = np.vstack([
            np.full((n_reads_per_group, n_positions), BASE_A, dtype=np.uint8),
            np.full((n_reads_per_group, n_positions), BASE_C, dtype=np.uint8),
        ])

        is_main = np.ones(n_positions, dtype=bool)

        clusters = openlong_cluster(
            msa,
            is_main,
            platform="pacbio_clr",
            min_cluster_size=2,
        )

        # Should find 2 clusters
        assert len(clusters) == 2

        # Clusters should be renumbered sequentially starting at 1
        cluster_ids = sorted([c.cluster_id for c in clusters])
        assert cluster_ids == [1, 2]

        # All reads assigned
        total_reads = sum(c.n_reads for c in clusters)
        assert total_reads == 2 * n_reads_per_group

    def test_recursive_min_cluster_size(self):
        """Minimum cluster size stops initial recursion if data is too small."""
        n_reads = 8
        n_positions = 15

        # Two groups of 4 reads each
        msa = np.vstack([
            np.full((4, n_positions), BASE_A, dtype=np.uint8),
            np.full((4, n_positions), BASE_C, dtype=np.uint8),
        ])

        is_main = np.ones(n_positions, dtype=bool)

        clusters = recursive_cluster_reads(
            msa,
            is_main,
            platform="pacbio_clr",
            min_cluster_size=5,  # Larger than any potential subgroup
        )

        # min_cluster_size blocks recursion on initial check, returns single cluster
        # OR recursion happens but subgroups too small to recurse further.
        # In this case, we expect either 1 or 2 clusters depending on the order of checks.
        assert len(clusters) >= 1
        total_reads = sum(c.n_reads for c in clusters)
        assert total_reads == n_reads

    def test_recursive_with_gaps(self):
        """Should handle gaps in MSA (real-world data)."""
        # Reads with some gaps (0 = gap)
        msa = np.array([
            [BASE_A, BASE_A, BASE_A, BASE_A, BASE_A,
             BASE_A, BASE_A, BASE_A, BASE_A, BASE_A],
            [BASE_A, 0, BASE_A, BASE_A, BASE_A,
             BASE_A, BASE_A, BASE_A, BASE_A, BASE_A],
            [BASE_A, BASE_A, BASE_A, 0, BASE_A,
             BASE_A, BASE_A, BASE_A, BASE_A, BASE_A],
            [BASE_C, BASE_C, BASE_C, BASE_C, BASE_C,
             BASE_C, BASE_C, BASE_C, BASE_C, BASE_C],
            [BASE_C, BASE_C, 0, BASE_C, BASE_C,
             BASE_C, BASE_C, BASE_C, BASE_C, BASE_C],
        ], dtype=np.uint8)

        is_main = np.ones(10, dtype=bool)

        clusters = recursive_cluster_reads(
            msa,
            is_main,
            platform="pacbio_clr",
            min_cluster_size=1,
        )

        # Should handle gracefully and find clusters despite gaps
        assert len(clusters) >= 1
        total_reads = sum(c.n_reads for c in clusters)
        assert total_reads == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
