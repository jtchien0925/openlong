"""Tests for read clustering and haplotype deconvolution."""

import numpy as np
import pytest

from openlong.correct.indel import BASE_A, BASE_C, BASE_G, BASE_T
from openlong.deconv.cluster import (
    hamming_distance_matrix,
    cluster_reads_hierarchical,
    estimate_haplotype_frequencies,
    HaplotypeCluster,
)


class TestHammingDistance:
    def test_identical_reads(self):
        """Identical reads should have zero distance."""
        vm = np.array([
            [BASE_A, BASE_C, BASE_G],
            [BASE_A, BASE_C, BASE_G],
        ], dtype=np.uint8)
        dist = hamming_distance_matrix(vm)
        assert dist[0, 1] == 0.0
        assert dist[1, 0] == 0.0

    def test_completely_different(self):
        """Completely different reads should have distance 1.0."""
        vm = np.array([
            [BASE_A, BASE_A, BASE_A],
            [BASE_C, BASE_C, BASE_C],
        ], dtype=np.uint8)
        dist = hamming_distance_matrix(vm)
        assert dist[0, 1] == 1.0

    def test_partial_difference(self):
        """One mismatch in 3 positions = 1/3 distance."""
        vm = np.array([
            [BASE_A, BASE_C, BASE_G],
            [BASE_A, BASE_C, BASE_T],  # Only last differs
        ], dtype=np.uint8)
        dist = hamming_distance_matrix(vm)
        assert pytest.approx(dist[0, 1], abs=0.01) == 1 / 3

    def test_gaps_ignored(self):
        """Gap positions should be excluded from distance."""
        vm = np.array([
            [BASE_A, 0, BASE_G],
            [BASE_A, BASE_C, BASE_G],
        ], dtype=np.uint8)
        dist = hamming_distance_matrix(vm, ignore_gaps=True)
        # Only positions 0 and 2 compared (both match)
        assert dist[0, 1] == 0.0

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        rng = np.random.RandomState(42)
        vm = rng.randint(1, 5, size=(5, 10)).astype(np.uint8)
        dist = hamming_distance_matrix(vm)
        assert np.allclose(dist, dist.T)


class TestClusterReads:
    def test_two_clear_clusters(self):
        """Two distinct variant profiles should form two clusters."""
        # Group 1: AAA
        # Group 2: CCC
        vm = np.array([
            [BASE_A, BASE_A, BASE_A],
            [BASE_A, BASE_A, BASE_A],
            [BASE_A, BASE_A, BASE_A],
            [BASE_C, BASE_C, BASE_C],
            [BASE_C, BASE_C, BASE_C],
            [BASE_C, BASE_C, BASE_C],
        ], dtype=np.uint8)

        clusters = cluster_reads_hierarchical(
            vm, distance_threshold=0.5, min_cluster_size=2,
        )
        assert len(clusters) == 2

    def test_single_group(self):
        """All identical reads should form one cluster."""
        vm = np.full((10, 5), BASE_A, dtype=np.uint8)
        clusters = cluster_reads_hierarchical(vm, distance_threshold=0.5)
        assert len(clusters) == 1

    def test_minimum_cluster_size(self):
        """Clusters below min size should be dropped."""
        vm = np.array([
            [BASE_A, BASE_A],
            [BASE_A, BASE_A],
            [BASE_A, BASE_A],
            [BASE_C, BASE_C],  # Only 1 read of this type
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
