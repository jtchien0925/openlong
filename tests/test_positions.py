"""Tests for variant position identification."""

import numpy as np
import pytest

from openlong.correct.indel import BASE_A, BASE_C, BASE_G, BASE_T, GAP
from openlong.deconv.positions import (
    identify_variant_positions,
    benjamini_hochberg,
    build_variant_matrix,
)


class TestBenjaminiHochberg:
    def test_empty_input(self):
        result = benjamini_hochberg(np.array([]))
        assert len(result) == 0

    def test_single_pvalue(self):
        result = benjamini_hochberg(np.array([0.01]))
        assert len(result) == 1
        assert result[0] == pytest.approx(0.01)

    def test_monotonicity(self):
        """Q-values should be non-decreasing when p-values are sorted."""
        p_values = np.array([0.001, 0.01, 0.02, 0.05, 0.1])
        q_values = benjamini_hochberg(p_values)
        # Original order q-values
        sorted_q = np.sort(q_values)
        for i in range(1, len(sorted_q)):
            assert sorted_q[i] >= sorted_q[i - 1] - 1e-10

    def test_bounded_by_one(self):
        """Q-values should never exceed 1.0."""
        p_values = np.array([0.5, 0.8, 0.9, 0.95, 0.99])
        q_values = benjamini_hochberg(p_values)
        assert np.all(q_values <= 1.0)

    def test_preserves_significance(self):
        """Very small p-values should remain significant after correction."""
        p_values = np.array([1e-10, 0.01, 0.5, 0.9])
        q_values = benjamini_hochberg(p_values)
        assert q_values[0] < 0.05


class TestIdentifyVariantPositions:
    def _make_msa_with_variant(self):
        """Create MSA with one clear variant position."""
        n_reads = 20
        n_positions = 10
        msa = np.full((n_reads, n_positions), BASE_A, dtype=np.uint8)

        # Position 5: 50% A, 50% C (true variant)
        msa[10:, 5] = BASE_C

        # All positions are main
        is_main = np.ones(n_positions, dtype=bool)

        return msa, is_main

    def test_finds_true_variant(self):
        """Should identify a position with 50/50 allele split."""
        msa, is_main = self._make_msa_with_variant()
        variants = identify_variant_positions(
            msa, is_main, platform="pacbio_clr",
            fdr_threshold=0.05, min_coverage=5, min_minor_count=2,
        )
        # Position 5 should be identified
        var_positions = [v.position for v in variants]
        assert 5 in var_positions

    def test_ignores_noise(self):
        """Should not identify positions with only 1 minor allele read."""
        n_reads = 20
        n_positions = 10
        msa = np.full((n_reads, n_positions), BASE_A, dtype=np.uint8)
        # Only 1 read has a different base at position 3
        msa[0, 3] = BASE_G
        is_main = np.ones(n_positions, dtype=bool)

        variants = identify_variant_positions(
            msa, is_main, min_minor_count=2,
        )
        var_positions = [v.position for v in variants]
        assert 3 not in var_positions

    def test_low_coverage_skipped(self):
        """Positions with too few reads should be skipped."""
        msa = np.zeros((3, 10), dtype=np.uint8)
        msa[:, 0] = BASE_A
        msa[1, 0] = BASE_C
        is_main = np.ones(10, dtype=bool)

        variants = identify_variant_positions(
            msa, is_main, min_coverage=5,
        )
        assert len(variants) == 0


class TestBuildVariantMatrix:
    def test_correct_shape(self):
        """Variant matrix should have correct dimensions."""
        from openlong.deconv.positions import VariantPosition

        msa = np.ones((10, 20), dtype=np.uint8)
        vps = [
            VariantPosition(position=3, major_allele=1, minor_alleles=[2],
                            major_freq=0.7, minor_freqs=[0.3], coverage=10,
                            p_value=0.001, q_value=0.01, entropy=0.88),
            VariantPosition(position=8, major_allele=1, minor_alleles=[3],
                            major_freq=0.6, minor_freqs=[0.4], coverage=10,
                            p_value=0.002, q_value=0.02, entropy=0.97),
        ]

        vm = build_variant_matrix(msa, vps)
        assert vm.shape == (10, 2)

    def test_empty_variants(self):
        """Empty variant list should give zero-width matrix."""
        msa = np.ones((10, 20), dtype=np.uint8)
        vm = build_variant_matrix(msa, [])
        assert vm.shape == (10, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
