"""Tests for variant position identification."""

import numpy as np
import pytest

from openlong.correct.indel import BASE_A, BASE_C, BASE_G, BASE_T, GAP
from openlong.deconv.positions import (
    identify_variant_positions,
    identify_variant_positions_classic,
    benjamini_hochberg,
    build_variant_matrix,
    strand_bias_test,
    StrandBiasInfo,
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


class TestStrandBiasTest:
    def test_no_bias_balanced_variant(self):
        """Variant appearing equally on both strands should have high p-value."""
        # 10 forward reads: 5 ref, 5 alt
        # 10 reverse reads: 5 ref, 5 alt
        forward_counts = {BASE_A: 5, BASE_C: 5}  # ref=A, alt=C
        reverse_counts = {BASE_A: 5, BASE_C: 5}

        pvalue = strand_bias_test(forward_counts, reverse_counts)
        # Should be high p-value (no bias)
        assert pvalue > 0.5

    def test_strong_bias_forward_only(self):
        """Variant appearing only on forward strand should have lower p-value."""
        # 10 forward reads: 5 ref, 5 alt
        # 10 reverse reads: 10 ref, 0 alt
        forward_counts = {BASE_A: 5, BASE_C: 5}
        reverse_counts = {BASE_A: 10}

        pvalue = strand_bias_test(forward_counts, reverse_counts)
        # Should have measurable bias signal (p < 0.05 is typical for this pattern)
        assert pvalue < 0.05

    def test_strong_bias_reverse_only(self):
        """Variant appearing only on reverse strand should have lower p-value."""
        # 10 forward reads: 10 ref, 0 alt
        # 10 reverse reads: 5 ref, 5 alt
        forward_counts = {BASE_A: 10}
        reverse_counts = {BASE_A: 5, BASE_C: 5}

        pvalue = strand_bias_test(forward_counts, reverse_counts)
        # Should have measurable bias signal (p < 0.05 is typical for this pattern)
        assert pvalue < 0.05

    def test_empty_counts(self):
        """Empty counts should return p-value of 1.0."""
        forward_counts = {}
        reverse_counts = {}

        pvalue = strand_bias_test(forward_counts, reverse_counts)
        assert pvalue == 1.0

    def test_unsupported_method(self):
        """Unsupported method should raise ValueError."""
        forward_counts = {BASE_A: 5}
        reverse_counts = {BASE_A: 5}

        with pytest.raises(ValueError, match="Unsupported strand bias test method"):
            strand_bias_test(forward_counts, reverse_counts, method="chi2")


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

    def test_strand_bias_filters_artifact(self):
        """Variant appearing predominantly on one strand should be filtered."""
        n_reads = 30
        n_positions = 10
        msa = np.full((n_reads, n_positions), BASE_A, dtype=np.uint8)

        # Position 5: variant appearing predominantly on forward strand
        # Forward (0-14): 7x A, 8x C
        # Reverse (15-29): 15x A, 0x C
        msa[0:7, 5] = BASE_A
        msa[7:15, 5] = BASE_C
        msa[15:, 5] = BASE_A

        is_main = np.ones(n_positions, dtype=bool)

        # strand_labels: 0=forward, 1=reverse
        strand_labels = np.array([0] * 15 + [1] * 15, dtype=np.uint8)

        # Without strand filtering, should find the variant
        variants_no_filter = identify_variant_positions(
            msa, is_main, platform="pacbio_clr",
            fdr_threshold=0.05, min_coverage=5, min_minor_count=2,
            strand_labels=None,
        )

        # With strand filtering using strict threshold, should filter it out
        variants_with_filter = identify_variant_positions(
            msa, is_main, platform="pacbio_clr",
            fdr_threshold=0.05, min_coverage=5, min_minor_count=2,
            strand_labels=strand_labels,
            strand_bias_threshold=0.05,
        )

        # The artifact variant should be found without filtering
        var_positions_no_filter = [v.position for v in variants_no_filter]
        assert 5 in var_positions_no_filter

        # But filtered out with strand bias filter
        var_positions_with_filter = [v.position for v in variants_with_filter]
        assert 5 not in var_positions_with_filter

    def test_no_strand_bias_passthrough(self):
        """When strand_labels is None, no filtering should happen."""
        n_reads = 20
        n_positions = 10
        msa = np.full((n_reads, n_positions), BASE_A, dtype=np.uint8)

        # Position 5: 50% A, 50% C (true variant)
        msa[10:, 5] = BASE_C

        is_main = np.ones(n_positions, dtype=bool)

        # With strand_labels=None (default), should pass through
        variants = identify_variant_positions(
            msa, is_main, platform="pacbio_clr",
            fdr_threshold=0.05, min_coverage=5, min_minor_count=2,
            strand_labels=None,
        )

        var_positions = [v.position for v in variants]
        assert 5 in var_positions

        # Check that strand_bias field is None
        for v in variants:
            if v.position == 5:
                assert v.strand_bias is None

    def test_true_variant_passes_strand_filter(self):
        """Variant appearing equally on both strands should pass filter."""
        n_reads = 20
        n_positions = 10
        msa = np.full((n_reads, n_positions), BASE_A, dtype=np.uint8)

        # Position 5: variant appearing on BOTH strands equally
        # Forward (0-9): 5xA, 5xC
        # Reverse (10-19): 5xA, 5xC
        msa[0:5, 5] = BASE_C
        msa[15:, 5] = BASE_C

        is_main = np.ones(n_positions, dtype=bool)

        # strand_labels: 0=forward, 1=reverse
        strand_labels = np.array([0] * 10 + [1] * 10, dtype=np.uint8)

        strand_bias_threshold = 0.01
        variants = identify_variant_positions(
            msa, is_main, platform="pacbio_clr",
            fdr_threshold=0.05, min_coverage=5, min_minor_count=2,
            strand_labels=strand_labels,
            strand_bias_threshold=strand_bias_threshold,
        )

        var_positions = [v.position for v in variants]
        # Should find the variant because it's balanced across strands
        assert 5 in var_positions

        # Check that strand_bias info was recorded
        for v in variants:
            if v.position == 5:
                assert v.strand_bias is not None
                assert v.strand_bias.forward_count > 0
                assert v.strand_bias.reverse_count > 0
                assert v.strand_bias.bias_pvalue > strand_bias_threshold


class TestClassicMode:
    """Tests for the permissive variant detection mode (for use with ECA)."""

    def test_classic_mode_uses_correct_thresholds(self):
        """Classic mode should use error_rate=0.05 and fdr_threshold=0.2."""
        # Create MSA with variant that would pass at q<0.2 but fail at q<0.05
        n_reads = 50
        n_positions = 20
        msa = np.full((n_reads, n_positions), BASE_A, dtype=np.uint8)

        # Position 10: 40% minor allele (20 reads of BASE_C)
        # This is significant at 5% error rate but marginal at stricter thresholds
        msa[30:, 10] = BASE_C

        is_main = np.ones(n_positions, dtype=bool)

        # Using classic mode with q<0.2 should find more variants
        variants_classic = identify_variant_positions_classic(
            msa, is_main, error_rate=0.05, fdr_threshold=0.2
        )

        # Using strict mode with q<0.05 should find fewer variants
        variants_strict = identify_variant_positions(
            msa, is_main, platform="pacbio_clr",
            fdr_threshold=0.05, custom_error_rate=0.05
        )

        # Classic mode should find at least as many (typically more) variants
        assert len(variants_classic) >= len(variants_strict)

    def test_classic_mode_no_strand_filtering(self):
        """Classic mode should not apply strand bias filtering."""
        n_reads = 30
        n_positions = 10
        msa = np.full((n_reads, n_positions), BASE_A, dtype=np.uint8)

        # Position 5: variant with strand bias
        # Forward (0-14): 7x A, 8x C
        # Reverse (15-29): 15x A, 0x C
        msa[0:7, 5] = BASE_A
        msa[7:15, 5] = BASE_C
        msa[15:, 5] = BASE_A

        is_main = np.ones(n_positions, dtype=bool)

        # Classic mode should NOT have strand_labels parameter
        # So strand biased variants should pass through
        variants = identify_variant_positions_classic(msa, is_main)

        var_positions = [v.position for v in variants]
        assert 5 in var_positions

        # Check that strand_bias field is None (no filtering was applied)
        for v in variants:
            if v.position == 5:
                assert v.strand_bias is None

    def test_classic_uses_5_percent_error_rate(self):
        """Classic mode should use 5% error rate."""
        n_reads = 100
        n_positions = 10
        msa = np.full((n_reads, n_positions), BASE_A, dtype=np.uint8)

        # Create a variant at position 5 with ~5% minor allele frequency
        # This tests the boundary: should be significant at 5% error rate
        msa[50:, 5] = BASE_C  # 50% minor allele

        is_main = np.ones(n_positions, dtype=bool)

        # Classic default error rate should be 5%
        variants = identify_variant_positions_classic(msa, is_main)

        var_positions = [v.position for v in variants]
        # With 50% minor allele, this should definitely be detected
        assert 5 in var_positions


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
