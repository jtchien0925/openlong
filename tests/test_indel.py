"""Tests for the INDEL correction algorithm."""

import numpy as np
import pytest

from openlong.correct.indel import (
    classify_positions,
    correct_indels,
    iterative_indel_correction,
    compute_position_entropy,
    apply_hysteresis,
    enforce_min_main_positions,
    GAP, BASE_A, BASE_C, BASE_G, BASE_T,
)


def make_test_msa(n_reads=10, n_positions=20, seed=42):
    """Create a synthetic MSA for testing."""
    rng = np.random.RandomState(seed)

    # Create a "true" reference with some variant positions
    ref = np.array([BASE_A] * n_positions, dtype=np.uint8)
    ref[5] = BASE_C  # Variant position
    ref[15] = BASE_G  # Variant position

    # Build reads with errors
    msa = np.zeros((n_reads, n_positions), dtype=np.uint8)
    for i in range(n_reads):
        msa[i] = ref.copy()

        # Add random INDEL errors at non-main positions
        for j in range(n_positions):
            if rng.random() < 0.1:  # 10% error rate
                if rng.random() < 0.5:
                    msa[i, j] = GAP  # Deletion
                else:
                    msa[i, j] = rng.choice([BASE_A, BASE_C, BASE_G, BASE_T])

    return msa


class TestClassifyPositions:
    def test_all_occupied(self):
        """All positions with bases should be classified as main."""
        msa = np.ones((5, 10), dtype=np.uint8)  # All A's
        is_main = classify_positions(msa, occupancy_threshold=0.5)
        assert np.all(is_main)

    def test_all_gaps(self):
        """All-gap positions should be INDEL."""
        msa = np.zeros((5, 10), dtype=np.uint8)
        is_main = classify_positions(msa, occupancy_threshold=0.5)
        assert not np.any(is_main)

    def test_mixed_occupancy(self):
        """Positions with partial occupancy should be classified correctly."""
        msa = np.zeros((10, 5), dtype=np.uint8)
        msa[:, 0] = BASE_A  # 100% occupancy
        msa[:5, 1] = BASE_C  # 50% occupancy
        msa[:2, 2] = BASE_G  # 20% occupancy
        msa[:8, 3] = BASE_T  # 80% occupancy

        is_main = classify_positions(msa, occupancy_threshold=0.5)
        assert is_main[0]  # 100% -> main
        assert is_main[1]  # 50% -> main (at threshold)
        assert not is_main[2]  # 20% -> INDEL
        assert is_main[3]  # 80% -> main

    def test_empty_msa(self):
        """Empty MSA should return empty classification."""
        msa = np.zeros((0, 10), dtype=np.uint8)
        is_main = classify_positions(msa)
        assert len(is_main) == 10
        assert not np.any(is_main)


class TestCorrectIndels:
    def test_no_corrections_needed(self):
        """Clean MSA should have zero corrections."""
        msa = np.ones((5, 10), dtype=np.uint8)  # All A's, no INDEL positions
        corrected, stats = correct_indels(msa, occupancy_threshold=0.5)
        assert stats.corrections_made == 0
        assert np.array_equal(corrected, msa)

    def test_indel_correction_applied(self):
        """Bases flanked by INDEL artifacts should be corrected to gaps."""
        # Create MSA where position 2 is INDEL and flanks main positions 1 and 3
        msa = np.zeros((3, 5), dtype=np.uint8)
        # All reads have bases at main positions
        msa[:, 0] = BASE_A
        msa[:, 1] = BASE_C
        msa[:, 3] = BASE_G
        msa[:, 4] = BASE_T

        # Position 2 is low-occupancy (INDEL position)
        # But read 0 has a base there (artifact)
        msa[0, 2] = BASE_T

        is_main = np.array([True, True, False, True, True])
        corrected, stats = correct_indels(msa, is_main=is_main)

        # Read 0 should have its base at position 1 or 3 corrected
        # because position 2 (INDEL) has a base in that read
        assert stats.corrections_made > 0

    def test_stats_correct(self):
        """Correction stats should be accurate."""
        msa = np.ones((5, 10), dtype=np.uint8)
        is_main = np.ones(10, dtype=bool)
        is_main[3] = False  # One INDEL position

        corrected, stats = correct_indels(msa, is_main=is_main)
        assert stats.total_positions == 10
        assert stats.main_positions == 9
        assert stats.indel_positions == 1


class TestIterativeCorrection:
    def test_convergence(self):
        """Iterative correction should converge."""
        msa = make_test_msa(n_reads=20, n_positions=50)
        corrected, all_stats = iterative_indel_correction(
            msa, max_iterations=5, occupancy_threshold=0.5
        )
        assert len(all_stats) > 0
        assert len(all_stats) <= 5

    def test_respects_max_iterations(self):
        """Should not exceed max iterations."""
        msa = make_test_msa()
        _, all_stats = iterative_indel_correction(msa, max_iterations=2)
        assert len(all_stats) <= 2

    def test_prevents_divergence(self):
        """Iterative correction should detect divergence and stop early."""
        msa = make_test_msa(n_reads=20, n_positions=50)
        corrected, all_stats = iterative_indel_correction(
            msa, max_iterations=5, occupancy_threshold=0.5
        )
        # Check that corrections don't monotonically increase (convergence behavior)
        corrections = [s.corrections_made for s in all_stats]
        # After first iteration, corrections should generally decrease or stay flat
        # (not consistently increase)
        if len(corrections) > 1:
            increasing_count = sum(
                1 for i in range(1, len(corrections))
                if corrections[i] > corrections[i - 1]
            )
            # Allow at most 1 increase due to noise, but shouldn't be many
            assert increasing_count <= 1

    def test_maintains_main_positions(self):
        """Main positions should not drop below minimum threshold."""
        msa = make_test_msa(n_reads=20, n_positions=100)
        corrected, all_stats = iterative_indel_correction(
            msa, max_iterations=5, occupancy_threshold=0.5, min_main_positions_fraction=0.8
        )

        # Get initial main position count
        initial_is_main = classify_positions(msa, 0.5)
        initial_main_count = np.sum(initial_is_main)
        min_allowed = int(initial_main_count * 0.8)

        # All iterations should maintain at least 80% of main positions
        for stats in all_stats:
            assert stats.main_positions >= min_allowed

    def test_hysteresis_prevents_reclassification(self):
        """Hysteresis should prevent aggressive reclassification of original main positions."""
        msa = make_test_msa(n_reads=20, n_positions=100)
        initial_is_main = classify_positions(msa, 0.5)
        initial_main_count = np.sum(initial_is_main)

        corrected, all_stats = iterative_indel_correction(
            msa, max_iterations=5, occupancy_threshold=0.5, use_hysteresis=True
        )

        # With hysteresis, we should preserve most original main positions
        # Compare to non-hysteresis version
        corrected_no_hysteresis, all_stats_no_hysteresis = iterative_indel_correction(
            msa, max_iterations=5, occupancy_threshold=0.5, use_hysteresis=False
        )

        # With hysteresis, we should have maintained more main positions
        final_main_with_hysteresis = all_stats[-1].main_positions
        final_main_no_hysteresis = all_stats_no_hysteresis[-1].main_positions

        # Hysteresis should generally preserve main positions better
        # (though not strictly guaranteed in all cases)
        assert final_main_with_hysteresis >= final_main_no_hysteresis * 0.9


class TestPositionEntropy:
    def test_uniform_entropy_zero(self):
        """All-same bases should have zero entropy."""
        msa = np.full((10, 5), BASE_A, dtype=np.uint8)
        entropies = compute_position_entropy(msa)
        assert np.allclose(entropies, 0.0)

    def test_two_alleles_high_entropy(self):
        """50/50 two-allele split should have entropy = 1.0."""
        msa = np.zeros((10, 1), dtype=np.uint8)
        msa[:5, 0] = BASE_A
        msa[5:, 0] = BASE_C
        entropies = compute_position_entropy(msa)
        assert pytest.approx(entropies[0], abs=0.01) == 1.0

    def test_gaps_ignored(self):
        """Gap positions should have zero entropy."""
        msa = np.zeros((10, 5), dtype=np.uint8)
        entropies = compute_position_entropy(msa)
        assert np.allclose(entropies, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
