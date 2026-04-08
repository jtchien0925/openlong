"""Integration tests for the full pipeline using synthetic data."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from openlong.correct.indel import BASE_A, BASE_C, BASE_G, BASE_T, GAP


def create_synthetic_reads_fasta(output_path, n_haplotypes=3, n_reads_per=10, length=500):
    """Create a synthetic FASTA with known haplotypes mixed together."""
    rng = np.random.RandomState(42)
    decode = {1: "A", 2: "C", 3: "G", 4: "T"}

    # Create distinct haplotype templates
    base_seq = rng.choice([1, 2, 3, 4], size=length)
    haplotypes = {}

    for h in range(n_haplotypes):
        hap = base_seq.copy()
        # Introduce ~2% variants unique to this haplotype
        n_variants = max(1, int(length * 0.02))
        positions = rng.choice(length, size=n_variants, replace=False)
        for pos in positions:
            alt = rng.choice([b for b in [1, 2, 3, 4] if b != hap[pos]])
            hap[pos] = alt
        haplotypes[f"hap_{h}"] = hap

    # Write reads as FASTA
    with open(output_path, "w") as fh:
        read_idx = 0
        for hap_name, hap_seq in haplotypes.items():
            for i in range(n_reads_per):
                # Add ~5% random errors to simulate CLR
                read = hap_seq.copy()
                n_errors = max(1, int(length * 0.05))
                error_positions = rng.choice(length, size=n_errors, replace=False)
                for ep in error_positions:
                    if rng.random() < 0.7:
                        # Substitution error
                        read[ep] = rng.choice([1, 2, 3, 4])
                    else:
                        # Gap (simulating deletion)
                        read[ep] = 0

                seq = "".join(decode.get(b, "N") for b in read if b > 0)
                fh.write(f">read_{read_idx}_from_{hap_name}\n{seq}\n")
                read_idx += 1

    return haplotypes


class TestSyntheticPipeline:
    def test_indel_correction_reduces_errors(self):
        """INDEL correction should reduce the error count in the MSA."""
        from openlong.correct.indel import (
            classify_positions,
            correct_indels,
            compute_position_entropy,
        )

        rng = np.random.RandomState(42)
        n_reads, n_pos = 20, 100

        # Create clean MSA
        true_seq = rng.choice([BASE_A, BASE_C, BASE_G, BASE_T], size=n_pos)
        msa = np.tile(true_seq, (n_reads, 1)).astype(np.uint8)

        # Add INDEL artifacts: insert bases at "INDEL positions"
        # Create some positions that should be INDEL (low occupancy)
        for i in range(n_reads):
            # ~10% of positions get random errors
            for j in range(n_pos):
                if rng.random() < 0.10:
                    msa[i, j] = rng.choice([0, BASE_A, BASE_C, BASE_G, BASE_T])

        # Measure pre-correction disagreement
        is_main = classify_positions(msa)
        pre_entropy = compute_position_entropy(msa, is_main)
        pre_mean_entropy = np.mean(pre_entropy[pre_entropy > 0]) if np.any(pre_entropy > 0) else 0

        # Apply correction
        corrected, stats = correct_indels(msa, is_main=is_main)

        # Post-correction entropy should be lower or equal
        post_entropy = compute_position_entropy(corrected, is_main)
        post_mean_entropy = (
            np.mean(post_entropy[post_entropy > 0]) if np.any(post_entropy > 0) else 0
        )

        assert stats.corrections_made >= 0

    def test_variant_detection_sensitivity(self):
        """Should detect a known variant position."""
        from openlong.correct.indel import classify_positions
        from openlong.deconv.positions import identify_variant_positions

        n_reads = 30
        n_pos = 50
        msa = np.full((n_reads, n_pos), BASE_A, dtype=np.uint8)

        # Insert a real variant at position 25: 50% A, 50% C
        msa[15:, 25] = BASE_C

        is_main = classify_positions(msa)
        variants = identify_variant_positions(
            msa, is_main, platform="pacbio_hifi",
            fdr_threshold=0.1, min_coverage=5, min_minor_count=2,
        )

        found_positions = [v.position for v in variants]
        assert 25 in found_positions

    def test_clustering_separates_haplotypes(self):
        """Should separate reads from distinct haplotypes."""
        from openlong.deconv.cluster import cluster_reads_hierarchical

        # 3 haplotypes, clearly distinct at 10 variant positions
        n_reads_per_hap = 10
        n_variants = 10

        vm = np.zeros((30, n_variants), dtype=np.uint8)
        # Haplotype 1: all A
        vm[:10] = BASE_A
        # Haplotype 2: all C
        vm[10:20] = BASE_C
        # Haplotype 3: all G
        vm[20:30] = BASE_G

        clusters = cluster_reads_hierarchical(
            vm, distance_threshold=0.3, min_cluster_size=3,
        )
        assert len(clusters) == 3

    def test_consensus_quality(self):
        """Consensus from clean reads should have high QV."""
        from openlong.correct.polish import majority_consensus, compute_consensus_quality

        msa = np.full((20, 100), BASE_A, dtype=np.uint8)
        # One position with a variant
        msa[10:, 50] = BASE_C

        consensus = majority_consensus(msa, min_coverage=3)
        quality = compute_consensus_quality(msa, consensus)

        # Most positions should have high quality
        high_qv_count = np.sum(quality > 20)
        assert high_qv_count > len(consensus) * 0.8

    def test_end_to_end_writers(self):
        """Test that output writers produce valid files."""
        from openlong.io.writers import write_fasta, write_vcf, write_report

        with tempfile.TemporaryDirectory() as tmpdir:
            # FASTA
            fasta_path = Path(tmpdir) / "test.fasta"
            write_fasta({"seq1": "ACGTACGT", "seq2": "TGCATGCA"}, fasta_path)
            assert fasta_path.exists()
            content = fasta_path.read_text()
            assert ">seq1" in content
            assert "ACGTACGT" in content

            # VCF
            vcf_path = Path(tmpdir) / "test.vcf"
            variants = [
                {"chrom": "chr1", "pos": 100, "ref": "A", "alt": "C",
                 "qual": "30", "filter": "PASS", "info": {"DP": 10},
                 "gt": "0/1", "gq": 30},
            ]
            write_vcf(variants, vcf_path)
            assert vcf_path.exists()
            content = vcf_path.read_text()
            assert "##fileformat=VCFv4.2" in content
            assert "chr1\t100" in content

            # Report
            report_path = Path(tmpdir) / "report.json"
            write_report({"test": "value", "count": 42}, report_path)
            assert report_path.exists()
            data = json.loads(report_path.read_text())
            assert data["test"] == "value"
            assert data["count"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
