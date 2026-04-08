"""Main pipeline orchestrator.

Coordinates the full OpenLong pipeline:
1. Read input data (BAM/FASTQ)
2. Align to reference (if needed)
3. Build MSA
4. INDEL correction
5. Identify variant positions
6. Cluster reads into haplotypes
7. Build per-haplotype consensus
8. Call variants (SNVs + SVs)
9. Phase variants
10. Output results (FASTA, VCF, reports)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the OpenLong pipeline."""

    # Input
    input_path: str = ""
    reference_path: str = ""
    output_dir: str = "openlong_output"

    # Read filtering
    min_read_length: int = 2000
    min_mapq: int = 20
    platform: str = "auto"  # auto, pacbio_clr, pacbio_hifi, ont

    # Alignment
    threads: int = 4
    minimap2_preset: str | None = None  # Auto-detect from platform

    # INDEL correction
    occupancy_threshold: float = 0.5
    max_correction_iterations: int = 3
    correction_convergence: float = 0.01

    # Variant position identification
    fdr_threshold: float = 0.05
    min_coverage: int = 5
    min_minor_count: int = 2
    custom_error_rate: float | None = None

    # Clustering
    cluster_distance_threshold: float = 0.05
    min_cluster_size: int = 2
    max_deconv_depth: int = 5

    # Consensus
    min_consensus_coverage: int = 3
    min_consensus_agreement: float = 0.6

    # Variant calling
    min_variant_qual: float = 20.0
    min_sv_size: int = 50

    # Genome mode
    genome_mode: bool = False
    chunk_size: int = 1_000_000
    chroms: list[str] | None = None

    # Region mode
    region: str | None = None  # e.g., 'chr1:1000-2000'


@dataclass
class PipelineResults:
    """Results from a pipeline run."""

    haplotypes: dict[str, str] = field(default_factory=dict)
    variants: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)


def run_pipeline(config: PipelineConfig) -> PipelineResults:
    """Run the full OpenLong pipeline.

    Args:
        config: Pipeline configuration.

    Returns:
        PipelineResults with haplotypes, variants, and statistics.
    """
    from openlong.io.readers import auto_read, read_bam, read_fasta
    from openlong.io.writers import write_fasta, write_vcf, write_report
    from openlong.align.aligner import align_to_reference, build_msa_matrix
    from openlong.correct.indel import (
        classify_positions,
        iterative_indel_correction,
    )
    from openlong.deconv.positions import (
        identify_variant_positions,
        build_variant_matrix,
    )
    from openlong.deconv.cluster import (
        cluster_reads_hierarchical,
        iterative_deconvolution,
        estimate_haplotype_frequencies,
    )
    from openlong.deconv.consensus import build_all_consensus, export_haplotypes
    from openlong.variants.snv import call_snvs_multi_haplotype
    from openlong.variants.sv import detect_deletions, detect_insertions

    start_time = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = PipelineResults()

    # === GENOME MODE ===
    if config.genome_mode:
        from openlong.genome.assembly import run_genome_assembly

        logger.info("Running in genome assembly mode")
        assembly_results = run_genome_assembly(
            bam_path=config.input_path,
            reference_path=config.reference_path,
            output_dir=output_dir / "assembly",
            threads=config.threads,
            chroms=config.chroms,
            chunk_size=config.chunk_size,
            pipeline_config={
                "min_read_length": config.min_read_length,
                "occupancy_threshold": config.occupancy_threshold,
                "max_correction_iterations": config.max_correction_iterations,
                "fdr_threshold": config.fdr_threshold,
                "cluster_threshold": config.cluster_distance_threshold,
                "min_coverage": config.min_coverage,
            },
        )

        # Collect results
        for ar in assembly_results:
            results.haplotypes.update(ar.contigs)
            results.variants.extend([])  # Variants collected per-region

        results.stats["genome_mode"] = True
        results.stats["regions_processed"] = len(assembly_results)
        results.stats["total_contigs"] = sum(len(r.contigs) for r in assembly_results)
        results.stats["total_variants"] = sum(r.variants_called for r in assembly_results)

        elapsed = time.time() - start_time
        results.stats["elapsed_seconds"] = elapsed
        logger.info(f"Genome assembly complete in {elapsed:.1f}s")

        # Write outputs
        _write_outputs(results, config, output_dir)
        return results

    # === STANDARD MODE (viral / targeted) ===
    logger.info("Running in standard mode")

    # Step 1: Load reads
    logger.info("Step 1: Loading reads")
    input_path = Path(config.input_path)

    if input_path.suffix == ".bam":
        reads = read_bam(
            input_path,
            min_length=config.min_read_length,
            min_mapq=config.min_mapq,
            region=config.region,
        )
    else:
        reads = auto_read(input_path, min_length=config.min_read_length)

    if config.platform != "auto":
        reads.platform = config.platform

    results.stats["input_reads"] = len(reads.reads)
    results.stats["input_bases"] = reads.total_bases
    results.stats["input_n50"] = reads.n50
    results.stats["platform"] = reads.platform

    if len(reads.reads) < config.min_coverage:
        logger.error(
            f"Insufficient reads: {len(reads.reads)} < {config.min_coverage}"
        )
        return results

    # Step 2: Align to reference (if reference provided and reads not aligned)
    needs_alignment = config.reference_path and any(
        r.reference_start < 0 for r in reads.reads
    )

    if needs_alignment:
        logger.info("Step 2: Aligning to reference")
        aligned_bam = output_dir / "aligned.bam"
        reads = align_to_reference(
            reads,
            config.reference_path,
            aligned_bam,
            threads=config.threads,
            preset=config.minimap2_preset,
        )
        results.output_files.append(str(aligned_bam))
    else:
        logger.info("Step 2: Reads already aligned, skipping")

    # Step 3: Build MSA
    logger.info("Step 3: Building MSA")
    ref_seqs = {}
    reference_seq = ""

    if config.reference_path:
        ref_seqs = read_fasta(config.reference_path)
        if ref_seqs:
            # Use first sequence or specified chrom
            ref_name = list(ref_seqs.keys())[0]
            reference_seq = ref_seqs[ref_name]

    if not reference_seq:
        # No reference: use longest read as pseudo-reference
        longest = max(reads.reads, key=lambda r: r.read_length)
        reference_seq = longest.sequence
        logger.info(f"Using longest read as pseudo-reference ({longest.read_length} bp)")

    msa, read_names = build_msa_matrix(reads, reference_seq)
    results.stats["msa_shape"] = list(msa.shape)

    if msa.shape[0] < 2:
        logger.error("MSA has fewer than 2 reads")
        return results

    # Step 4: INDEL correction
    logger.info("Step 4: INDEL correction")
    corrected_msa, correction_stats = iterative_indel_correction(
        msa,
        max_iterations=config.max_correction_iterations,
        occupancy_threshold=config.occupancy_threshold,
        convergence_threshold=config.correction_convergence,
    )
    results.stats["correction_iterations"] = len(correction_stats)
    results.stats["total_corrections"] = sum(s.corrections_made for s in correction_stats)

    # Step 5: Identify variant positions
    logger.info("Step 5: Identifying variant positions")
    is_main = classify_positions(corrected_msa, config.occupancy_threshold)
    variant_positions = identify_variant_positions(
        corrected_msa,
        is_main,
        platform=reads.platform,
        fdr_threshold=config.fdr_threshold,
        min_coverage=config.min_coverage,
        min_minor_count=config.min_minor_count,
        custom_error_rate=config.custom_error_rate,
    )
    results.stats["variant_positions"] = len(variant_positions)
    results.stats["main_positions"] = int(np.sum(is_main))

    # Step 6: Cluster reads
    logger.info("Step 6: Clustering reads into haplotypes")
    variant_matrix = build_variant_matrix(corrected_msa, variant_positions)

    if config.max_deconv_depth > 1 and len(variant_positions) > 0:
        clusters = iterative_deconvolution(
            corrected_msa,
            variant_matrix,
            max_depth=config.max_deconv_depth,
            distance_threshold=config.cluster_distance_threshold,
            min_cluster_size=config.min_cluster_size,
        )
    elif len(variant_positions) > 0:
        clusters = cluster_reads_hierarchical(
            variant_matrix,
            distance_threshold=config.cluster_distance_threshold,
            min_cluster_size=config.min_cluster_size,
        )
    else:
        from openlong.deconv.cluster import HaplotypeCluster

        clusters = [
            HaplotypeCluster(
                cluster_id=1,
                read_indices=list(range(corrected_msa.shape[0])),
            )
        ]

    results.stats["haplotypes_found"] = len(clusters)
    freqs = estimate_haplotype_frequencies(clusters)
    results.stats["haplotype_frequencies"] = freqs

    # Step 7: Build consensus
    logger.info("Step 7: Building haplotype consensus sequences")
    clusters = build_all_consensus(
        corrected_msa,
        clusters,
        is_main,
        min_coverage=config.min_consensus_coverage,
        min_agreement=config.min_consensus_agreement,
    )
    results.haplotypes = export_haplotypes(clusters)
    results.stats["consensus_qualities"] = {
        c.cluster_id: c.mean_qv for c in clusters
    }

    # Step 8: Call variants
    logger.info("Step 8: Calling variants")
    if reference_seq:
        qualities_dict = {
            name: cluster.quality
            for cluster in clusters
            for name in [
                f"haplotype_{cluster.cluster_id}"
                f"_reads={cluster.n_reads}"
                f"_qv={cluster.mean_qv:.0f}"
            ]
        }
        snvs = call_snvs_multi_haplotype(
            results.haplotypes, reference_seq, qualities=qualities_dict
        )

        # SVs
        svs_del = detect_deletions(corrected_msa, is_main, reference_seq)
        svs_ins = detect_insertions(corrected_msa, is_main)
        sv_variants = [sv.to_vcf_dict() for sv in svs_del + svs_ins]

        results.variants = snvs + sv_variants
        results.stats["snvs"] = len(snvs)
        results.stats["svs"] = len(sv_variants)

    # Write outputs
    elapsed = time.time() - start_time
    results.stats["elapsed_seconds"] = elapsed
    logger.info(f"Pipeline complete in {elapsed:.1f}s")

    _write_outputs(results, config, output_dir)
    return results


def _write_outputs(
    results: PipelineResults,
    config: PipelineConfig,
    output_dir: Path,
) -> None:
    """Write all output files."""
    from openlong.io.writers import write_fasta, write_vcf, write_report

    # Haplotype FASTA
    if results.haplotypes:
        fasta_path = output_dir / "haplotypes.fasta"
        write_fasta(results.haplotypes, fasta_path)
        results.output_files.append(str(fasta_path))

    # Variants VCF
    if results.variants:
        vcf_path = output_dir / "variants.vcf"
        write_vcf(results.variants, vcf_path)
        results.output_files.append(str(vcf_path))

    # Report
    report_path = output_dir / "report.json"
    write_report(results.stats, report_path)
    results.output_files.append(str(report_path))

    logger.info(f"Output files: {results.output_files}")
