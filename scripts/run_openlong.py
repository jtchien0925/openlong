#!/usr/bin/env python3
"""OpenLong CLI entry point.

Usage:
    openlong run --input reads.bam --reference ref.fasta --output results/
    openlong run --input reads.bam --reference ref.fasta --genome --output results/
"""

from __future__ import annotations

import logging
import sys

import click

from openlong import __version__
from openlong.pipeline import PipelineConfig, run_pipeline


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(__version__)
def cli():
    """OpenLong - Open Source Long-Read Sequencing Pipeline.

    Deconvolute closely-related genomic variants from PacBio and
    Oxford Nanopore long-read sequencing data with >QV50 accuracy.
    """
    pass


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Input BAM or FASTQ file")
@click.option("--reference", "-r", "reference_path", default="", help="Reference FASTA file")
@click.option("--output", "-o", "output_dir", default="openlong_output", help="Output directory")
@click.option("--threads", "-t", default=4, help="Number of threads")
@click.option(
    "--platform",
    type=click.Choice(["auto", "pacbio_clr", "pacbio_hifi", "ont"]),
    default="auto",
    help="Sequencing platform",
)
@click.option("--min-length", default=2000, help="Minimum read length")
@click.option("--min-mapq", default=20, help="Minimum mapping quality")
@click.option("--fdr", default=0.05, help="FDR threshold for variant positions")
@click.option("--cluster-threshold", default=0.05, help="Distance threshold for clustering")
@click.option("--genome", is_flag=True, help="Enable genome assembly mode")
@click.option("--chunk-size", default=1_000_000, help="Chunk size for genome mode")
@click.option("--chroms", default=None, help="Comma-separated chromosome list")
@click.option("--region", default=None, help="Genomic region (e.g., chr1:1000-2000)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def run(
    input_path,
    reference_path,
    output_dir,
    threads,
    platform,
    min_length,
    min_mapq,
    fdr,
    cluster_threshold,
    genome,
    chunk_size,
    chroms,
    region,
    verbose,
):
    """Run the full OpenLong pipeline."""
    setup_logging(verbose)
    logger = logging.getLogger("openlong")

    logger.info(f"OpenLong v{__version__}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Reference: {reference_path or '(none)'}")
    logger.info(f"Output: {output_dir}")

    chrom_list = chroms.split(",") if chroms else None

    config = PipelineConfig(
        input_path=input_path,
        reference_path=reference_path,
        output_dir=output_dir,
        threads=threads,
        platform=platform,
        min_read_length=min_length,
        min_mapq=min_mapq,
        fdr_threshold=fdr,
        cluster_distance_threshold=cluster_threshold,
        genome_mode=genome,
        chunk_size=chunk_size,
        chroms=chrom_list,
        region=region,
    )

    results = run_pipeline(config)

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("OpenLong Pipeline Summary")
    click.echo("=" * 60)
    click.echo(f"  Haplotypes found:  {len(results.haplotypes)}")
    click.echo(f"  Variants called:   {len(results.variants)}")
    click.echo(f"  Elapsed time:      {results.stats.get('elapsed_seconds', 0):.1f}s")
    click.echo(f"  Output directory:  {output_dir}")
    for f in results.output_files:
        click.echo(f"    - {f}")
    click.echo("=" * 60)


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Input BAM or FASTQ")
@click.option("--reference", "-r", "reference_path", required=True, help="Reference FASTA")
@click.option("--output", "-o", "output_bam", required=True, help="Output BAM")
@click.option("--threads", "-t", default=4, help="Number of threads")
@click.option(
    "--platform",
    type=click.Choice(["auto", "pacbio_clr", "pacbio_hifi", "ont"]),
    default="auto",
)
@click.option("--verbose", "-v", is_flag=True)
def align(input_path, reference_path, output_bam, threads, platform, verbose):
    """Align reads to a reference genome."""
    setup_logging(verbose)
    from openlong.io.readers import auto_read
    from openlong.align.aligner import align_to_reference

    reads = auto_read(input_path)
    if platform != "auto":
        reads.platform = platform
    align_to_reference(reads, reference_path, output_bam, threads=threads)
    click.echo(f"Aligned BAM written to: {output_bam}")


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Input BAM")
@click.option("--reference", "-r", "reference_path", required=True, help="Reference FASTA")
@click.option("--output", "-o", "output_dir", required=True, help="Output directory")
@click.option("--verbose", "-v", is_flag=True)
def correct(input_path, reference_path, output_dir, verbose):
    """Run INDEL correction on aligned reads."""
    setup_logging(verbose)
    from pathlib import Path
    import numpy as np
    from openlong.io.readers import read_bam, read_fasta
    from openlong.align.aligner import build_msa_matrix
    from openlong.correct.indel import iterative_indel_correction

    logger = logging.getLogger("openlong")

    # Load reads and reference
    reads = read_bam(input_path)
    logger.info(f"Loaded {len(reads.reads)} reads")

    ref_seqs = read_fasta(reference_path)
    if not ref_seqs:
        click.echo("Error: No sequences found in reference FASTA", err=True)
        sys.exit(1)
    reference_seq = list(ref_seqs.values())[0]

    # Build MSA and run INDEL correction
    msa, read_names, ref_to_msa = build_msa_matrix(reads, reference_seq)
    corrected_msa, correction_stats = iterative_indel_correction(msa)

    # Save corrected MSA
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    msa_file = output_path / "corrected_msa.npz"
    np.savez(msa_file, msa=corrected_msa)

    # Print statistics
    total_corrections = sum(s.corrections_made for s in correction_stats)
    click.echo(f"Correction iterations: {len(correction_stats)}")
    click.echo(f"Total corrections made: {total_corrections}")
    click.echo(f"Corrected MSA saved to: {msa_file}")


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="Input BAM")
@click.option("--reference", "-r", "reference_path", required=True, help="Reference FASTA")
@click.option("--output", "-o", "output_dir", required=True, help="Output directory")
@click.option("--verbose", "-v", is_flag=True)
def deconv(input_path, reference_path, output_dir, verbose):
    """Deconvolute reads into haplotypes."""
    setup_logging(verbose)
    from pathlib import Path
    from openlong.io.readers import read_bam, read_fasta
    from openlong.io.writers import write_fasta
    from openlong.align.aligner import build_msa_matrix
    from openlong.correct.indel import iterative_indel_correction
    from openlong.deconv.positions import identify_variant_positions, build_variant_matrix
    from openlong.deconv.cluster import iterative_deconvolution
    from openlong.deconv.consensus import build_all_consensus, export_haplotypes

    logger = logging.getLogger("openlong")

    # Load reads and reference
    reads = read_bam(input_path)
    logger.info(f"Loaded {len(reads.reads)} reads")

    ref_seqs = read_fasta(reference_path)
    if not ref_seqs:
        click.echo("Error: No sequences found in reference FASTA", err=True)
        sys.exit(1)
    reference_seq = list(ref_seqs.values())[0]

    # Build MSA and run INDEL correction
    msa, read_names, ref_to_msa = build_msa_matrix(reads, reference_seq)
    corrected_msa, correction_stats = iterative_indel_correction(msa)

    # Identify variant positions and build variant matrix
    variant_positions = identify_variant_positions(corrected_msa, platform=reads.platform)
    variant_matrix = build_variant_matrix(corrected_msa, variant_positions)

    # Cluster reads into haplotypes
    clusters = iterative_deconvolution(corrected_msa, variant_matrix)

    # Build consensus sequences
    clusters = build_all_consensus(corrected_msa, clusters)
    haplotypes = export_haplotypes(clusters)

    # Write output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fasta_file = output_path / "haplotypes.fasta"
    write_fasta(haplotypes, fasta_file)

    # Print statistics
    frequencies = {i: len(c.read_indices) / len(read_names) for i, c in enumerate(clusters)}
    click.echo(f"Haplotypes found: {len(haplotypes)}")
    click.echo(f"Haplotype frequencies: {frequencies}")
    click.echo(f"Haplotypes written to: {fasta_file}")


@cli.command()
@click.option("--input", "-i", "input_dir", required=True, help="Input haplotype directory or FASTA file")
@click.option("--reference", "-r", "reference_path", required=True, help="Reference FASTA")
@click.option("--output", "-o", "output_vcf", required=True, help="Output VCF")
@click.option("--verbose", "-v", is_flag=True)
def call(input_dir, reference_path, output_vcf, verbose):
    """Call variants from reconstructed haplotypes."""
    setup_logging(verbose)
    from pathlib import Path
    from openlong.io.readers import read_fasta
    from openlong.io.writers import write_vcf
    from openlong.variants.snv import call_snvs_multi_haplotype

    logger = logging.getLogger("openlong")

    # Handle input: can be directory with FASTA files or a single FASTA file
    input_path = Path(input_dir)
    haplotypes = {}

    if input_path.is_file():
        # Single FASTA file
        if not input_path.suffix.lower() in [".fasta", ".fa", ".fasta.gz", ".fa.gz"]:
            click.echo(f"Error: Input file must be FASTA (.fasta or .fa)", err=True)
            sys.exit(1)
        haplotypes = read_fasta(input_path)
        logger.info(f"Loaded {len(haplotypes)} sequences from {input_path}")
    elif input_path.is_dir():
        # Directory with FASTA files
        import glob
        fasta_files = list(input_path.glob("*.fasta")) + list(input_path.glob("*.fa"))
        if not fasta_files:
            click.echo(f"Error: No FASTA files (.fasta or .fa) found in {input_dir}", err=True)
            sys.exit(1)
        for fasta_file in fasta_files:
            seqs = read_fasta(fasta_file)
            haplotypes.update(seqs)
        logger.info(f"Loaded {len(haplotypes)} sequences from {len(fasta_files)} files in {input_dir}")
    else:
        click.echo(f"Error: Input path not found: {input_dir}", err=True)
        sys.exit(1)

    # Load reference
    ref_seqs = read_fasta(reference_path)
    if not ref_seqs:
        click.echo(f"Error: No sequences found in reference FASTA", err=True)
        sys.exit(1)
    ref_seq = list(ref_seqs.values())[0]

    # Call variants
    if not haplotypes:
        click.echo("Error: No haplotype sequences to process", err=True)
        sys.exit(1)

    variants = call_snvs_multi_haplotype(haplotypes, ref_seq)
    write_vcf(variants, output_vcf)
    click.echo(f"Called {len(variants)} variants -> {output_vcf}")


if __name__ == "__main__":
    cli()
