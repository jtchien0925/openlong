"""Human genome assembly support.

Extends the core viral quasispecies pipeline to handle human genome
scale data. This includes chunked processing, chromosome-level
assembly, and integration with standard genome assembly tools.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Standard human chromosome names
HUMAN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]

DEFAULT_CHUNK_SIZE = 1_000_000  # 1 Mb chunks for processing
DEFAULT_OVERLAP = 50_000  # 50 kb overlap between chunks


@dataclass
class GenomeRegion:
    """A genomic region for processing."""

    chrom: str
    start: int
    end: int
    size: int = 0

    def __post_init__(self):
        self.size = self.end - self.start

    def __str__(self):
        return f"{self.chrom}:{self.start}-{self.end}"


@dataclass
class AssemblyResult:
    """Result from genome assembly of a region."""

    region: GenomeRegion
    contigs: dict[str, str] = field(default_factory=dict)
    n50: int = 0
    coverage: float = 0.0
    variants_called: int = 0


def get_reference_regions(
    reference_fasta: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    chroms: list[str] | None = None,
) -> list[GenomeRegion]:
    """Generate processing regions from a reference genome.

    Splits the reference into overlapping chunks for parallel processing.

    Args:
        reference_fasta: Path to reference FASTA.
        chunk_size: Size of each processing chunk.
        overlap: Overlap between adjacent chunks.
        chroms: Specific chromosomes to process (None = all).

    Returns:
        List of GenomeRegion objects.
    """
    from openlong.io.readers import read_fasta

    ref_seqs = read_fasta(reference_fasta)
    regions = []

    for name, seq in ref_seqs.items():
        chrom = name.split()[0]
        if chroms and chrom not in chroms:
            continue

        seq_len = len(seq)
        start = 0
        while start < seq_len:
            end = min(start + chunk_size, seq_len)
            regions.append(GenomeRegion(chrom=chrom, start=start, end=end))
            start += chunk_size - overlap

    logger.info(
        f"Generated {len(regions)} processing regions "
        f"(chunk_size={chunk_size:,}, overlap={overlap:,})"
    )
    return regions


def process_region(
    region: GenomeRegion,
    bam_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    pipeline_config: dict | None = None,
) -> AssemblyResult:
    """Process a single genomic region through the full pipeline.

    This runs the complete OpenLong pipeline (align → correct →
    deconvolute → call) on a single region.

    Args:
        region: Genomic region to process.
        bam_path: Path to input BAM.
        reference_path: Path to reference FASTA.
        output_dir: Output directory for results.
        pipeline_config: Pipeline configuration dict.

    Returns:
        AssemblyResult for this region.
    """
    from openlong.io.readers import read_bam, read_fasta
    from openlong.align.aligner import build_msa_matrix
    from openlong.correct.indel import iterative_indel_correction, classify_positions
    from openlong.deconv.positions import identify_variant_positions, build_variant_matrix
    from openlong.deconv.cluster import cluster_reads_hierarchical
    from openlong.deconv.consensus import build_all_consensus, export_haplotypes
    from openlong.variants.snv import call_snvs_multi_haplotype

    config = pipeline_config or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing region: {region}")

    # 1. Load reads for this region
    reads = read_bam(
        bam_path,
        min_length=config.get("min_read_length", 2000),
        region=str(region),
    )

    if len(reads.reads) < config.get("min_coverage", 5):
        logger.warning(f"Insufficient coverage at {region}")
        return AssemblyResult(region=region)

    # 2. Load reference sequence for this region
    ref_seqs = read_fasta(reference_path)
    chrom_seq = ref_seqs.get(region.chrom, "")
    region_ref = chrom_seq[region.start : region.end]

    if not region_ref:
        logger.warning(f"No reference sequence for {region}")
        return AssemblyResult(region=region)

    # 3. Build MSA
    msa, read_names = build_msa_matrix(
        reads, region_ref, region_start=region.start, region_end=region.end
    )

    if msa.shape[0] < 2:
        return AssemblyResult(region=region)

    # 4. INDEL correction
    corrected_msa, correction_stats = iterative_indel_correction(
        msa,
        max_iterations=config.get("max_correction_iterations", 3),
        occupancy_threshold=config.get("occupancy_threshold", 0.5),
    )

    # 5. Identify variant positions
    is_main = classify_positions(corrected_msa)
    platform = reads.platform or "pacbio_clr"
    variant_positions = identify_variant_positions(
        corrected_msa,
        is_main,
        platform=platform,
        fdr_threshold=config.get("fdr_threshold", 0.05),
    )

    # 6. Cluster reads into haplotypes
    variant_matrix = build_variant_matrix(corrected_msa, variant_positions)
    clusters = cluster_reads_hierarchical(
        variant_matrix,
        distance_threshold=config.get("cluster_threshold", 0.05),
    )

    # 7. Build consensus for each haplotype
    clusters = build_all_consensus(corrected_msa, clusters, is_main)
    haplotypes = export_haplotypes(clusters, prefix=f"{region.chrom}_{region.start}")

    # 8. Call variants
    variants = call_snvs_multi_haplotype(
        haplotypes, region_ref, chrom=region.chrom, offset=region.start
    )

    result = AssemblyResult(
        region=region,
        contigs=haplotypes,
        coverage=len(reads.reads),
        variants_called=len(variants),
    )

    # Compute contig N50
    if haplotypes:
        lengths = sorted([len(s) for s in haplotypes.values()], reverse=True)
        total = sum(lengths)
        cumsum = 0
        for length in lengths:
            cumsum += length
            if cumsum >= total / 2:
                result.n50 = length
                break

    logger.info(
        f"Region {region}: {len(haplotypes)} haplotypes, "
        f"{len(variants)} variants, N50={result.n50:,}"
    )
    return result


def run_genome_assembly(
    bam_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
    threads: int = 4,
    chroms: list[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    pipeline_config: dict | None = None,
) -> list[AssemblyResult]:
    """Run full genome assembly pipeline.

    Processes the genome in chunks, optionally in parallel.

    Args:
        bam_path: Input BAM path.
        reference_path: Reference FASTA path.
        output_dir: Output directory.
        threads: Number of parallel threads.
        chroms: Specific chromosomes (None = all).
        chunk_size: Processing chunk size.
        pipeline_config: Pipeline configuration.

    Returns:
        List of AssemblyResult objects.
    """
    regions = get_reference_regions(
        reference_path, chunk_size=chunk_size, chroms=chroms
    )

    results = []
    for i, region in enumerate(regions):
        logger.info(f"Processing region {i + 1}/{len(regions)}: {region}")
        result = process_region(
            region, bam_path, reference_path, output_dir, pipeline_config
        )
        results.append(result)

    # Summary
    total_variants = sum(r.variants_called for r in results)
    total_contigs = sum(len(r.contigs) for r in results)
    logger.info(
        f"Genome assembly complete: {len(regions)} regions, "
        f"{total_contigs} contigs, {total_variants} variants"
    )
    return results
