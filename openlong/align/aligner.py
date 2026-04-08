"""Alignment module.

Handles alignment of long reads to a reference genome or to a
consensus sequence, using minimap2 as the backend aligner.
Generates the multiple sequence alignment (MSA) view needed
by downstream correction and deconvolution steps.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from openlong.io.readers import LongRead, ReadCollection, read_bam

logger = logging.getLogger(__name__)

MINIMAP2_PRESETS = {
    "pacbio_clr": "map-pb",
    "pacbio_hifi": "map-hifi",
    "ont": "map-ont",
    "unknown": "map-pb",
}


def get_minimap2_path() -> str:
    """Get path to minimap2 binary."""
    custom = os.environ.get("OPENLONG_MINIMAP2")
    if custom and Path(custom).exists():
        return custom
    return "minimap2"


def align_to_reference(
    reads: ReadCollection,
    reference: str | Path,
    output_bam: str | Path,
    threads: int = 4,
    preset: str | None = None,
) -> ReadCollection:
    """Align reads to a reference genome using minimap2.

    Args:
        reads: Input read collection.
        reference: Path to reference FASTA.
        output_bam: Path for output sorted BAM.
        threads: Number of threads.
        preset: minimap2 preset (auto-detected from platform if None).

    Returns:
        ReadCollection with alignment info populated.
    """
    reference = Path(reference)
    output_bam = Path(output_bam)
    output_bam.parent.mkdir(parents=True, exist_ok=True)

    if preset is None:
        preset = MINIMAP2_PRESETS.get(reads.platform, "map-pb")

    logger.info(
        f"Aligning {len(reads.reads)} reads to {reference} "
        f"(preset={preset}, threads={threads})"
    )

    # Write reads to temporary FASTQ
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fastq", delete=False
    ) as tmp_fq:
        for read in reads.reads:
            qual_str = ""
            if read.quality is not None:
                qual_str = "".join(chr(q + 33) for q in read.quality)
            else:
                qual_str = "I" * read.read_length  # Placeholder quality
            tmp_fq.write(f"@{read.name}\n{read.sequence}\n+\n{qual_str}\n")
        tmp_fastq_path = tmp_fq.name

    try:
        minimap2 = get_minimap2_path()

        # minimap2 → SAM → samtools sort → BAM
        mm2_cmd = [
            minimap2,
            "-a",  # Output SAM
            f"-x{preset}",
            f"-t{threads}",
            "--secondary=no",
            "--eqx",  # Use =/X CIGAR operators
            str(reference),
            tmp_fastq_path,
        ]

        sort_cmd = [
            "samtools",
            "sort",
            f"-@{threads}",
            "-o",
            str(output_bam),
        ]

        logger.info(f"Running: {' '.join(mm2_cmd)} | {' '.join(sort_cmd)}")

        mm2_proc = subprocess.Popen(mm2_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sort_proc = subprocess.Popen(
            sort_cmd, stdin=mm2_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        mm2_proc.stdout.close()
        sort_out, sort_err = sort_proc.communicate()

        if sort_proc.returncode != 0:
            raise RuntimeError(f"samtools sort failed: {sort_err.decode()}")

        # Index the BAM
        subprocess.run(["samtools", "index", str(output_bam)], check=True)

        logger.info(f"Alignment complete: {output_bam}")

    finally:
        os.unlink(tmp_fastq_path)

    # Read back aligned BAM
    return read_bam(output_bam, min_length=0)


def build_msa_matrix(
    reads: ReadCollection,
    reference_seq: str,
    region_start: int = 0,
    region_end: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a multiple sequence alignment matrix from aligned reads.

    This generates the alignment view described in Dilernia et al. 2015:
    each read becomes a row vector where elements are aligned to the
    reference positions.

    Args:
        reads: Aligned ReadCollection.
        reference_seq: Reference sequence string.
        region_start: Start position in reference.
        region_end: End position in reference.

    Returns:
        Tuple of (MSA matrix as numpy uint8 array, list of read names).
        Matrix encoding: A=1, C=2, G=3, T=4, gap=0, N=5
    """
    if region_end is None:
        region_end = len(reference_seq)

    ref_len = region_end - region_start
    base_map = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5, "-": 0}

    # Filter reads overlapping region
    aligned_reads = [
        r
        for r in reads.reads
        if r.reference_start >= 0
        and r.reference_start < region_end
        and r.reference_end > region_start
    ]

    if not aligned_reads:
        logger.warning("No reads overlap the specified region")
        return np.zeros((0, ref_len), dtype=np.uint8), []

    n_reads = len(aligned_reads)
    msa = np.zeros((n_reads, ref_len), dtype=np.uint8)
    read_names = []

    for i, read in enumerate(aligned_reads):
        read_names.append(read.name)
        seq = read.sequence.upper()

        # Simple pairwise alignment projection onto reference coordinates
        # In production, this would parse the CIGAR string properly
        read_offset = max(0, region_start - read.reference_start)
        ref_offset = max(0, read.reference_start - region_start)

        copy_len = min(
            len(seq) - read_offset,
            ref_len - ref_offset,
        )

        for j in range(copy_len):
            base = seq[read_offset + j] if (read_offset + j) < len(seq) else "-"
            msa[i, ref_offset + j] = base_map.get(base, 0)

    logger.info(
        f"Built MSA matrix: {n_reads} reads x {ref_len} positions "
        f"(region {region_start}-{region_end})"
    )
    return msa, read_names


def parse_cigar_to_alignment(
    cigar_tuples: list[tuple[int, int]],
    query_seq: str,
    ref_start: int,
    ref_length: int,
) -> np.ndarray:
    """Parse CIGAR tuples into a reference-coordinate aligned array.

    This is critical for proper MSA construction. CIGAR operations:
    0/7 = M/= (match), 1 = I (insertion), 2 = D (deletion),
    4 = S (soft clip), 8 = X (mismatch)

    Args:
        cigar_tuples: List of (operation, length) from pysam.
        query_seq: Query sequence string.
        ref_start: Reference start position.
        ref_length: Total reference length for output array.

    Returns:
        1D numpy array of base encodings aligned to reference coordinates.
    """
    base_map = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
    result = np.zeros(ref_length, dtype=np.uint8)

    query_pos = 0
    ref_pos = ref_start

    for op, length in cigar_tuples:
        if op in (0, 7, 8):  # M, =, X — consumes both query and reference
            for _ in range(length):
                if 0 <= ref_pos < ref_length and query_pos < len(query_seq):
                    result[ref_pos] = base_map.get(query_seq[query_pos].upper(), 0)
                ref_pos += 1
                query_pos += 1
        elif op == 1:  # I — consumes query only (insertion to reference)
            query_pos += length
        elif op == 2:  # D — consumes reference only (deletion from reference)
            ref_pos += length
        elif op == 4:  # S — soft clip, consumes query only
            query_pos += length
        elif op == 5:  # H — hard clip, consumes nothing
            pass
        elif op == 3:  # N — skip region on reference
            ref_pos += length

    return result
