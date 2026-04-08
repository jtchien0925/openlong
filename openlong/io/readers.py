"""Readers for long-read sequencing data.

Supports:
- PacBio BAM (CLR and CCS/HiFi reads)
- Oxford Nanopore BAM/FASTQ
- FASTA reference genomes
- Standard FASTQ
"""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LongRead:
    """Represents a single long read from any platform."""

    name: str
    sequence: str
    quality: np.ndarray | None = None  # Phred scores
    platform: str = "unknown"  # pacbio_clr, pacbio_hifi, ont
    read_length: int = 0
    passes: int = 1  # Number of passes (CCS)
    mean_qv: float = 0.0

    # Alignment info (populated after alignment)
    reference_start: int = -1
    reference_end: int = -1
    cigar: str = ""
    is_reverse: bool = False
    mapping_quality: int = 0

    def __post_init__(self):
        self.read_length = len(self.sequence)
        if self.quality is not None:
            self.mean_qv = float(np.mean(self.quality))


@dataclass
class ReadCollection:
    """Collection of reads with metadata."""

    reads: list[LongRead] = field(default_factory=list)
    platform: str = "unknown"
    total_bases: int = 0
    n50: int = 0

    def add(self, read: LongRead):
        self.reads.append(read)
        self.total_bases += read.read_length

    def compute_stats(self):
        """Compute N50 and other collection statistics."""
        if not self.reads:
            return
        lengths = sorted([r.read_length for r in self.reads], reverse=True)
        cumsum = np.cumsum(lengths)
        half_total = self.total_bases / 2
        for length, cs in zip(lengths, cumsum):
            if cs >= half_total:
                self.n50 = length
                break

    def filter_by_length(self, min_length: int = 2000) -> ReadCollection:
        """Filter reads by minimum length."""
        filtered = ReadCollection(platform=self.platform)
        for read in self.reads:
            if read.read_length >= min_length:
                filtered.add(read)
        filtered.compute_stats()
        logger.info(
            f"Filtered {len(self.reads)} -> {len(filtered.reads)} reads "
            f"(min_length={min_length})"
        )
        return filtered


def detect_platform(filepath: str | Path) -> str:
    """Detect sequencing platform from BAM read group headers."""
    try:
        import pysam

        with pysam.AlignmentFile(str(filepath), check_sq=False) as bam:
            header = bam.header.to_dict()
            for rg in header.get("RG", []):
                pl = rg.get("PL", "").upper()
                ds = rg.get("DS", "").upper()
                if "PACBIO" in pl or "SEQUEL" in ds or "REVIO" in ds:
                    # Check if CCS/HiFi or CLR
                    if "CCS" in ds or "HIFI" in ds:
                        return "pacbio_hifi"
                    return "pacbio_clr"
                if "ONT" in pl or "NANOPORE" in pl or "MINION" in ds:
                    return "ont"
    except Exception:
        pass
    return "unknown"


def read_bam(
    filepath: str | Path,
    min_length: int = 2000,
    min_mapq: int = 0,
    region: str | None = None,
) -> ReadCollection:
    """Read long reads from a BAM file.

    Args:
        filepath: Path to BAM file.
        min_length: Minimum read length to include.
        min_mapq: Minimum mapping quality.
        region: Optional genomic region (e.g., 'chr1:1000-2000').

    Returns:
        ReadCollection with parsed reads.
    """
    import pysam

    filepath = Path(filepath)
    platform = detect_platform(filepath)
    collection = ReadCollection(platform=platform)

    logger.info(f"Reading BAM: {filepath} (platform={platform})")

    with pysam.AlignmentFile(str(filepath), "rb") as bam:
        iterator = bam.fetch(region=region) if region else bam.fetch(until_eof=True)

        for aln in iterator:
            if aln.is_unmapped and min_mapq > 0:
                continue
            if aln.query_length and aln.query_length < min_length:
                continue
            if aln.mapping_quality < min_mapq:
                continue

            quality = np.array(aln.query_qualities) if aln.query_qualities else None
            passes = 1
            if aln.has_tag("np"):
                passes = aln.get_tag("np")

            read = LongRead(
                name=aln.query_name,
                sequence=aln.query_sequence or "",
                quality=quality,
                platform=platform,
                passes=passes,
                reference_start=aln.reference_start if not aln.is_unmapped else -1,
                reference_end=aln.reference_end if not aln.is_unmapped else -1,
                cigar=aln.cigarstring or "",
                is_reverse=aln.is_reverse,
                mapping_quality=aln.mapping_quality,
            )
            collection.add(read)

    collection.compute_stats()
    logger.info(
        f"Loaded {len(collection.reads)} reads, "
        f"{collection.total_bases:,} bases, N50={collection.n50:,}"
    )
    return collection


def read_fastq(
    filepath: str | Path,
    min_length: int = 2000,
    platform: str = "unknown",
) -> ReadCollection:
    """Read long reads from a FASTQ file (plain or gzipped).

    Args:
        filepath: Path to FASTQ file.
        min_length: Minimum read length to include.
        platform: Sequencing platform hint.

    Returns:
        ReadCollection with parsed reads.
    """
    filepath = Path(filepath)
    collection = ReadCollection(platform=platform)

    opener = gzip.open if filepath.suffix == ".gz" else open
    logger.info(f"Reading FASTQ: {filepath}")

    with opener(str(filepath), "rt") as fh:
        while True:
            header = fh.readline().strip()
            if not header:
                break
            sequence = fh.readline().strip()
            fh.readline()  # + line
            quality_str = fh.readline().strip()

            if len(sequence) < min_length:
                continue

            quality = np.array([ord(c) - 33 for c in quality_str], dtype=np.int8)
            name = header[1:].split()[0]

            read = LongRead(
                name=name,
                sequence=sequence,
                quality=quality,
                platform=platform,
            )
            collection.add(read)

    collection.compute_stats()
    logger.info(
        f"Loaded {len(collection.reads)} reads, "
        f"{collection.total_bases:,} bases, N50={collection.n50:,}"
    )
    return collection


def read_fasta(filepath: str | Path) -> dict[str, str]:
    """Read a FASTA file into a dict of name -> sequence.

    Args:
        filepath: Path to FASTA file (plain or gzipped).

    Returns:
        Dictionary mapping sequence names to sequences.
    """
    filepath = Path(filepath)
    sequences: dict[str, str] = {}
    opener = gzip.open if filepath.suffix == ".gz" else open

    logger.info(f"Reading FASTA: {filepath}")

    current_name = ""
    current_seq: list[str] = []

    with opener(str(filepath), "rt") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    sequences[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name:
            sequences[current_name] = "".join(current_seq)

    logger.info(f"Loaded {len(sequences)} sequences from FASTA")
    return sequences


def auto_read(
    filepath: str | Path,
    min_length: int = 2000,
    **kwargs,
) -> ReadCollection:
    """Auto-detect file format and read accordingly."""
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix == ".gz":
        suffix = Path(filepath.stem).suffix.lower()

    if suffix == ".bam":
        return read_bam(filepath, min_length=min_length, **kwargs)
    elif suffix in (".fastq", ".fq"):
        return read_fastq(filepath, min_length=min_length, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
