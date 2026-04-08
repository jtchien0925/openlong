"""Structural variant detection from long reads.

Leverages the long read lengths to detect larger genomic alterations
including insertions, deletions, inversions, duplications, and
translocations that are typically missed by short-read sequencing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

MIN_SV_SIZE = 50  # Minimum size for a structural variant


@dataclass
class StructuralVariant:
    """A detected structural variant."""

    sv_type: str  # DEL, INS, INV, DUP, BND
    chrom: str
    start: int
    end: int
    size: int
    support: int  # Number of supporting reads
    quality: float
    genotype: str = "0/1"
    info: dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}

    def to_vcf_dict(self) -> dict:
        """Convert to VCF-compatible dictionary."""
        return {
            "chrom": self.chrom,
            "pos": self.start,
            "id": f"sv_{self.sv_type}_{self.chrom}_{self.start}",
            "ref": "N",
            "alt": f"<{self.sv_type}>",
            "qual": f"{self.quality:.0f}",
            "filter": "PASS" if self.quality >= 20 else "LowQual",
            "info": {
                "SVTYPE": self.sv_type,
                "SVLEN": self.size if self.sv_type != "INS" else self.size,
                "END": self.end,
                "DP": self.support,
                **self.info,
            },
            "gt": self.genotype,
            "gq": int(self.quality),
        }


def detect_deletions(
    msa: np.ndarray,
    is_main: np.ndarray,
    reference_seq: str,
    chrom: str = "chr1",
    offset: int = 0,
    min_size: int = MIN_SV_SIZE,
    min_support_fraction: float = 0.2,
) -> list[StructuralVariant]:
    """Detect deletions from gap patterns in the MSA.

    A deletion is indicated by a contiguous stretch of gaps in multiple
    reads at positions that are classified as 'main' (i.e., normally
    occupied in the reference).

    Args:
        msa: Corrected MSA matrix.
        is_main: Position classification.
        reference_seq: Reference sequence.
        chrom: Chromosome name.
        offset: Coordinate offset.
        min_size: Minimum deletion size.
        min_support_fraction: Minimum fraction of reads supporting.

    Returns:
        List of StructuralVariant objects.
    """
    n_reads, n_positions = msa.shape
    deletions = []

    # For each read, find contiguous gap regions at main positions
    for read_idx in range(n_reads):
        row = msa[read_idx]
        in_gap = False
        gap_start = 0

        for pos in range(n_positions):
            if not is_main[pos]:
                continue

            if row[pos] == 0:  # Gap
                if not in_gap:
                    gap_start = pos
                    in_gap = True
            else:
                if in_gap:
                    gap_size = pos - gap_start
                    if gap_size >= min_size:
                        # Count supporting reads
                        support = 0
                        for other_idx in range(n_reads):
                            other_gaps = np.sum(msa[other_idx, gap_start:pos] == 0)
                            if other_gaps > gap_size * 0.8:
                                support += 1

                        if support / n_reads >= min_support_fraction:
                            qual = min(support * 10, 60)
                            sv = StructuralVariant(
                                sv_type="DEL",
                                chrom=chrom,
                                start=offset + gap_start,
                                end=offset + pos,
                                size=gap_size,
                                support=support,
                                quality=qual,
                            )
                            deletions.append(sv)
                    in_gap = False

    # Deduplicate overlapping deletions
    deletions = _merge_overlapping_svs(deletions)
    logger.info(f"Detected {len(deletions)} deletions (>= {min_size} bp)")
    return deletions


def detect_insertions(
    msa: np.ndarray,
    is_main: np.ndarray,
    chrom: str = "chr1",
    offset: int = 0,
    min_size: int = MIN_SV_SIZE,
    min_support_fraction: float = 0.2,
) -> list[StructuralVariant]:
    """Detect insertions from excess bases at INDEL positions.

    Insertions appear as positions classified as 'INDEL' where multiple
    reads consistently have non-gap bases.

    Args:
        msa: Corrected MSA matrix.
        is_main: Position classification.
        chrom: Chromosome name.
        offset: Coordinate offset.
        min_size: Minimum insertion size.
        min_support_fraction: Minimum fraction of reads supporting.

    Returns:
        List of StructuralVariant objects.
    """
    n_reads, n_positions = msa.shape
    insertions = []

    # Find stretches of INDEL positions with high occupancy
    in_insertion = False
    ins_start = 0
    ins_occupancy = []

    for pos in range(n_positions):
        if is_main[pos]:
            if in_insertion:
                ins_size = pos - ins_start
                if ins_size >= min_size:
                    mean_occ = np.mean(ins_occupancy)
                    support = int(mean_occ * n_reads)
                    if support / n_reads >= min_support_fraction:
                        qual = min(support * 10, 60)
                        sv = StructuralVariant(
                            sv_type="INS",
                            chrom=chrom,
                            start=offset + ins_start,
                            end=offset + ins_start,
                            size=ins_size,
                            support=support,
                            quality=qual,
                        )
                        insertions.append(sv)
                in_insertion = False
                ins_occupancy = []
        else:
            if not in_insertion:
                ins_start = pos
                in_insertion = True
            occ = np.sum(msa[:, pos] > 0) / n_reads
            ins_occupancy.append(occ)

    logger.info(f"Detected {len(insertions)} insertions (>= {min_size} bp)")
    return insertions


def _merge_overlapping_svs(
    svs: list[StructuralVariant],
    max_distance: int = 100,
) -> list[StructuralVariant]:
    """Merge overlapping or nearby structural variants."""
    if not svs:
        return []

    sorted_svs = sorted(svs, key=lambda sv: sv.start)
    merged = [sorted_svs[0]]

    for sv in sorted_svs[1:]:
        last = merged[-1]
        if sv.sv_type == last.sv_type and sv.start <= last.end + max_distance:
            # Merge: extend the last SV
            last.end = max(last.end, sv.end)
            last.size = last.end - last.start
            last.support = max(last.support, sv.support)
            last.quality = max(last.quality, sv.quality)
        else:
            merged.append(sv)

    return merged
