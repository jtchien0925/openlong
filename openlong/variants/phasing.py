"""Haplotype phasing module.

Assigns variants to specific haplotypes using the long-range linkage
information inherent in long reads. This is one of the key advantages
of long-read sequencing over short-read approaches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from openlong.deconv.cluster import HaplotypeCluster

logger = logging.getLogger(__name__)


@dataclass
class PhasedBlock:
    """A block of phased variants."""

    block_id: int
    chrom: str
    start: int
    end: int
    haplotype_1: list[dict] = field(default_factory=list)
    haplotype_2: list[dict] = field(default_factory=list)
    n_variants: int = 0
    phase_quality: float = 0.0


def phase_variants(
    variants: list[dict],
    clusters: list[HaplotypeCluster],
    msa: np.ndarray,
    is_main: np.ndarray,
    chrom: str = "chr1",
    ref_to_msa: np.ndarray | None = None,
) -> list[PhasedBlock]:
    """Phase variants into haplotype blocks using cluster assignments.

    Since reads are already clustered into haplotypes, phasing is
    straightforward: each variant is assigned to the haplotype(s)
    whose reads support it.

    Args:
        variants: List of variant dicts from SNV/SV calling.
        clusters: Haplotype clusters with read assignments.
        msa: Full corrected MSA matrix.
        is_main: Position classification.
        chrom: Chromosome name.
        ref_to_msa: Coordinate mapping array from reference positions to MSA columns.
                    If None, falls back to approximate mapping (pos - 1).

    Returns:
        List of PhasedBlock objects.
    """
    if not variants or not clusters:
        return []

    # Build read-to-cluster mapping
    read_to_cluster = {}
    for cluster in clusters:
        for read_idx in cluster.read_indices:
            read_to_cluster[read_idx] = cluster.cluster_id

    # For each variant, determine which clusters support it
    phased_variants = []
    for var in variants:
        pos = var.get("pos", 0)
        alt_base = var.get("alt", "")

        # Map VCF position to MSA column index using coordinate mapping if available
        if ref_to_msa is not None:
            # Use proper coordinate mapping with bounds checking
            if 0 <= pos < len(ref_to_msa):
                msa_col = ref_to_msa[pos]
            else:
                continue
        else:
            # Fallback: approximate 1:1 mapping (backwards compatibility)
            msa_col = pos - 1

        if msa_col < 0 or msa_col >= msa.shape[1]:
            continue

        # Check which clusters have the alt allele at this position
        base_map = {"A": 1, "C": 2, "G": 3, "T": 4}
        alt_encoded = base_map.get(alt_base.upper(), 0)

        supporting_clusters = set()
        for read_idx in range(msa.shape[0]):
            if msa[read_idx, msa_col] == alt_encoded:
                cluster_id = read_to_cluster.get(read_idx)
                if cluster_id is not None:
                    supporting_clusters.add(cluster_id)

        var_copy = dict(var)
        var_copy["phase_clusters"] = list(supporting_clusters)
        phased_variants.append(var_copy)

    # Group variants into phased blocks (contiguous regions with consistent phasing)
    if not phased_variants:
        return []

    blocks = []
    current_block_variants = [phased_variants[0]]

    for i in range(1, len(phased_variants)):
        prev = phased_variants[i - 1]
        curr = phased_variants[i]

        # Start new block if gap > 10kb or phasing pattern changes drastically
        gap = curr["pos"] - prev["pos"]
        if gap > 10000:
            blocks.append(current_block_variants)
            current_block_variants = [curr]
        else:
            current_block_variants.append(curr)

    blocks.append(current_block_variants)

    # Convert to PhasedBlock objects
    phased_blocks = []
    for block_id, block_vars in enumerate(blocks):
        if not block_vars:
            continue

        pb = PhasedBlock(
            block_id=block_id,
            chrom=chrom,
            start=block_vars[0]["pos"],
            end=block_vars[-1]["pos"],
            n_variants=len(block_vars),
        )

        # Assign variants to haplotypes based on cluster support
        for var in block_vars:
            phase_clusters = var.get("phase_clusters", [])
            if len(clusters) >= 2:
                if clusters[0].cluster_id in phase_clusters:
                    pb.haplotype_1.append(var)
                if len(clusters) > 1 and clusters[1].cluster_id in phase_clusters:
                    pb.haplotype_2.append(var)
            else:
                pb.haplotype_1.append(var)

        phased_blocks.append(pb)

    logger.info(
        f"Phased {len(variants)} variants into {len(phased_blocks)} blocks"
    )
    return phased_blocks
