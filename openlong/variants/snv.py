"""SNV calling from deconvoluted haplotypes.

Identifies single nucleotide variants by comparing reconstructed
haplotype consensus sequences against a reference genome.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DECODE = {1: "A", 2: "C", 3: "G", 4: "T", 5: "N"}
ENCODE = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}


def call_snvs(
    haplotype_seq: str,
    reference_seq: str,
    haplotype_name: str = "HAP1",
    chrom: str = "chr1",
    offset: int = 0,
    min_qual: float = 20.0,
    haplotype_quality: np.ndarray | None = None,
) -> list[dict]:
    """Call SNVs between a haplotype and reference.

    Args:
        haplotype_seq: Haplotype consensus sequence.
        reference_seq: Reference sequence (same length region).
        haplotype_name: Name for INFO field.
        chrom: Chromosome name for VCF.
        offset: Genomic coordinate offset.
        min_qual: Minimum quality score for a call.
        haplotype_quality: Per-base quality array.

    Returns:
        List of variant dictionaries for VCF output.
    """
    variants = []
    min_len = min(len(haplotype_seq), len(reference_seq))

    for i in range(min_len):
        hap_base = haplotype_seq[i].upper()
        ref_base = reference_seq[i].upper()

        if hap_base == "N" or ref_base == "N":
            continue
        if hap_base == ref_base:
            continue

        qual = 30.0
        if haplotype_quality is not None and i < len(haplotype_quality):
            qual = float(haplotype_quality[i])

        if qual < min_qual:
            continue

        pos = offset + i + 1  # VCF is 1-based

        variants.append(
            {
                "chrom": chrom,
                "pos": pos,
                "id": f"snv_{chrom}_{pos}",
                "ref": ref_base,
                "alt": hap_base,
                "qual": f"{qual:.0f}",
                "filter": "PASS",
                "info": {
                    "HAP": haplotype_name,
                    "DP": 1,
                },
                "gt": "0/1",
                "gq": int(qual),
            }
        )

    logger.info(
        f"Called {len(variants)} SNVs for {haplotype_name} "
        f"({min_len} bp compared)"
    )
    return variants


def call_snvs_multi_haplotype(
    haplotypes: dict[str, str],
    reference_seq: str,
    chrom: str = "chr1",
    offset: int = 0,
    qualities: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """Call SNVs across all haplotypes vs reference.

    Args:
        haplotypes: Dict of name -> consensus sequence.
        reference_seq: Reference sequence.
        chrom: Chromosome name.
        offset: Genomic coordinate offset.
        qualities: Dict of name -> quality arrays.

    Returns:
        Combined list of variant dictionaries, deduplicated.
    """
    all_variants = []

    for name, seq in haplotypes.items():
        qual = qualities.get(name) if qualities else None
        variants = call_snvs(
            seq, reference_seq, haplotype_name=name,
            chrom=chrom, offset=offset, haplotype_quality=qual,
        )
        all_variants.extend(variants)

    # Deduplicate by position + alt allele, keeping highest quality
    seen = {}
    for var in all_variants:
        key = (var["chrom"], var["pos"], var["alt"])
        if key not in seen or int(var["qual"]) > int(seen[key]["qual"]):
            seen[key] = var

    deduped = sorted(seen.values(), key=lambda v: v["pos"])
    logger.info(f"Total unique SNVs across haplotypes: {len(deduped)}")
    return deduped
