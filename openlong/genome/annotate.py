"""Variant annotation module.

Provides basic functional annotation for detected variants,
including gene overlap, coding impact, and known variant databases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    """Functional annotation for a variant."""

    gene: str = ""
    feature_type: str = ""  # exon, intron, UTR, intergenic
    impact: str = ""  # HIGH, MODERATE, LOW, MODIFIER
    consequence: str = ""  # missense, synonymous, frameshift, etc.
    protein_change: str = ""
    known_id: str = ""  # dbSNP rsID or ClinVar ID


def annotate_variants_bed(
    variants: list[dict],
    gene_bed: str | Path,
) -> list[dict]:
    """Annotate variants with gene information from a BED file.

    This is a lightweight annotation approach that intersects variant
    positions with gene coordinates from a BED file.

    Args:
        variants: List of variant dicts.
        gene_bed: Path to gene annotation BED file.
            Expected format: chrom, start, end, gene_name, score, strand

    Returns:
        Variants with added 'annotation' field.
    """
    # Load gene intervals
    genes = []
    gene_bed = Path(gene_bed)
    if not gene_bed.exists():
        logger.warning(f"Gene BED file not found: {gene_bed}")
        return variants

    with open(gene_bed) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("track"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                genes.append(
                    {
                        "chrom": parts[0],
                        "start": int(parts[1]),
                        "end": int(parts[2]),
                        "name": parts[3],
                        "strand": parts[5] if len(parts) > 5 else "+",
                    }
                )

    logger.info(f"Loaded {len(genes)} gene regions from {gene_bed}")

    # Simple interval intersection
    for var in variants:
        v_chrom = var.get("chrom", "")
        v_pos = var.get("pos", 0)

        overlapping_genes = [
            g["name"]
            for g in genes
            if g["chrom"] == v_chrom and g["start"] <= v_pos <= g["end"]
        ]

        if overlapping_genes:
            var.setdefault("info", {})["GENE"] = ",".join(overlapping_genes)

    annotated_count = sum(1 for v in variants if "GENE" in v.get("info", {}))
    logger.info(f"Annotated {annotated_count}/{len(variants)} variants with gene info")
    return variants


def predict_coding_impact(
    ref_base: str,
    alt_base: str,
    codon_position: int,
    ref_codon: str,
) -> str:
    """Predict the coding impact of a single nucleotide change.

    Args:
        ref_base: Reference base.
        alt_base: Alternate base.
        codon_position: Position within the codon (0, 1, or 2).
        ref_codon: The reference codon (3 bases).

    Returns:
        Impact description string.
    """
    codon_table = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }

    ref_codon = ref_codon.upper()
    alt_codon = list(ref_codon)
    alt_codon[codon_position] = alt_base.upper()
    alt_codon = "".join(alt_codon)

    ref_aa = codon_table.get(ref_codon, "?")
    alt_aa = codon_table.get(alt_codon, "?")

    if ref_aa == alt_aa:
        return "synonymous"
    elif alt_aa == "*":
        return "nonsense"
    elif ref_aa == "*":
        return "stop_lost"
    else:
        return "missense"
