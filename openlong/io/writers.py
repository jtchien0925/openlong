"""Writers for pipeline output.

Supports:
- FASTA output for reconstructed haplotypes
- VCF output for variant calls
- Summary reports
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def write_fasta(
    sequences: dict[str, str],
    filepath: str | Path,
    line_width: int = 80,
) -> None:
    """Write sequences to a FASTA file.

    Args:
        sequences: Dictionary mapping names to sequences.
        filepath: Output path.
        line_width: Characters per line in output.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as fh:
        for name, seq in sequences.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), line_width):
                fh.write(seq[i : i + line_width] + "\n")

    logger.info(f"Wrote {len(sequences)} sequences to {filepath}")


def write_vcf(
    variants: list[dict],
    filepath: str | Path,
    reference_name: str = "ref",
    sample_name: str = "SAMPLE",
) -> None:
    """Write variants to a VCF file.

    Args:
        variants: List of variant dicts with keys:
            chrom, pos, ref, alt, qual, filter, info
        filepath: Output path.
        reference_name: Reference genome name.
        sample_name: Sample name.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as fh:
        # VCF header
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(f"##fileDate={datetime.now().strftime('%Y%m%d')}\n")
        fh.write(f'##source=OpenLong v0.1.0\n')
        fh.write(f'##reference={reference_name}\n')
        fh.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total read depth">\n')
        fh.write(
            '##INFO=<ID=AF,Number=A,Type=Float,'
            'Description="Allele frequency in the population">\n'
        )
        fh.write(
            '##INFO=<ID=HAP,Number=1,Type=String,'
            'Description="Haplotype assignment">\n'
        )
        fh.write(
            '##INFO=<ID=SVTYPE,Number=1,Type=String,'
            'Description="Type of structural variant">\n'
        )
        fh.write(
            '##INFO=<ID=SVLEN,Number=1,Type=Integer,'
            'Description="Length of structural variant">\n'
        )
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">\n')
        fh.write(
            f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_name}\n"
        )

        # Variant records
        for var in sorted(variants, key=lambda v: (v.get("chrom", ""), v.get("pos", 0))):
            chrom = var.get("chrom", ".")
            pos = var.get("pos", 0)
            vid = var.get("id", ".")
            ref = var.get("ref", ".")
            alt = var.get("alt", ".")
            qual = var.get("qual", ".")
            filt = var.get("filter", "PASS")
            info_dict = var.get("info", {})
            info_str = ";".join(f"{k}={v}" for k, v in info_dict.items()) or "."
            fmt = "GT:GQ"
            gt = var.get("gt", "0/1")
            gq = var.get("gq", 30)
            sample = f"{gt}:{gq}"

            fh.write(
                f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t{qual}\t{filt}\t"
                f"{info_str}\t{fmt}\t{sample}\n"
            )

    logger.info(f"Wrote {len(variants)} variants to {filepath}")


def write_report(
    stats: dict,
    filepath: str | Path,
) -> None:
    """Write a JSON summary report.

    Args:
        stats: Dictionary of pipeline statistics.
        filepath: Output path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    stats["timestamp"] = datetime.now().isoformat()
    stats["version"] = "0.1.0"

    with open(filepath, "w") as fh:
        json.dump(stats, fh, indent=2, default=str)

    logger.info(f"Wrote report to {filepath}")
