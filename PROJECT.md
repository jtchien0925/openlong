# OpenLong - Open Source Long-Read Sequencing Pipeline

## What Is This

OpenLong is an open-source Python pipeline for deconvoluting closely-related genomic
variants from long-read sequencing data (PacBio CLR/HiFi and Oxford Nanopore).

It implements and extends the algorithmic approach described in:
> Dilernia et al. (2015) "Multiplexed highly-accurate DNA sequencing of closely-related
> HIV-1 variants using continuous long reads from single molecule, real-time sequencing"
> Nucleic Acids Research, 43(20), e129.

The pipeline reconstructs individual haplotype sequences from mixed populations with
>QV50 accuracy, supporting applications from viral quasispecies analysis to human
genome structural variant detection and rare disease diagnostics.

## Architecture

```
openlong/
├── openlong/               # Core library
│   ├── __init__.py
│   ├── io/                 # Input/output handlers
│   │   ├── __init__.py
│   │   ├── readers.py      # BAM/FASTQ/FASTA readers (PacBio + ONT)
│   │   └── writers.py      # Output writers (VCF, FASTA, reports)
│   ├── align/              # Alignment module
│   │   ├── __init__.py
│   │   └── aligner.py      # Reference alignment + self-alignment
│   ├── correct/            # Error correction module
│   │   ├── __init__.py
│   │   ├── indel.py        # INDEL correction algorithm (core paper algo)
│   │   └── polish.py       # Consensus polishing
│   ├── deconv/             # Variant deconvolution module
│   │   ├── __init__.py
│   │   ├── positions.py    # True variant position identification
│   │   ├── cluster.py      # Read clustering / haplotype assignment
│   │   └── consensus.py    # Per-cluster consensus building
│   ├── variants/           # Variant calling module
│   │   ├── __init__.py
│   │   ├── snv.py          # SNV calling
│   │   ├── sv.py           # Structural variant detection
│   │   └── phasing.py      # Haplotype phasing
│   ├── genome/             # Human genome application module
│   │   ├── __init__.py
│   │   ├── assembly.py     # Genome assembly support
│   │   └── annotate.py     # Variant annotation
│   └── pipeline.py         # Main pipeline orchestrator
├── scripts/
│   └── run_openlong.py     # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_indel.py
│   ├── test_positions.py
│   ├── test_cluster.py
│   └── test_pipeline.py
├── docs/
│   └── algorithm.md        # Detailed algorithm documentation
├── PROJECT.md              # This file
├── README.md               # User-facing README
├── setup.py
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Key Decisions

1. **Python-first**: Entire codebase in Python for maintainability. NumPy/SciPy for
   hot paths. Can add Cython/Rust extensions later for performance-critical sections.
2. **Platform-agnostic**: Supports PacBio CLR, PacBio HiFi (CCS), and ONT reads via
   pysam for BAM and standard FASTQ parsing.
3. **Modular design**: Each stage (align → correct → deconvolute → call) is independent
   and can be run standalone or as part of the full pipeline.
4. **Statistical INDEL correction**: Implements the alignment correction algorithm from
   Dilernia et al. 2015 — the core innovation that enables CLR-based haplotype reconstruction.

## Tech Stack

- Python 3.10+
- pysam (BAM/CRAM I/O, wraps htslib)
- minimap2 (alignment, called via subprocess)
- NumPy / SciPy (numerical computation, statistics)
- scikit-learn (clustering)
- Click (CLI)

## How to Run

```bash
# Install
pip install -e .

# Full pipeline
openlong run --input reads.bam --reference ref.fasta --output results/

# Individual stages
openlong align --input reads.fastq --reference ref.fasta --output aligned.bam
openlong correct --input aligned.bam --output corrected.bam
openlong deconv --input corrected.bam --output haplotypes/
openlong call --input haplotypes/ --reference ref.fasta --output variants.vcf
```

## Branch Strategy

- `main` — stable releases
- `feat/*` — new features
- `fix/*` — bug fixes
- `experiment/*` — experimental approaches

## Env Variables

- `OPENLONG_THREADS` — number of threads (default: 4)
- `OPENLONG_TMPDIR` — temp directory for intermediate files
- `OPENLONG_MINIMAP2` — path to minimap2 binary

## Current Status

- [x] Project structure and architecture
- [x] Core algorithm implementation (INDEL correction, variant position ID, clustering)
- [x] I/O handlers for PacBio and ONT
- [x] CLI entry point
- [ ] Full test suite with real data
- [ ] Human genome assembly module
- [ ] Cloud deployment scripts (AWS)
- [ ] Benchmarking against existing tools

## Known Issues

- First release — needs validation against published datasets
- Human genome module is scaffolded but not production-ready
- ONT error profile differs from PacBio CLR; correction params may need tuning

## Roadmap

1. v0.1 — Core pipeline (viral quasispecies focus)
2. v0.2 — Human genome support + structural variant calling
3. v0.3 — Cloud-native mode (AWS Batch / Nextflow integration)
4. v1.0 — Production release with full benchmarking

## Important Links

- Paper: https://doi.org/10.1093/nar/gkv630
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC4787755/
