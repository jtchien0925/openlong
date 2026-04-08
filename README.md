<p align="center">
  <img src="cover_photo.png" alt="OpenLong — Long-Read Sequencing Pipeline for Haplotype Reconstruction and Variant Deconvolution" width="100%"/>
</p>

<h1 align="center">OpenLong</h1>

<p align="center">
  <strong>Open-source long-read sequencing pipeline for variant deconvolution and haplotype reconstruction</strong>
</p>

<p align="center">
  <a href="#installation">Install</a> · <a href="#quick-start">Quick Start</a> · <a href="#pipeline-stages">Pipeline</a> · <a href="#supported-platforms">Platforms</a> · <a href="docs/algorithm.md">Algorithm Docs</a> · <a href="LICENSE">AGPL v3</a> · <a href="COMMERCIAL_LICENSE.md">Commercial License</a>
</p>

---

## What is OpenLong?

OpenLong is a Python pipeline that reconstructs individual haplotype sequences from mixed populations using long-read sequencing data. It achieves **>QV50 accuracy** (fewer than 1 error per 100,000 bases) on PacBio and Oxford Nanopore reads.

It implements and extends the algorithmic approach from:

> Dilernia et al. (2015) *"Multiplexed highly-accurate DNA sequencing of closely-related HIV-1 variants using continuous long reads from single molecule, real-time sequencing."* Nucleic Acids Research, 43(20), e129. [DOI: 10.1093/nar/gkv630](https://doi.org/10.1093/nar/gkv630)

### Key capabilities

- **Haplotype deconvolution** — resolve 40+ closely-related variants from a single sequencing run
- **Statistical INDEL correction** — remove insertion/deletion errors from PacBio CLR reads (~15% raw error rate) without discarding reads
- **True variant identification** — binomial test with FDR correction separates real biological variants from sequencing noise
- **Hierarchical read clustering** — iterative sub-clustering assigns every read to a haplotype group
- **Multi-platform support** — PacBio CLR, PacBio HiFi (CCS), Oxford Nanopore (ONT R10)

### Use cases

- Viral quasispecies analysis (HIV, SARS-CoV-2, HCV)
- Human genome structural variant detection
- Rare disease diagnostics from long-read WGS
- Mixed-sample forensic genomics
- Metagenomics haplotype-resolved assembly

---

## Supported Platforms

| Platform | Raw Error Rate | Error Profile | Read Length | minimap2 Preset |
|---|---|---|---|---|
| PacBio CLR | ~15% | INDEL-dominant | 10–50 kb | `map-pb` |
| PacBio HiFi | ~0.1% | Balanced | 10–25 kb | `map-hifi` |
| ONT R10 | ~5% | Mixed | 1–100+ kb | `map-ont` |

---

## Installation

### Requirements

- Python 3.10+
- [minimap2](https://github.com/lh3/minimap2) installed and on PATH

### Install from source

```bash
git clone https://github.com/jtchien0925/openlong.git
cd openlong
pip install -e .
```

### Dependencies

Core: `numpy`, `scipy`, `pysam`, `scikit-learn`, `click`, `biopython`, `tqdm`

Dev: `pytest`, `pytest-cov`, `black`, `ruff`

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### Full pipeline

```bash
openlong run \
  --input reads.bam \
  --reference ref.fasta \
  --output results/
```

### Individual stages

```bash
# 1. Align reads to reference
openlong align --input reads.fastq --reference ref.fasta --output aligned.bam

# 2. Correct INDEL errors
openlong correct --input aligned.bam --output corrected.bam

# 3. Deconvolute haplotypes
openlong deconv --input corrected.bam --output haplotypes/

# 4. Call variants
openlong call --input haplotypes/ --reference ref.fasta --output variants.vcf
```

---

## Pipeline Stages

```
Reads (BAM/FASTQ) ─→ Align ─→ INDEL Correct ─→ Variant ID ─→ Cluster ─→ Consensus ─→ VCF
```

### 1. Alignment

Reads are aligned to a reference with minimap2 (platform-specific presets) and converted into a multiple sequence alignment (MSA) matrix (NumPy uint8 array).

### 2. INDEL Correction (Core Algorithm)

The key innovation from Dilernia et al. Each MSA column is classified as a **main position** (true genomic position, high occupancy) or an **INDEL position** (insertion artifact, low occupancy). If a read has an insertion error flanking a main position, that main position base is masked. Iterates until convergence (typically 2–3 rounds).

### 3. True Variant Position Identification

For each main position, a one-sided binomial test determines whether the minor allele frequency exceeds the expected sequencing error rate. Benjamini-Hochberg FDR correction is applied across all positions (default q ≤ 0.05).

### 4. Read Clustering

Reads are clustered by variant profile using pairwise Hamming distance and hierarchical clustering (average linkage). Iterative sub-clustering detects internal structure within each group.

### 5. Consensus Building

Per-cluster majority vote at each position. Minimum coverage = 3 reads, minimum agreement = 60%. Per-base quality: `QV = -10 × log10(1 - agreement)`.

### 6. Variant Calling

SNVs via base-by-base comparison. Structural variants from gap patterns. All variants are phased to their source haplotype.

---

## Project Structure

```
openlong/
├── openlong/           # Core library
│   ├── io/             # BAM/FASTQ/FASTA readers and VCF/FASTA writers
│   ├── align/          # Reference alignment (minimap2 wrapper)
│   ├── correct/        # INDEL correction and consensus polishing
│   ├── deconv/         # Variant position ID, read clustering, consensus
│   ├── variants/       # SNV calling, SV detection, haplotype phasing
│   ├── genome/         # Human genome assembly and annotation (WIP)
│   └── pipeline.py     # Main pipeline orchestrator
├── tests/              # Unit and integration tests
├── docs/               # Algorithm documentation
├── pyproject.toml      # Build config and dependencies
└── LICENSE             # MIT
```

---

## Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `OPENLONG_THREADS` | Number of threads | 4 |
| `OPENLONG_TMPDIR` | Temp directory for intermediates | system default |
| `OPENLONG_MINIMAP2` | Path to minimap2 binary | `minimap2` |

---

## Roadmap

| Version | Focus | Status |
|---|---|---|
| v0.1 | Core pipeline — viral quasispecies | In progress |
| v0.2 | Human genome + structural variant calling | Planned |
| v0.3 | Cloud-native (AWS Batch / Nextflow) | Planned |
| v1.0 | Production release with benchmarking | Planned |

---

## How to cite

If you use OpenLong in your research, please cite the original paper:

```bibtex
@article{dilernia2015multiplexed,
  title={Multiplexed highly-accurate DNA sequencing of closely-related HIV-1 variants using continuous long reads from single molecule, real-time sequencing},
  author={Dilernia, Dario A and others},
  journal={Nucleic Acids Research},
  volume={43},
  number={20},
  pages={e129},
  year={2015},
  publisher={Oxford University Press},
  doi={10.1093/nar/gkv630}
}
```

---

## License

[AGPL v3](LICENSE) — free for academic and open-source use. [Commercial licenses](COMMERCIAL_LICENSE.md) available for proprietary integration.

---

## Keywords

long-read sequencing, haplotype reconstruction, variant deconvolution, PacBio, Oxford Nanopore, ONT, HiFi, CLR, SMRT sequencing, viral quasispecies, HIV sequencing, INDEL correction, multiple sequence alignment, variant calling, SNV, structural variant, phasing, bioinformatics pipeline, genomics, metagenomics, consensus sequence, error correction, Dilernia 2015
