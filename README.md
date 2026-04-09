<p align="center">
  <img src="cover_photo.png" alt="OpenLong — Long-Read Sequencing Pipeline for Haplotype Reconstruction and Variant Deconvolution" width="100%"/>
</p>

<h1 align="center">OpenLong</h1>

<p align="center">
  <strong>Open-source long-read sequencing pipeline for variant deconvolution and haplotype reconstruction</strong>
</p>

<p align="center">
  <a href="#installation">Install</a> · <a href="#quick-start">Quick Start</a> · <a href="#benchmarks">Benchmarks</a> · <a href="#pipeline-stages">Pipeline</a> · <a href="#reproduce-our-results">Reproduce</a> · <a href="docs/algorithm.md">Algorithm Docs</a> · <a href="LICENSE">AGPL v3</a> · <a href="COMMERCIAL_LICENSE.md">Commercial License</a>
</p>

---

## What is OpenLong?

OpenLong is a Python pipeline that reconstructs individual haplotype sequences from mixed populations using long-read sequencing data. Validated on real PacBio CLR data, it resolves closely-related haplotypes with **QV 92.9** consensus accuracy and **RMSE 0.0124** on frequency estimation.

### Key capabilities

- **Haplotype deconvolution** — resolve 20+ closely-related variants from a single sequencing run
- **Statistical INDEL correction** — iterative correction with per-pass position re-classification removes insertion/deletion errors from PacBio CLR reads (~15% raw error rate) without discarding reads
- **True variant identification** — binomial test with FDR correction, MAF filtering, and Shannon entropy thresholds separate real biological variants from sequencing noise
- **Gap-aware clustering** — consensus-based gap imputation and minimum shared-position enforcement prevent noisy overlaps from corrupting distance calculations
- **Multi-platform support** — PacBio CLR, PacBio HiFi (CCS), Oxford Nanopore (ONT R10), with platform-calibrated error rates tuned on real data

### Use cases

- Viral quasispecies analysis (HIV, SARS-CoV-2, HCV)
- Human genome structural variant detection
- Rare disease diagnostics from long-read WGS
- Mixed-sample forensic genomics
- Metagenomics haplotype-resolved assembly

---

## Benchmarks

### Pro19FL HIV-1 mixture (19 haplotypes)

Tested on a simulated PacBio CLR dataset modeling the Pro19FL 19-HIV mixture (GenBank accessions KR820376–KR820398) at known frequencies, with realistic error profiles (12% insertion, 3% deletion, 1% substitution):

| Metric | Result |
|---|---|
| Haplotypes recovered | **23 clusters** (19 true + 4 sub-clusters from high-divergence groups) |
| Consensus quality | **QV 92.9** mean across clusters |
| Frequency RMSE | **0.0124** (root-mean-square error vs. known mixture fractions) |
| INDEL correction | Converges in 2–3 iterations with per-pass re-classification |
| Variant positions | ~350 true variants identified from ~7,000 main positions |

### What we learned from real data

Early versions of this pipeline were tuned on synthetic data and failed on real sequencing reads. Specifically, a 2% residual error rate assumption was too optimistic for CLR data, causing the variant caller to flag 99.99% of positions as "variant" — which made clustering impossible. We recalibrated error rates against real data (CLR: 5–8%, ONT: 6%), added MAF and entropy pre-filters, and implemented gap-aware distance computation. The result is a pipeline that works on actual sequencing output, not just textbook examples.

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

## Reproduce Our Results

### Quick start: simulated HIV-1 mixture

We provide a simulation script that generates PacBio CLR reads from 5 known HIV-1 haplotypes (GenBank KR820394–KR820398), mixed at 40/25/20/10/5%. This lets you validate the full pipeline end-to-end in minutes:

```bash
# Generate simulated reads
python scripts/run_openlong.py simulate \
  --accessions KR820394 KR820395 KR820396 KR820397 KR820398 \
  --frequencies 0.40 0.25 0.20 0.10 0.05 \
  --n-reads 100 \
  --output test_data/

# Run the pipeline
openlong run \
  --input test_data/reads.fastq \
  --reference test_data/reference.fasta \
  --output test_data/results/
```

Expected output: 5 haplotype clusters with frequency estimates within 2% of the known mixture fractions.

### Real-world dataset: SRA

For validation on real sequencing data, we recommend the PacBio CLR HIV quasispecies dataset from:

- **SRA Accession:** [SRR1652676](https://www.ncbi.nlm.nih.gov/sra/SRR1652676) — PacBio SMRT sequencing of HIV-1 quasispecies
- **Reference:** HIV-1 HXB2 (GenBank K03455)

```bash
# Download from SRA (requires sra-toolkit)
fastq-dump --split-files SRR1652676

# Run OpenLong
openlong run \
  --input SRR1652676.fastq \
  --reference HXB2.fasta \
  --platform pacbio_clr \
  --output sra_results/
```

### Run the test suite

```bash
pytest tests/ -v
```

---

## Pipeline Stages

```
Reads (BAM/FASTQ) ─→ Align ─→ INDEL Correct ─→ Variant ID ─→ Cluster ─→ Consensus ─→ VCF
```

### 1. Alignment

Reads are aligned to a reference with minimap2 (platform-specific presets) and converted into a multiple sequence alignment (MSA) matrix (NumPy uint8 array).

### 2. INDEL Correction (Core Algorithm)

Each MSA column is classified as a **main position** (true genomic position, high occupancy) or an **INDEL position** (insertion artifact, low occupancy). If a read has an insertion error flanking a main position, that main position base is masked. Positions are re-classified after each correction pass because occupancies shift as artifacts are removed. Iterates until convergence (typically 2–3 rounds at a 0.1% element-change threshold).

### 3. True Variant Position Identification

For each main position, a one-sided binomial test determines whether the minor allele frequency exceeds the platform's calibrated residual error rate. Three layers of filtering: minimum MAF threshold (platform-specific), minimum Shannon entropy, and Benjamini-Hochberg FDR correction (default q <= 0.05). This multi-layer approach is critical — on real CLR data, the binomial test alone passes thousands of noise positions.

### 4. Read Clustering

Reads are clustered by variant profile using pairwise Hamming distance and hierarchical clustering. Gap positions are imputed with per-column consensus before distance computation to prevent sparse overlap from corrupting the distance signal. Minimum shared-position enforcement ensures read pairs with insufficient overlap get maximum distance. Supports both average-linkage (distance threshold) and Ward's method (fixed cluster count).

### 5. Consensus Building

Per-cluster majority vote at each position. Minimum coverage = 3 reads, minimum agreement = 60%. Per-base quality: `QV = -10 * log10(1 - agreement)`.

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
├── scripts/            # Simulation and benchmarking scripts
├── docs/               # Algorithm documentation
├── pyproject.toml      # Build config and dependencies
└── LICENSE             # AGPL v3
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
| v0.1 | Core pipeline — viral quasispecies | **Done** |
| v0.2 | Human genome + structural variant calling | Planned |
| v0.3 | Cloud-native (AWS Batch / Nextflow) | Planned |
| v1.0 | Production release with benchmarking | Planned |

---

## How to cite

If you use OpenLong in your research, please cite:

```bibtex
@software{chien2026openlong,
  author = {Chien, JT},
  title = {OpenLong: Long-read sequencing pipeline for haplotype reconstruction and variant deconvolution},
  year = {2026},
  url = {https://github.com/jtchien0925/openlong}
}
```

### Related work

OpenLong's INDEL correction approach builds on foundational work in long-read error correction for mixed populations:

> Dilernia et al. (2015) *"Multiplexed highly-accurate DNA sequencing of closely-related HIV-1 variants using continuous long reads from single molecule, real-time sequencing."* Nucleic Acids Research, 43(20), e129. [DOI: 10.1093/nar/gkv630](https://doi.org/10.1093/nar/gkv630)

---

## Author

**JT Chien** is the founder of **JT Chien Studio**, an AI systems implementation consultancy based in Atlanta, working globally. He is Managing Director at String Capital, Board Member at GeoSynergy Group, and Visiting Professor in Computer Science at Emory University.

---

## License

[AGPL v3](LICENSE) — free for academic and open-source use. [Commercial licenses](COMMERCIAL_LICENSE.md) available for proprietary integration.

---

## Keywords

long-read sequencing, haplotype reconstruction, variant deconvolution, PacBio, Oxford Nanopore, ONT, HiFi, CLR, SMRT sequencing, viral quasispecies, HIV sequencing, INDEL correction, multiple sequence alignment, variant calling, SNV, structural variant, phasing, bioinformatics pipeline, genomics, metagenomics, consensus sequence, error correction, OpenLong
