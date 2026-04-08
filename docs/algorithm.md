# OpenLong Algorithm Documentation

## Overview

OpenLong implements and extends the algorithmic approach described in:

> Dilernia et al. (2015) "Multiplexed highly-accurate DNA sequencing of
> closely-related HIV-1 variants using continuous long reads from single
> molecule, real-time sequencing" Nucleic Acids Research, 43(20), e129.
> DOI: 10.1093/nar/gkv630

The core innovation enables accurate reconstruction of individual haplotype
sequences from mixed populations using error-prone long reads (PacBio CLR,
ONT). The approach achieves >QV50 accuracy (< 1 error per 100,000 bases).

## Pipeline Stages

### Stage 1: Alignment

Reads are aligned to a reference sequence using minimap2 with platform-specific
presets. The alignment produces a pairwise mapping of each read against the
reference, which is then converted into a Multiple Sequence Alignment (MSA)
matrix.

The MSA matrix is encoded as a numpy uint8 array where:
- 0 = gap
- 1 = A, 2 = C, 3 = G, 4 = T
- 5 = N (ambiguous)

Each row is a read, each column is a reference position.

### Stage 2: INDEL Correction (Core Algorithm)

This is the key innovation from the paper. PacBio CLR reads have ~15% raw
error rate, dominated by insertions and deletions (INDELs). These INDEL
errors corrupt the MSA and create false variant signals.

#### Position Classification

Each MSA column is classified as either:
- **Main position (P)**: A true genomic position where most reads have a
  non-gap base. Occupancy >= threshold (default 50%).
- **INDEL position (p)**: An artifact position created by insertion errors.
  Low occupancy across reads.

#### Correction Rule

For each read, at each main position P_x with flanking main positions P_y
(downstream) and P_z (upstream):

```
If any INDEL position between P_y and P_x, or between P_x and P_z,
contains a nucleotide (not a gap) in this read:
    → Replace the base at P_x with a gap (non-informative)
```

The intuition: if a read has an insertion error between two true positions,
the alignment of the adjacent true positions is likely shifted/corrupted.
By masking those corrupted main-position bases, we remove false signal
without losing true signal from uncorrupted reads.

#### Iterative Refinement

The correction is applied iteratively:
1. Classify positions → correct → update MSA
2. Re-classify positions (occupancies may have changed) → correct again
3. Repeat until convergence (< 0.01% of elements change)

Typically converges in 2-3 iterations.

### Stage 3: True Variant Position Identification

After INDEL correction, remaining variation at main positions could be:
- True biological variants (different haplotypes)
- Residual sequencing errors

We distinguish these using a statistical test:

1. For each main position, compute the minor allele frequency (MAF)
2. Apply a one-sided binomial test:
   - H0: observed minor allele count arose from errors at rate ε
   - H1: observed count is higher than expected from errors
   - ε is platform-specific: ~2% for CLR, ~0.1% for HiFi, ~3% for ONT
3. Apply Benjamini-Hochberg FDR correction across all tested positions
4. Positions with q-value <= threshold (default 0.05) are true variants

### Stage 4: Read Clustering (Haplotype Deconvolution)

Reads are clustered into haplotype groups based on their variant profiles:

1. Extract the variant-only submatrix (reads × true variant positions)
2. Compute pairwise Hamming distance between reads (ignoring gap positions)
3. Apply hierarchical clustering (average linkage)
4. Cut the dendrogram at a distance threshold to define clusters
5. **Iterative sub-clustering**: For each cluster, re-examine for internal
   structure by checking for new high-entropy positions within the cluster.
   If found, split and recurse.

This iterative approach was key to the paper's success in resolving up to
40 closely-related HIV-1 variants from a single SMRT Cell.

### Stage 5: Consensus Building

For each cluster, build a consensus sequence:

1. Extract the rows (reads) belonging to this cluster from the corrected MSA
2. At each main position, take the majority vote
3. Require minimum coverage (default 3 reads) and minimum agreement
   (default 60%) for a confident call
4. Compute per-base quality: QV = -10 × log10(1 - agreement_fraction)

The resulting consensus sequences represent the reconstructed haplotypes.

### Stage 6: Variant Calling

Compare reconstructed haplotypes against the reference:

- **SNVs**: Direct base-by-base comparison
- **Structural Variants**: Detected from gap patterns in the MSA
  - Deletions: contiguous gaps at main positions in multiple reads
  - Insertions: occupied INDEL positions in multiple reads
- **Phasing**: Variants are assigned to haplotypes based on which
  cluster's reads support each allele

### Extension: Human Genome Mode

For genome-scale applications (like Sequegenics' commercial platform):

1. Split genome into 1 Mb overlapping chunks
2. Process each chunk through the full pipeline independently
3. Merge overlapping results at chunk boundaries
4. Aggregate variants into genome-wide VCF

This enables the pipeline to scale from viral genomes (~10 kb)
to human genomes (~3 Gb) without architectural changes.

## Platform-Specific Considerations

| Parameter | PacBio CLR | PacBio HiFi | ONT R10 |
|-----------|-----------|-------------|---------|
| Raw error rate | ~15% | ~0.1% | ~5% |
| Error type | INDEL-dominant | Balanced | Mixed |
| Read length | 10-50 kb | 10-25 kb | 1-100+ kb |
| Residual error (post-correction) | ~2% | ~0.1% | ~3% |
| minimap2 preset | map-pb | map-hifi | map-ont |

For HiFi reads, the INDEL correction step is less critical (already high
accuracy) but the pipeline still benefits from the statistical variant
identification and clustering stages.

## References

1. Dilernia DA et al. (2015) Nucleic Acids Research 43(20):e129
2. Benjamini Y, Hochberg Y (1995) J Roy Stat Soc Ser B 57:289-300
3. Li H (2018) Bioinformatics 34(18):3094-3100 (minimap2)
