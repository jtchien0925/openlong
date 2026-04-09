# OpenLong Algorithm Documentation

## Overview

OpenLong reconstructs individual haplotype sequences from mixed populations
using error-prone long reads (PacBio CLR, HiFi, ONT). The pipeline combines
occupancy-based INDEL correction, multi-layer variant filtering, and
gap-aware hierarchical clustering to achieve QV 92.9 consensus accuracy
on real PacBio CLR data.

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

PacBio CLR reads have ~15% raw error rate, dominated by insertions and
deletions (INDELs). These INDEL errors corrupt the MSA and create false
variant signals.

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

The correction is applied iteratively with per-pass position re-classification:
1. Classify positions → correct → update MSA
2. Re-classify positions (occupancies shift after corrections) → correct again
3. Repeat until convergence (< 0.1% of elements change)

This re-classification step is critical: after corrections, borderline
positions may shift category, exposing artifacts missed in the first pass.
Typically converges in 2-3 iterations.

### Stage 3: True Variant Position Identification

After INDEL correction, remaining variation at main positions could be:
- True biological variants (different haplotypes)
- Residual sequencing errors

We distinguish these using a multi-layer filter:

1. **MAF pre-filter**: minimum minor allele frequency (platform-calibrated:
   CLR=10%, HiFi=2%, ONT=10%). Eliminates low-frequency noise positions
   before statistical testing.
2. **Entropy pre-filter**: minimum Shannon entropy threshold (CLR=0.40,
   HiFi=0.10, ONT=0.40). True variant positions have higher entropy
   than noise.
3. **Binomial test**: one-sided test for whether observed minor allele
   count exceeds platform's residual error rate (CLR=5%, HiFi=0.5%,
   ONT=6%). These rates are calibrated on real data — early versions
   used 2% for CLR which was too optimistic.
4. **FDR correction**: Benjamini-Hochberg across all tested positions
   (default q <= 0.05).
5. **Optional strand bias filter**: Fisher's exact test to remove
   variants appearing predominantly on one strand.

The multi-layer approach is essential. On real CLR data, the binomial test
alone passes thousands of noise positions. MAF and entropy pre-filters
reduce the candidate set to true biological variants.

### Stage 4: Read Clustering (Haplotype Deconvolution)

Reads are clustered into haplotype groups based on their variant profiles:

1. Extract the variant-only submatrix (reads x true variant positions)
2. **Gap imputation**: if gap fraction > 5%, fill gaps with per-column
   consensus to prevent sparse overlap from corrupting distances
3. Compute pairwise Hamming distance with minimum shared-position
   enforcement (pairs with < 10 shared positions get maximum distance)
4. Apply hierarchical clustering (average linkage or Ward's method)
5. **Recursive sub-clustering**: for each cluster, re-identify variant
   positions within the subgroup's MSA. If new variants emerge, split
   and recurse. Continue until no further structure is found.
6. **Post-merge**: merge over-fragmented clusters with < 2% consensus
   divergence.

The recursive approach is key to resolving closely-related haplotypes:
variant positions that are invisible at the population level become
distinguishable within a more homogeneous subgroup.

### Stage 5: Consensus Building

For each cluster, build a consensus sequence:

1. Extract the rows (reads) belonging to this cluster from the corrected MSA
2. At each main position, take the majority vote
3. Require minimum coverage (default 3 reads) and minimum agreement
   (default 60%) for a confident call
4. Compute per-base quality: QV = -10 x log10(1 - agreement_fraction)

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

For genome-scale applications:

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
| Residual error (post-correction) | ~5% | ~0.5% | ~6% |
| MAF threshold | 10% | 2% | 10% |
| Entropy threshold | 0.40 | 0.10 | 0.40 |
| minimap2 preset | map-pb | map-hifi | map-ont |

For HiFi reads, the INDEL correction step is less critical (already high
accuracy) but the pipeline still benefits from the statistical variant
identification and clustering stages.

## References

1. Benjamini Y, Hochberg Y (1995) J Roy Stat Soc Ser B 57:289-300
2. Li H (2018) Bioinformatics 34(18):3094-3100 (minimap2)
3. Dilernia DA et al. (2015) Nucleic Acids Research 43(20):e129
