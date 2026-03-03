# RNA-Seq Preprocessing API Reference

The RNA-Seq module extracts per-position, per-strand coverage and splice junction counts from BAM files, compresses them via the Borzoi transformation, and outputs float16 tensors suitable for the track encoder.

## Borzoi transformation

The Borzoi transformation compresses high dynamic range count data:

$$f(x) = \min(x^a, b) + \sqrt{\max(0, x^a - b)}$$

with $a = 3/4$ and $b = 384$. This is approximately $x^{3/4}$ for small counts and $\sqrt{x^{3/4}}$ for very large counts.

## Output format

Each RNA-Seq track produces a `(2, 3, C)` tensor:

| Dimension | Index | Channel |
|-----------|-------|---------|
| Strand 0 | `[0, 0, :]` | Forward coverage |
| Strand 0 | `[0, 1, :]` | Forward donor junctions |
| Strand 0 | `[0, 2, :]` | Forward acceptor junctions |
| Strand 1 | `[1, 0, :]` | Reverse coverage |
| Strand 1 | `[1, 1, :]` | Reverse donor junctions |
| Strand 1 | `[1, 2, :]` | Reverse acceptor junctions |

For multiple tracks, flattened to `(T, 6, C)` where the 6 channels are `[fwd_cov, fwd_don, fwd_acc, rev_cov, rev_don, rev_acc]`.

## Functions

### `borzoi_transform(x, a=0.75, b=384.0)`

Apply the Borzoi transformation to an array.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | array | — | Non-negative counts |
| `a` | float | 0.75 | Exponent |
| `b` | float | 384.0 | Breakpoint |

**Returns:** float32 array (same shape).

### `extract_track_from_bam(bam_path, chrom, start, end)`

Extract raw coverage and junction counts from a BAM file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bam_path` | str | Path to indexed BAM file |
| `chrom` | str | Chromosome name (e.g. `'chr1'`) |
| `start` | int | 0-based start coordinate |
| `end` | int | 0-based end coordinate (exclusive) |

**Returns:** `(2, 3, C)` float32 array of raw counts.

Filters: unmapped, secondary, supplementary, and MAPQ < 1 reads are skipped.
Coverage is computed from aligned blocks. Donor/acceptor counts come from CIGAR `N` (intron) operations.

### `process_track(raw_track, a=0.75, b=384.0)`

Apply Borzoi transform and convert to float16.

**Returns:** `(2, 3, C)` float16 array.

### `prepare_rnaseq_tensor(bam_path, chrom, start, end)`

End-to-end: BAM to Borzoi-transformed float16 tensor.

**Returns:** `(2, 3, C)` float16.

### `prepare_multi_track_tensor(bam_paths, chrom, start, end)`

Process multiple BAM files into a single tensor.

**Returns:** `(T, 6, C)` float16 where T = number of tracks.

### `prepare_rnaseq_from_arrays(coverage_fwd, coverage_rev, donors_fwd, donors_rev, acceptors_fwd, acceptors_rev)`

Construct RNA-Seq tensor from pre-computed count arrays (e.g. from BigWig or synthetic data).

**Returns:** `(2, 3, C)` float16.
