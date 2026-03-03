"""RNA-Seq track preprocessing: BAM → per-position coverage/junction tensors.

Extracts per-position, per-strand:
  - Read coverage
  - Donor junction counts (splice reads starting here)
  - Acceptor junction counts (splice reads ending here)

Applies the Borzoi transformation f(x) = min(x^a, b) + sqrt(max(0, x^a - b))
with a=3/4, b=384 to compress dynamic range into float16.

Output shape per track: (2, 3, C) — 2 strands × 3 channels × C positions.
When multiple tracks are provided: (T, 6, C) — T tracks × 6 channels × C.
"""

import numpy as np

# Borzoi transformation parameters
BORZOI_A = 0.75
BORZOI_B = 384.0


def borzoi_transform(x, a=BORZOI_A, b=BORZOI_B):
    """Borzoi compression: f(x) = min(x^a, b) + sqrt(max(0, x^a - b)).

    Compresses high dynamic range count data into a bounded range
    suitable for neural network input.

    Args:
        x: non-negative array of counts
        a: exponent (default 3/4)
        b: breakpoint (default 384)

    Returns:
        Transformed array (same shape as x), dtype float32.
    """
    x = np.asarray(x, dtype=np.float64)
    xa = np.power(np.maximum(x, 0.0), a)
    return (np.minimum(xa, b) + np.sqrt(np.maximum(0.0, xa - b))).astype(np.float32)


def extract_track_from_bam(bam_path, chrom, start, end):
    """Extract coverage and junction counts from a BAM file for a genomic region.

    Args:
        bam_path: path to indexed BAM file
        chrom: chromosome name (e.g. 'chr1')
        start: 0-based start coordinate
        end: 0-based end coordinate (exclusive)

    Returns:
        (2, 3, C) float32 array:
            [0, :, :] = forward strand (coverage, donors, acceptors)
            [1, :, :] = reverse strand (coverage, donors, acceptors)
    """
    import pysam

    C = end - start
    # coverage[strand][pos], donors[strand][pos], acceptors[strand][pos]
    coverage = np.zeros((2, C), dtype=np.float64)
    donors = np.zeros((2, C), dtype=np.float64)
    acceptors = np.zeros((2, C), dtype=np.float64)

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom, start, end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < 1:
                continue

            strand = 1 if read.is_reverse else 0

            # Coverage from aligned blocks
            for block_start, block_end in read.get_blocks():
                bs = max(block_start - start, 0)
                be = min(block_end - start, C)
                if bs < be:
                    coverage[strand, bs:be] += 1.0

            # Junction reads from CIGAR N operations (introns)
            ref_pos = read.reference_start
            for op, length in read.cigartuples:
                if op == 0:  # M (match/mismatch)
                    ref_pos += length
                elif op == 1:  # I (insertion)
                    pass
                elif op == 2:  # D (deletion)
                    ref_pos += length
                elif op == 3:  # N (skipped region = intron)
                    # Donor at ref_pos (start of intron)
                    donor_pos = ref_pos - start
                    if 0 <= donor_pos < C:
                        donors[strand, donor_pos] += 1.0
                    # Acceptor at ref_pos + length (end of intron)
                    acceptor_pos = ref_pos + length - start
                    if 0 <= acceptor_pos < C:
                        acceptors[strand, acceptor_pos] += 1.0
                    ref_pos += length
                elif op == 4:  # S (soft clip)
                    pass
                elif op == 5:  # H (hard clip)
                    pass
                elif op == 7:  # = (sequence match)
                    ref_pos += length
                elif op == 8:  # X (sequence mismatch)
                    ref_pos += length

    # Stack: (2, 3, C)
    track = np.stack([coverage, donors, acceptors], axis=1)
    return track.astype(np.float32)


def process_track(raw_track, a=BORZOI_A, b=BORZOI_B):
    """Apply Borzoi transform and convert to float16.

    Args:
        raw_track: (2, 3, C) raw count array from extract_track_from_bam
        a: Borzoi exponent
        b: Borzoi breakpoint

    Returns:
        (2, 3, C) float16 array of transformed values.
    """
    transformed = borzoi_transform(raw_track, a=a, b=b)
    return transformed.astype(np.float16)


def prepare_rnaseq_tensor(bam_path, chrom, start, end, a=BORZOI_A, b=BORZOI_B):
    """End-to-end: BAM → Borzoi-transformed float16 tensor.

    Args:
        bam_path: path to indexed BAM file
        chrom: chromosome name
        start: 0-based start
        end: 0-based end (exclusive)
        a: Borzoi exponent (default 3/4)
        b: Borzoi breakpoint (default 384)

    Returns:
        (2, 3, C) float16 tensor.
    """
    raw = extract_track_from_bam(bam_path, chrom, start, end)
    return process_track(raw, a=a, b=b)


def prepare_multi_track_tensor(bam_paths, chrom, start, end, a=BORZOI_A, b=BORZOI_B):
    """Process multiple BAM files into a single tensor.

    Each track's forward and reverse strands are concatenated into 6 channels.

    Args:
        bam_paths: list of BAM file paths
        chrom: chromosome name
        start: 0-based start
        end: 0-based end (exclusive)

    Returns:
        (T, 6, C) float16 tensor where T = number of tracks.
    """
    C = end - start
    T = len(bam_paths)
    result = np.zeros((T, 6, C), dtype=np.float16)

    for t, bam_path in enumerate(bam_paths):
        track = prepare_rnaseq_tensor(bam_path, chrom, start, end, a=a, b=b)
        # Flatten (2, 3, C) → (6, C): [fwd_cov, fwd_don, fwd_acc, rev_cov, rev_don, rev_acc]
        result[t] = track.reshape(6, C)

    return result


def prepare_rnaseq_from_arrays(coverage_fwd, coverage_rev,
                                donors_fwd, donors_rev,
                                acceptors_fwd, acceptors_rev,
                                a=BORZOI_A, b=BORZOI_B):
    """Construct RNA-Seq tensor from pre-computed count arrays.

    Useful when coverage/junction data comes from sources other than BAM
    (e.g. BigWig, precomputed arrays, or synthetic test data).

    Args:
        coverage_fwd, coverage_rev: (C,) arrays of per-position coverage
        donors_fwd, donors_rev: (C,) arrays of donor junction counts
        acceptors_fwd, acceptors_rev: (C,) arrays of acceptor junction counts

    Returns:
        (2, 3, C) float16 tensor.
    """
    raw = np.array([
        [coverage_fwd, donors_fwd, acceptors_fwd],
        [coverage_rev, donors_rev, acceptors_rev],
    ], dtype=np.float32)
    return process_track(raw, a=a, b=b)
