"""Tests for RNA-Seq track preprocessing."""

import numpy as np
import pytest
import tempfile
import os

from src.data.rnaseq import (
    borzoi_transform,
    process_track,
    prepare_rnaseq_from_arrays,
    extract_track_from_bam,
    prepare_rnaseq_tensor,
    prepare_multi_track_tensor,
    BORZOI_A,
    BORZOI_B,
)


class TestBorzoiTransform:
    """Test the Borzoi compression function."""

    def test_zero_input(self):
        x = np.array([0.0])
        result = borzoi_transform(x)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_small_values(self):
        """For small x, x^a < b, so f(x) = x^a."""
        x = np.array([1.0, 4.0, 16.0])
        result = borzoi_transform(x)
        expected = x ** BORZOI_A  # all below breakpoint
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_large_values(self):
        """For large x, x^a >> b, so f(x) ≈ b + sqrt(x^a - b)."""
        x = np.array([1e6, 1e8])
        result = borzoi_transform(x)
        xa = x ** BORZOI_A
        expected = BORZOI_B + np.sqrt(xa - BORZOI_B)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_monotonic(self):
        """Transform should be monotonically increasing."""
        x = np.logspace(0, 6, 100)
        result = borzoi_transform(x)
        assert np.all(np.diff(result) >= 0)

    def test_compression(self):
        """Large values should be heavily compressed."""
        small = borzoi_transform(np.array([10.0]))[0]
        large = borzoi_transform(np.array([1e6]))[0]
        # The ratio of outputs should be much smaller than the ratio of inputs
        assert large / small < 1e6 / 10.0

    def test_vectorized(self):
        """Should work on arrays of any shape."""
        x = np.random.rand(3, 4, 5) * 100
        result = borzoi_transform(x)
        assert result.shape == (3, 4, 5)
        assert result.dtype == np.float32

    def test_negative_clipped(self):
        """Negative values should be treated as zero."""
        x = np.array([-1.0, -100.0])
        result = borzoi_transform(x)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-6)


class TestProcessTrack:
    """Test the process_track function."""

    def test_shape_preserved(self):
        raw = np.random.rand(2, 3, 100).astype(np.float32) * 50
        result = process_track(raw)
        assert result.shape == (2, 3, 100)
        assert result.dtype == np.float16

    def test_nonnegative(self):
        raw = np.random.rand(2, 3, 50).astype(np.float32) * 100
        result = process_track(raw)
        assert np.all(result >= 0)

    def test_zeros(self):
        raw = np.zeros((2, 3, 10), dtype=np.float32)
        result = process_track(raw)
        np.testing.assert_allclose(result, 0.0, atol=1e-3)


class TestPrepareFromArrays:
    """Test the prepare_rnaseq_from_arrays convenience function."""

    def test_basic(self):
        C = 50
        cov_f = np.random.rand(C) * 30
        cov_r = np.random.rand(C) * 20
        don_f = np.random.rand(C) * 5
        don_r = np.random.rand(C) * 3
        acc_f = np.random.rand(C) * 5
        acc_r = np.random.rand(C) * 3

        result = prepare_rnaseq_from_arrays(cov_f, cov_r, don_f, don_r, acc_f, acc_r)
        assert result.shape == (2, 3, C)
        assert result.dtype == np.float16

    def test_channels_independent(self):
        """Each channel should be transformed independently."""
        C = 10
        zeros = np.zeros(C)
        ones = np.ones(C) * 100

        result = prepare_rnaseq_from_arrays(ones, zeros, zeros, zeros, zeros, zeros)
        # Forward coverage should be nonzero, everything else should be ~zero
        assert np.all(result[0, 0, :] > 0)  # fwd coverage
        np.testing.assert_allclose(result[1, :, :], 0.0, atol=1e-3)  # rev all zero


class TestExtractFromBAM:
    """Test BAM extraction with a synthetic BAM file."""

    @pytest.fixture
    def synthetic_bam(self, tmp_path):
        """Create a minimal synthetic BAM file with known reads."""
        import pysam

        bam_path = str(tmp_path / "test.bam")

        # Create BAM with a simple header
        header = pysam.AlignmentHeader.from_dict({
            'HD': {'VN': '1.6', 'SO': 'coordinate'},
            'SQ': [{'SN': 'chr1', 'LN': 1000}],
        })

        with pysam.AlignmentFile(bam_path, "wb", header=header) as outf:
            # Read 1: forward strand, simple match at positions 10-20
            a = pysam.AlignedSegment()
            a.query_name = 'read1'
            a.query_sequence = 'ACGTACGTAC'
            a.flag = 0  # forward strand
            a.reference_id = 0
            a.reference_start = 10
            a.mapping_quality = 30
            a.cigar = [(0, 10)]  # 10M
            a.query_qualities = pysam.qualitystring_to_array('IIIIIIIIII')
            outf.write(a)

            # Read 2: reverse strand, match 30-40
            a = pysam.AlignedSegment()
            a.query_name = 'read2'
            a.query_sequence = 'ACGTACGTAC'
            a.flag = 16  # reverse strand
            a.reference_id = 0
            a.reference_start = 30
            a.mapping_quality = 30
            a.cigar = [(0, 10)]  # 10M
            a.query_qualities = pysam.qualitystring_to_array('IIIIIIIIII')
            outf.write(a)

            # Read 3: forward strand, spliced read: 50-55 then skip 10, then 65-70
            a = pysam.AlignedSegment()
            a.query_name = 'read3'
            a.query_sequence = 'ACGTAACGTA'
            a.flag = 0
            a.reference_id = 0
            a.reference_start = 50
            a.mapping_quality = 30
            a.cigar = [(0, 5), (3, 10), (0, 5)]  # 5M10N5M
            a.query_qualities = pysam.qualitystring_to_array('IIIIIIIIII')
            outf.write(a)

        # Sort and index
        sorted_bam = str(tmp_path / "test_sorted.bam")
        pysam.sort("-o", sorted_bam, bam_path)
        pysam.index(sorted_bam)

        return sorted_bam

    def test_coverage_forward(self, synthetic_bam):
        """Forward strand coverage should be nonzero at read1 positions."""
        track = extract_track_from_bam(synthetic_bam, 'chr1', 0, 100)
        assert track.shape == (2, 3, 100)
        # Forward coverage at positions 10-19
        assert np.all(track[0, 0, 10:20] >= 1.0)
        # No coverage at position 0
        assert track[0, 0, 0] == 0.0

    def test_coverage_reverse(self, synthetic_bam):
        """Reverse strand coverage should be nonzero at read2 positions."""
        track = extract_track_from_bam(synthetic_bam, 'chr1', 0, 100)
        # Reverse coverage at positions 30-39
        assert np.all(track[1, 0, 30:40] >= 1.0)

    def test_junction_donor(self, synthetic_bam):
        """Spliced read should produce donor junction at intron start."""
        track = extract_track_from_bam(synthetic_bam, 'chr1', 0, 100)
        # Read3 has 5M10N5M starting at 50: intron at 55-65
        # Donor at position 55
        assert track[0, 1, 55] >= 1.0  # forward donor

    def test_junction_acceptor(self, synthetic_bam):
        """Spliced read should produce acceptor junction at intron end."""
        track = extract_track_from_bam(synthetic_bam, 'chr1', 0, 100)
        # Read3: intron at 55-65, acceptor at 65
        assert track[0, 2, 65] >= 1.0  # forward acceptor

    def test_full_pipeline(self, synthetic_bam):
        """End-to-end: BAM → float16 tensor."""
        result = prepare_rnaseq_tensor(synthetic_bam, 'chr1', 0, 100)
        assert result.shape == (2, 3, 100)
        assert result.dtype == np.float16
        assert np.all(np.isfinite(result))

    def test_multi_track(self, synthetic_bam):
        """Multiple BAMs → (T, 6, C) tensor."""
        result = prepare_multi_track_tensor(
            [synthetic_bam, synthetic_bam], 'chr1', 0, 100
        )
        assert result.shape == (2, 6, 100)
        assert result.dtype == np.float16
        # Both tracks should be identical
        np.testing.assert_array_equal(result[0], result[1])

    def test_subregion(self, synthetic_bam):
        """Extracting a subregion should work correctly."""
        track = extract_track_from_bam(synthetic_bam, 'chr1', 10, 30)
        assert track.shape == (2, 3, 20)
        # Read1 covers positions 10-19, which is 0-9 in the subregion
        assert np.all(track[0, 0, 0:10] >= 1.0)
