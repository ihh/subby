"""Tests for subby JSON I/O and CLI."""

import json
import subprocess
import sys
import tempfile
import os

import numpy as np
import pytest

from subby.io import load_input, run, _parse_model, _parse_alignment
from subby.oracle import oracle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def jc4_input():
    """Jukes-Cantor 4-state input with a 5-node tree."""
    return {
        "model": {
            "name": "jukes-cantor",
            "alphabetSize": 4,
            "alphabet": ["A", "C", "G", "T"],
        },
        "tree": {
            "parentIndex": [-1, 0, 0, 1, 1],
            "distanceToParent": [0.0, 0.1, 0.2, 0.15, 0.25],
        },
        "alignment": [[0, 1], [2, 3], [1, 0], [3, 2], [0, 1]],
    }


@pytest.fixture
def jc4_char_input():
    """Same as jc4_input but with character alignment."""
    return {
        "model": {
            "name": "jukes-cantor",
            "alphabetSize": 4,
            "alphabet": ["A", "C", "G", "T"],
        },
        "tree": {
            "parentIndex": [-1, 0, 0, 1, 1],
            "distanceToParent": [0.0, 0.1, 0.2, 0.15, 0.25],
        },
        "alignment": ["AC", "GT", "CA", "TG", "AC"],
    }


@pytest.fixture
def rate_matrix_input():
    """Rate matrix model input (JC4 as explicit rate matrix)."""
    mu = 4.0 / 3.0
    off = mu / 4.0
    diag = -mu + off
    return {
        "model": {
            "alphabet": ["A", "C", "G", "T"],
            "rootProb": [0.25, 0.25, 0.25, 0.25],
            "subRate": [
                [diag, off, off, off],
                [off, diag, off, off],
                [off, off, diag, off],
                [off, off, off, diag],
            ],
        },
        "tree": {
            "parentIndex": [-1, 0, 0, 1, 1],
            "distanceToParent": [0.0, 0.1, 0.2, 0.15, 0.25],
        },
        "alignment": [[0, 1], [2, 3], [1, 0], [3, 2], [0, 1]],
    }


# ---------------------------------------------------------------------------
# Test model parsing
# ---------------------------------------------------------------------------

class TestParseModel:
    def test_jukes_cantor(self):
        model, alphabet = _parse_model({"name": "jukes-cantor", "alphabetSize": 4})
        assert len(model["pi"]) == 4
        assert alphabet is None

    def test_hky85(self):
        model, alphabet = _parse_model({
            "name": "hky85",
            "kappa": 2.0,
            "pi": [0.3, 0.2, 0.2, 0.3],
        })
        assert len(model["pi"]) == 4
        np.testing.assert_allclose(model["pi"], [0.3, 0.2, 0.2, 0.3])

    def test_f81(self):
        model, alphabet = _parse_model({
            "name": "f81",
            "pi": [0.1, 0.2, 0.3, 0.4],
        })
        assert len(model["pi"]) == 4

    def test_rate_matrix(self):
        model, alphabet = _parse_model({
            "alphabet": ["A", "C", "G", "T"],
            "rootProb": [0.25, 0.25, 0.25, 0.25],
            "subRate": [
                [-1, 1.0/3, 1.0/3, 1.0/3],
                [1.0/3, -1, 1.0/3, 1.0/3],
                [1.0/3, 1.0/3, -1, 1.0/3],
                [1.0/3, 1.0/3, 1.0/3, -1],
            ],
        })
        assert "eigenvalues" in model
        assert alphabet == ["A", "C", "G", "T"]

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            _parse_model({"name": "bogus", "alphabetSize": 4})

    def test_missing_spec(self):
        with pytest.raises(ValueError, match="must have"):
            _parse_model({"alphabet": ["A", "C"]})


# ---------------------------------------------------------------------------
# Test alignment parsing
# ---------------------------------------------------------------------------

class TestParseAlignment:
    def test_integer_matrix(self):
        aln = _parse_alignment([[0, 1], [2, 3]], None)
        assert aln.shape == (2, 2)
        assert aln.dtype == np.int32

    def test_character_strings(self):
        aln = _parse_alignment(["AC", "GT"], ["A", "C", "G", "T"])
        np.testing.assert_array_equal(aln, [[0, 1], [2, 3]])

    def test_character_strings_with_gap(self):
        aln = _parse_alignment(["A-", "GT"], ["A", "C", "G", "T"])
        assert aln[0, 1] == 5  # gap token = A+1 = 5

    def test_character_matrix(self):
        aln = _parse_alignment([["A", "C"], ["G", "T"]], ["A", "C", "G", "T"])
        np.testing.assert_array_equal(aln, [[0, 1], [2, 3]])

    def test_requires_alphabet(self):
        with pytest.raises(ValueError, match="Alphabet required"):
            _parse_alignment(["AC", "GT"], None)


# ---------------------------------------------------------------------------
# Test run() end-to-end
# ---------------------------------------------------------------------------

class TestRun:
    def test_all_outputs(self, jc4_input):
        result = run(jc4_input)
        assert "logLike" in result
        assert "counts" in result
        assert "dwellTimes" in result
        assert "rootProb" in result

        # Check shapes via list lengths
        C = 2
        A = 4
        assert len(result["logLike"]) == C
        assert len(result["counts"]) == A
        assert len(result["counts"][0]) == A
        assert len(result["counts"][0][0]) == C
        assert len(result["dwellTimes"]) == A
        assert len(result["dwellTimes"][0]) == C
        assert len(result["rootProb"]) == A
        assert len(result["rootProb"][0]) == C

    def test_selective_compute(self, jc4_input):
        jc4_input["compute"] = ["logLike"]
        result = run(jc4_input)
        assert "logLike" in result
        assert "counts" not in result
        assert "rootProb" not in result

    def test_char_alignment_matches_int(self, jc4_input, jc4_char_input):
        result_int = run(jc4_input)
        result_char = run(jc4_char_input)
        np.testing.assert_allclose(result_int["logLike"], result_char["logLike"])
        np.testing.assert_allclose(result_int["counts"], result_char["counts"])
        np.testing.assert_allclose(result_int["rootProb"], result_char["rootProb"])

    def test_rate_matrix_matches_named(self, jc4_input, rate_matrix_input):
        result_named = run(jc4_input)
        result_rate = run(rate_matrix_input)
        np.testing.assert_allclose(
            result_named["logLike"], result_rate["logLike"], atol=1e-10,
        )
        np.testing.assert_allclose(
            result_named["counts"], result_rate["counts"], atol=1e-8,
        )

    def test_matches_oracle_directly(self, jc4_input):
        """Verify JSON I/O results match direct oracle calls."""
        result = run(jc4_input)

        # Build oracle inputs directly
        from subby.formats import Tree
        model = oracle.jukes_cantor_model(4)
        tree = Tree(
            parentIndex=np.array([-1, 0, 0, 1, 1], dtype=np.intp),
            distanceToParent=np.array([0.0, 0.1, 0.2, 0.15, 0.25]),
        )
        alignment = np.array([[0, 1], [2, 3], [1, 0], [3, 2], [0, 1]], dtype=np.int32)

        ll = oracle.LogLike(alignment, tree, model)
        counts = oracle.Counts(alignment, tree, model)
        rp = oracle.RootProb(alignment, tree, model)

        np.testing.assert_allclose(result["logLike"], ll.tolist(), atol=1e-15)
        np.testing.assert_allclose(result["counts"], counts.tolist(), atol=1e-15)
        np.testing.assert_allclose(result["rootProb"], rp.tolist(), atol=1e-15)

    def test_dwell_times_are_counts_diagonal(self, jc4_input):
        result = run(jc4_input)
        A = len(result["counts"])
        for a in range(A):
            np.testing.assert_allclose(
                result["dwellTimes"][a], result["counts"][a][a],
            )


# ---------------------------------------------------------------------------
# Test CLI (subprocess)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_file_input(self, jc4_input, tmp_path):
        in_file = tmp_path / "input.json"
        in_file.write_text(json.dumps(jc4_input))

        result = subprocess.run(
            [sys.executable, "-m", "subby", str(in_file)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "logLike" in output
        assert "counts" in output

    def test_stdin_input(self, jc4_input):
        result = subprocess.run(
            [sys.executable, "-m", "subby"],
            input=json.dumps(jc4_input),
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "logLike" in output

    def test_output_file(self, jc4_input, tmp_path):
        in_file = tmp_path / "input.json"
        out_file = tmp_path / "output.json"
        in_file.write_text(json.dumps(jc4_input))

        result = subprocess.run(
            [sys.executable, "-m", "subby", str(in_file), "-o", str(out_file)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        output = json.loads(out_file.read_text())
        assert "logLike" in output

    def test_validate_valid(self, jc4_input, tmp_path):
        in_file = tmp_path / "input.json"
        in_file.write_text(json.dumps(jc4_input))

        result = subprocess.run(
            [sys.executable, "-m", "subby", "--validate", str(in_file)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Valid" in result.stderr

    def test_validate_invalid(self, tmp_path):
        in_file = tmp_path / "input.json"
        in_file.write_text(json.dumps({"model": {"name": "bogus"}}))

        result = subprocess.run(
            [sys.executable, "-m", "subby", "--validate", str(in_file)],
            capture_output=True, text=True,
        )
        assert result.returncode == 1
        assert "Invalid" in result.stderr
