"""Tests for batch_eigensub_maf.py — batched eigensub from MAF files.

Verifies that:
1. vmap batched results match per-block Counts() calls
2. Column concatenation within species groups matches per-block
3. Species exclusion works correctly
4. Validation diagnostics produce reasonable values
5. Tree pruning works correctly

All tests run on CPU (no GPU required).
"""
import os
import sys
import tempfile
import textwrap

os.environ['JAX_ENABLE_X64'] = '1'
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Ensure scripts/ is importable
_subby_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _subby_root not in sys.path:
    sys.path.insert(0, _subby_root)

from subby.formats import Tree, parse_newick
from subby.jax import Counts, pad_alignment
from subby.jax.models import hky85_diag

from scripts.batch_eigensub_maf import (
    iter_maf_blocks,
    _seqs_to_subby_alignment,
    prune_tree,
    _expand_alignment_to_tree,
    _counts_to_features,
    validate_block,
    precompute_maf_gpu,
    FEATURE_DIM,
    COL_BIN,
)


# ---------------------------------------------------------------------------
# Newick guide tree for tests
# ---------------------------------------------------------------------------

# A small tree containing all species used in test blocks:
# mm10, rn6, canFam3, bosTau8, equCab2, hg38, panTro4
GUIDE_TREE_NEWICK = (
    "((hg38:0.01,panTro4:0.015):0.05,"
    "((mm10:0.1,rn6:0.1):0.15,"
    "(canFam3:0.2,(bosTau8:0.18,equCab2:0.18):0.02):0.1):0.05);"
)

@pytest.fixture(scope="module")
def guide_tree():
    return parse_newick(GUIDE_TREE_NEWICK)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hky_model():
    pi = jnp.array([0.295, 0.205, 0.205, 0.295], dtype=jnp.float64)
    return hky85_diag(4.0, pi)


def _make_maf_text(blocks):
    """Build MAF text from a list of block dicts.

    Each block: {'species': [...], 'seqs': [...]}
    Species are used as both the source prefix and species name.
    """
    lines = []
    for block in blocks:
        lines.append("a score=0")
        for sp, seq in zip(block['species'], block['seqs']):
            # s src start size strand srcSize sequence
            lines.append(f"s {sp}.chr1 0 {len(seq)} + 100000 {seq}")
        lines.append("")
    return "\n".join(lines)


def _write_maf_file(blocks):
    """Write blocks to a temporary MAF file, return path."""
    text = _make_maf_text(blocks)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.maf', delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


# Synthetic blocks with 3-5 species, 10-20 columns
BLOCK_A = {
    'species': ['mm10', 'rn6', 'canFam3'],
    'seqs': [
        'ACGTACGTACGTAC',
        'ACGTACGTACGTAC',
        'ACGTACGTATGTAC',
    ],
}

BLOCK_B = {
    'species': ['mm10', 'rn6', 'canFam3'],
    'seqs': [
        'GGCCTTAAGGCCTTAA',
        'GGCCTTAAGGCCTTAA',
        'GGCCTTAAGGCCTTAA',
    ],
}

BLOCK_C = {
    'species': ['mm10', 'rn6', 'canFam3', 'bosTau8', 'equCab2'],
    'seqs': [
        'ACGTACGTACGT',
        'ACGTACGTACGT',
        'ACGTACGTATGT',
        'ACGTACGTATGT',
        'ACGTACGTACGT',
    ],
}

BLOCK_D = {
    'species': ['hg38', 'panTro4', 'mm10', 'rn6'],
    'seqs': [
        'AACCGGTTAACCGGTT',
        'AACCGGTTAACCGGTT',
        'AACCGGTTAACCGGTT',
        'AACCGGTTAACCGGTT',
    ],
}


# ---------------------------------------------------------------------------
# MAF parsing tests
# ---------------------------------------------------------------------------

class TestMafParsing:

    def test_parse_basic(self):
        path = _write_maf_file([BLOCK_A, BLOCK_C])
        try:
            blocks = list(iter_maf_blocks(path))
            assert len(blocks) == 2
            assert blocks[0]['species'] == ['mm10', 'rn6', 'canFam3']
            assert len(blocks[0]['seqs'][0]) == 14
            assert blocks[1]['species'] == ['mm10', 'rn6', 'canFam3', 'bosTau8', 'equCab2']
        finally:
            os.unlink(path)

    def test_parse_preserves_sequences(self):
        path = _write_maf_file([BLOCK_A])
        try:
            blocks = list(iter_maf_blocks(path))
            assert blocks[0]['seqs'][2] == 'ACGTACGTATGTAC'
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Encoding and tree tests
# ---------------------------------------------------------------------------

class TestEncoding:

    def test_alignment_encoding(self):
        seqs = ['ACGT', 'A-GT', 'ANGT']
        aln = _seqs_to_subby_alignment(seqs)
        assert aln.shape == (3, 4)
        # A=0, C=1, G=2, T=3
        np.testing.assert_array_equal(aln[0], [0, 1, 2, 3])
        # gap = 5 (A+1 for A=4)
        assert aln[1, 1] == 5
        # N = 4 (A for A=4)
        assert aln[2, 1] == 4

    def test_prune_tree_basic(self, guide_tree):
        """Pruning to 3 leaves gives a valid tree."""
        pruned = prune_tree(guide_tree, {'mm10', 'rn6', 'canFam3'})
        assert pruned is not None
        assert set(pruned['leaf_names']) == {'mm10', 'rn6', 'canFam3'}
        n_nodes = len(pruned['parentIndex'])
        assert pruned['parentIndex'][0] == -1  # root
        # All non-root nodes have valid parents
        for i in range(1, n_nodes):
            assert 0 <= pruned['parentIndex'][i] < i

    def test_prune_tree_collapses_single_child(self, guide_tree):
        """Pruning to 2 sibling leaves should collapse intermediate nodes."""
        pruned = prune_tree(guide_tree, {'mm10', 'rn6'})
        assert pruned is not None
        assert set(pruned['leaf_names']) == {'mm10', 'rn6'}
        # mm10 and rn6 are siblings, so after collapsing we get 3 nodes:
        # root, mm10, rn6
        assert len(pruned['parentIndex']) == 3

    def test_prune_tree_returns_none_for_single_leaf(self, guide_tree):
        """Pruning to <2 leaves returns None."""
        assert prune_tree(guide_tree, {'mm10'}) is None
        assert prune_tree(guide_tree, set()) is None

    def test_expand_alignment_to_tree(self, guide_tree):
        """Expand alignment maps rows to correct leaf positions by name."""
        species = ['mm10', 'rn6', 'canFam3']
        pruned = prune_tree(guide_tree, set(species))
        aln = np.array([[0, 1, 2], [3, 0, 1], [2, 3, 0]], dtype=np.int32)
        expanded = _expand_alignment_to_tree(aln, species, pruned)

        n_nodes = len(pruned['parentIndex'])
        assert expanded.shape == (n_nodes, 3)

        # Each species should appear at its leaf position
        node_names = pruned['node_names']
        for sp_idx, sp in enumerate(species):
            tree_idx = node_names.index(sp)
            np.testing.assert_array_equal(expanded[tree_idx], aln[sp_idx])

        # Internal nodes should be gap (5)
        parent_index = pruned['parentIndex']
        children_count = np.zeros(n_nodes, dtype=int)
        for i in range(1, n_nodes):
            children_count[parent_index[i]] += 1
        for i in range(n_nodes):
            if children_count[i] > 0:  # internal
                assert expanded[i, 0] == 5


# ---------------------------------------------------------------------------
# Core test: vmap matches per-block Counts()
# ---------------------------------------------------------------------------

class TestVmapMatchesPerBlock:
    """Verify that the vmap batch approach gives identical results to
    calling Counts() on each block individually."""

    def _compute_single_block(self, block, model, guide_tree):
        """Compute eigensub features for one block using plain Counts()."""
        species = block['species']
        seqs = block['seqs']
        L = len(seqs[0])

        pruned = prune_tree(guide_tree, set(species))
        tree = Tree(
            parentIndex=jnp.array(pruned['parentIndex']),
            distanceToParent=jnp.array(pruned['distanceToParent']),
        )

        alignment = _seqs_to_subby_alignment(seqs)
        expanded = _expand_alignment_to_tree(alignment, species, pruned)

        # Pad columns
        expanded_jnp = jnp.array(expanded, dtype=jnp.int32)
        expanded_padded, C_orig = pad_alignment(expanded_jnp, bin_size=COL_BIN)

        counts = Counts(
            expanded_padded, tree, model,
            maxChunkSize=256, branch_mask="auto",
        )

        # Unpad
        counts_np = np.array(counts[:, :, :C_orig], dtype=np.float32)
        return _counts_to_features(counts_np, L)

    def test_single_block_features_shape(self, hky_model, guide_tree):
        features = self._compute_single_block(BLOCK_A, hky_model, guide_tree)
        assert features.shape == (14, FEATURE_DIM)

    def test_vmap_matches_individual(self, hky_model, guide_tree):
        """Two blocks with same species set: vmap batch must match individual."""
        # Both BLOCK_A and BLOCK_B have the same species set
        feat_a = self._compute_single_block(BLOCK_A, hky_model, guide_tree)
        feat_b = self._compute_single_block(BLOCK_B, hky_model, guide_tree)

        # Now do the vmap approach: stack both blocks
        species = BLOCK_A['species']

        pruned = prune_tree(guide_tree, set(species))
        n_nodes = len(pruned['parentIndex'])

        aln_a = _seqs_to_subby_alignment(BLOCK_A['seqs'])
        exp_a = _expand_alignment_to_tree(aln_a, species, pruned)
        aln_b = _seqs_to_subby_alignment(BLOCK_B['seqs'])
        exp_b = _expand_alignment_to_tree(aln_b, species, pruned)

        # Pad to same size
        max_cols = max(exp_a.shape[1], exp_b.shape[1])
        C_padded = ((max_cols + COL_BIN - 1) // COL_BIN) * COL_BIN

        stack = np.full((2, n_nodes, C_padded), 5, dtype=np.int32)
        stack[0, :, :exp_a.shape[1]] = exp_a
        stack[1, :, :exp_b.shape[1]] = exp_b

        parent_stack = np.stack([pruned['parentIndex'], pruned['parentIndex']])
        dist_stack = np.stack([pruned['distanceToParent'], pruned['distanceToParent']])

        def _counts_one(alignment, parent_index, dist_to_parent):
            tree = Tree(parentIndex=parent_index, distanceToParent=dist_to_parent)
            return Counts(alignment, tree, hky_model, maxChunkSize=256, branch_mask="auto")

        all_counts = jax.vmap(_counts_one)(
            jnp.array(stack),
            jnp.array(parent_stack),
            jnp.array(dist_stack),
        )
        all_counts_np = np.array(all_counts, dtype=np.float32)

        vmap_feat_a = _counts_to_features(all_counts_np[0, :, :, :exp_a.shape[1]], exp_a.shape[1])
        vmap_feat_b = _counts_to_features(all_counts_np[1, :, :, :exp_b.shape[1]], exp_b.shape[1])

        np.testing.assert_allclose(vmap_feat_a, feat_a, atol=1e-5)
        np.testing.assert_allclose(vmap_feat_b, feat_b, atol=1e-5)


# ---------------------------------------------------------------------------
# Column concatenation test
# ---------------------------------------------------------------------------

class TestColumnConcatenation:
    """Verify that concatenating columns from same-species blocks and
    computing one Counts() call matches per-block computation."""

    def test_concat_matches_per_block(self, hky_model, guide_tree):
        species = BLOCK_A['species']

        pruned = prune_tree(guide_tree, set(species))
        n_nodes = len(pruned['parentIndex'])
        tree = Tree(
            parentIndex=jnp.array(pruned['parentIndex']),
            distanceToParent=jnp.array(pruned['distanceToParent']),
        )

        # Per-block
        aln_a = _seqs_to_subby_alignment(BLOCK_A['seqs'])
        exp_a = _expand_alignment_to_tree(aln_a, species, pruned)
        aln_b = _seqs_to_subby_alignment(BLOCK_B['seqs'])
        exp_b = _expand_alignment_to_tree(aln_b, species, pruned)

        L_a, L_b = exp_a.shape[1], exp_b.shape[1]

        padded_a, _ = pad_alignment(jnp.array(exp_a), bin_size=COL_BIN)
        padded_b, _ = pad_alignment(jnp.array(exp_b), bin_size=COL_BIN)

        counts_a = np.array(
            Counts(padded_a, tree, hky_model, maxChunkSize=256, branch_mask="auto"),
            dtype=np.float32)[:, :, :L_a]
        counts_b = np.array(
            Counts(padded_b, tree, hky_model, maxChunkSize=256, branch_mask="auto"),
            dtype=np.float32)[:, :, :L_b]

        # Concatenated
        big = np.concatenate([exp_a, exp_b], axis=1)  # (n_nodes, L_a + L_b)
        big_padded, _ = pad_alignment(jnp.array(big), bin_size=COL_BIN)
        counts_concat = np.array(
            Counts(big_padded, tree, hky_model, maxChunkSize=256, branch_mask="auto"),
            dtype=np.float32)

        counts_concat_a = counts_concat[:, :, :L_a]
        counts_concat_b = counts_concat[:, :, L_a:L_a + L_b]

        np.testing.assert_allclose(counts_concat_a, counts_a, atol=1e-5)
        np.testing.assert_allclose(counts_concat_b, counts_b, atol=1e-5)


# ---------------------------------------------------------------------------
# Species exclusion test
# ---------------------------------------------------------------------------

class TestSpeciesExclusion:

    def test_exclude_removes_species(self, guide_tree):
        path = _write_maf_file([BLOCK_D])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pi = jnp.array([0.295, 0.205, 0.205, 0.295], dtype=jnp.float64)
                model = hky85_diag(4.0, pi)

                # Exclude hg38 and panTro4 — leaves mm10, rn6 (2 species)
                manifest = precompute_maf_gpu(
                    path, tmpdir, model,
                    guide_tree=guide_tree,
                    exclude_species={'hg38', 'panTro4'},
                    min_species=2,
                )

                assert len(manifest) == 1
                entry = manifest[0]
                assert entry is not None
                assert entry['n_species'] == 2
                assert entry['n_species_original'] == 4

                # Check saved file
                data = np.load(os.path.join(tmpdir, entry['file']))
                assert data['n_species'] == 2
                species_in_file = list(data['species'])
                assert 'hg38' not in species_in_file
                assert 'panTro4' not in species_in_file
                assert 'mm10' in species_in_file
                assert 'rn6' in species_in_file
        finally:
            os.unlink(path)

    def test_exclude_filters_below_min_species(self, guide_tree):
        """If exclusion drops below min_species, block is skipped."""
        path = _write_maf_file([BLOCK_A])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                pi = jnp.array([0.295, 0.205, 0.205, 0.295], dtype=jnp.float64)
                model = hky85_diag(4.0, pi)

                # Exclude 2 of 3 species — only 1 remains, below min_species=2
                manifest = precompute_maf_gpu(
                    path, tmpdir, model,
                    guide_tree=guide_tree,
                    exclude_species={'mm10', 'rn6'},
                    min_species=2,
                )

                # Block should be skipped
                assert all(m is None for m in manifest) or len(manifest) == 0
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Validation diagnostics test
# ---------------------------------------------------------------------------

class TestValidationDiagnostics:

    def test_diagnostics_shape_and_ranges(self, hky_model, guide_tree):
        """Check that diagnostics produce reasonable values on synthetic data."""
        species = BLOCK_C['species']
        R = len(species)

        pruned = prune_tree(guide_tree, set(species))
        tree = Tree(
            parentIndex=jnp.array(pruned['parentIndex']),
            distanceToParent=jnp.array(pruned['distanceToParent']),
        )

        alignment = _seqs_to_subby_alignment(BLOCK_C['seqs'])
        expanded = _expand_alignment_to_tree(alignment, species, pruned)

        expanded_padded, C_orig = pad_alignment(jnp.array(expanded), bin_size=COL_BIN)
        counts = Counts(expanded_padded, tree, hky_model,
                        maxChunkSize=256, branch_mask="auto")
        counts_np = np.array(counts[:, :, :C_orig], dtype=np.float32)
        L = expanded.shape[1]
        features = _counts_to_features(counts_np, L)

        diag = validate_block(features, R, block_idx=0)

        # Basic checks
        assert diag['n_species'] == 5
        assert diag['n_cols'] == 12
        assert diag['block_idx'] == 0

        # Ti/Tv ratio should be positive (HKY85 with kappa=4 biases transitions)
        assert diag['ti_tv_ratio'] > 0

        # Symmetry ratio should be reasonable (not wildly off from 1)
        assert 0.01 < diag['symmetry_ratio'] < 100.0

        # Dwell times should be positive
        assert diag['total_dwell_per_col'] > 0

        # Count CV should be non-negative
        assert diag['count_cv'] >= 0

        # Mean trans matrix should be 4x4
        mat = np.array(diag['mean_trans_matrix'])
        assert mat.shape == (4, 4)

        # Diagonal should be non-negative (dwell times)
        for i in range(4):
            assert mat[i, i] >= 0

    def test_conserved_block_has_high_dominant_frac(self, hky_model, guide_tree):
        """A perfectly conserved block should have high dominant base fraction."""
        # All species have identical sequence
        conserved_block = {
            'species': ['mm10', 'rn6', 'canFam3'],
            'seqs': [
                'AAAAAAAAAAAAAAAA',
                'AAAAAAAAAAAAAAAA',
                'AAAAAAAAAAAAAAAA',
            ],
        }

        species = conserved_block['species']
        R = len(species)
        pruned = prune_tree(guide_tree, set(species))
        tree = Tree(
            parentIndex=jnp.array(pruned['parentIndex']),
            distanceToParent=jnp.array(pruned['distanceToParent']),
        )

        alignment = _seqs_to_subby_alignment(conserved_block['seqs'])
        expanded = _expand_alignment_to_tree(alignment, species, pruned)
        expanded_padded, C_orig = pad_alignment(jnp.array(expanded), bin_size=COL_BIN)
        counts = Counts(expanded_padded, tree, hky_model,
                        maxChunkSize=256, branch_mask="auto")
        counts_np = np.array(counts[:, :, :C_orig], dtype=np.float32)
        features = _counts_to_features(counts_np, expanded.shape[1])

        diag = validate_block(features, R, block_idx=0)

        # For a perfectly conserved block, the dominant base should overwhelm
        if not np.isnan(diag['conserved_dominant_frac']):
            assert diag['conserved_dominant_frac'] > 0.5


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline(self, hky_model, guide_tree):
        """Run the full pipeline on a small synthetic MAF and check outputs."""
        path = _write_maf_file([BLOCK_A, BLOCK_B, BLOCK_C])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                manifest = precompute_maf_gpu(
                    path, tmpdir, hky_model,
                    guide_tree=guide_tree,
                    min_species=3,
                    min_cols=5,
                    validate_n=2,
                )

                # Should have 3 blocks
                assert len(manifest) == 3
                for entry in manifest:
                    assert entry is not None
                    fpath = os.path.join(tmpdir, entry['file'])
                    assert os.path.exists(fpath)

                    data = np.load(fpath)
                    assert data['features'].shape[1] == FEATURE_DIM
                    assert data['features'].shape[0] == entry['length']

                # Check manifest.json would be valid
                assert manifest[0]['n_species'] == 3
                assert manifest[2]['n_species'] == 5
        finally:
            os.unlink(path)
