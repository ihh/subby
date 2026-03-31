"""Tests for geometric bin padding (tree + alignment)."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax.numpy as jnp
import numpy as np
import pytest

from subby.jax._utils import (
    GEOM_BINS,
    pad_to_geom_bin,
    pad_alignment,
    pad_tree,
    pad_tree_and_alignment,
)
from subby.jax.types import Tree


# --- pad_to_geom_bin ---

def test_pad_to_geom_bin_small():
    assert pad_to_geom_bin(1) == 1
    assert pad_to_geom_bin(2) == 2
    assert pad_to_geom_bin(3) == 3
    assert pad_to_geom_bin(4) == 4
    assert pad_to_geom_bin(5) == 5


def test_pad_to_geom_bin_exact():
    """Exact bin sizes should not be rounded up."""
    for b in GEOM_BINS[:20]:
        assert pad_to_geom_bin(b) == b


def test_pad_to_geom_bin_between():
    """Values between bins round up to next bin."""
    for i in range(len(GEOM_BINS) - 1):
        lo, hi = GEOM_BINS[i], GEOM_BINS[i + 1]
        if hi > lo + 1:
            assert pad_to_geom_bin(lo + 1) == hi


def test_pad_to_geom_bin_monotone():
    """pad_to_geom_bin(n) >= n for all n."""
    for n in range(1, 500):
        assert pad_to_geom_bin(n) >= n


def test_geom_bins_ologn():
    """Number of distinct bin sizes up to 10000 should be O(log n)."""
    bins_up_to_10k = [b for b in GEOM_BINS if b <= 10000]
    # With ratio ~1.19, we expect ~log(10000)/log(1.19) ~ 53 bins
    assert len(bins_up_to_10k) < 80


# --- pad_alignment backward compat ---

def test_pad_alignment_linear_default():
    """Default behavior: pad to next multiple of bin_size."""
    aln = jnp.zeros((5, 100), dtype=jnp.int32)
    padded, C_orig = pad_alignment(aln, bin_size=128)
    assert C_orig == 100
    assert padded.shape == (5, 128)


def test_pad_alignment_geom():
    """With bin_fn=pad_to_geom_bin, uses geometric bins."""
    aln = jnp.zeros((5, 100), dtype=jnp.int32)
    padded, C_orig = pad_alignment(aln, bin_fn=pad_to_geom_bin)
    assert C_orig == 100
    assert padded.shape[1] == pad_to_geom_bin(100)
    assert padded.shape[1] >= 100


# --- pad_tree ---

def test_pad_tree_valid():
    """Padded tree maintains parentIndex[i] < i (except root) invariant."""
    parent = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
    dist = jnp.array([0.0, 0.1, 0.2, 0.15, 0.25])
    tree = Tree(parentIndex=parent, distanceToParent=dist)
    padded = pad_tree(tree, 8)
    assert padded.parentIndex.shape[0] == 8
    assert padded.distanceToParent.shape[0] == 8
    # Root
    assert int(padded.parentIndex[0]) == -1
    # Original nodes preserved
    np.testing.assert_array_equal(padded.parentIndex[:5], parent)
    np.testing.assert_array_equal(padded.distanceToParent[:5], dist)
    # Padded nodes: parent=0, dist=0
    for i in range(5, 8):
        assert int(padded.parentIndex[i]) == 0
        assert float(padded.distanceToParent[i]) == 0.0
    # parentIndex[i] < i for all i > 0
    for i in range(1, 8):
        assert int(padded.parentIndex[i]) < i


def test_pad_tree_noop():
    """No-op when already at target size."""
    parent = jnp.array([-1, 0, 0], dtype=jnp.int32)
    dist = jnp.array([0.0, 0.1, 0.2])
    tree = Tree(parentIndex=parent, distanceToParent=dist)
    padded = pad_tree(tree, 3)
    assert padded.parentIndex.shape[0] == 3


# --- pad_tree_and_alignment ---

def test_pad_tree_and_alignment_shapes():
    parent = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
    dist = jnp.array([0.0, 0.1, 0.2, 0.15, 0.25])
    tree = Tree(parentIndex=parent, distanceToParent=dist)
    aln = jnp.zeros((5, 37), dtype=jnp.int32)

    ptree, paln, R_real, C_real = pad_tree_and_alignment(tree, aln)
    assert R_real == 5
    assert C_real == 37
    assert ptree.parentIndex.shape[0] == paln.shape[0]
    assert paln.shape[0] >= 5
    assert paln.shape[1] >= 37
    assert paln.shape[0] == pad_to_geom_bin(5)
    assert paln.shape[1] == pad_to_geom_bin(37)


def test_pad_tree_and_alignment_preserves_data():
    """Original alignment data is preserved in top-left block."""
    parent = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
    dist = jnp.array([0.0, 0.1, 0.2, 0.15, 0.25])
    tree = Tree(parentIndex=parent, distanceToParent=dist)
    aln = jnp.array([
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
        [0, 0, 0, 0],
    ], dtype=jnp.int32)

    ptree, paln, R_real, C_real = pad_tree_and_alignment(tree, aln)
    np.testing.assert_array_equal(paln[:R_real, :C_real], aln)
    # Padded regions should be -1 (gap)
    if paln.shape[1] > C_real:
        assert jnp.all(paln[:R_real, C_real:] == -1)
    if paln.shape[0] > R_real:
        assert jnp.all(paln[R_real:, :] == -1)


def test_pad_tree_and_alignment_pruning_neutral():
    """Padded tree+alignment gives same log-likelihoods on real columns.

    This is the key correctness test: padding must not change results.
    """
    from subby.jax import LogLike
    from subby.jax.models import jukes_cantor_model

    # 3-leaf tree: root(0) -> internal(1), leaf(2); internal(1) -> leaf(3), leaf(4)
    parent = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
    dist = jnp.array([0.0, 0.1, 0.2, 0.15, 0.25])
    tree = Tree(parentIndex=parent, distanceToParent=dist)

    # Alignment: 5 nodes, 7 columns
    # Leaves (2, 3, 4) observed; internal nodes (0, 1) unobserved (token=4 for A=4)
    A = 4
    rng = np.random.RandomState(42)
    leaf_data = rng.randint(0, A, size=(3, 7))
    aln = np.full((5, 7), A, dtype=np.int32)  # A = ungapped-unobserved
    aln[2] = leaf_data[0]
    aln[3] = leaf_data[1]
    aln[4] = leaf_data[2]
    aln = jnp.array(aln)

    model = jukes_cantor_model(A)

    # Unpadded
    ll_orig = LogLike(aln, tree, model)

    # Padded
    ptree, paln, R_real, C_real = pad_tree_and_alignment(tree, aln)
    ll_padded = LogLike(paln, ptree, model)

    # Real columns must match
    np.testing.assert_allclose(
        np.array(ll_padded[..., :C_real]),
        np.array(ll_orig),
        atol=1e-12,
    )


def test_pad_tree_and_alignment_mixture_posterior_neutral():
    """Padded tree+alignment gives same MixturePosterior on real columns."""
    from subby.jax import MixturePosterior
    from subby.jax.models import jukes_cantor_model, scale_model

    parent = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
    dist = jnp.array([0.0, 0.1, 0.2, 0.15, 0.25])
    tree = Tree(parentIndex=parent, distanceToParent=dist)

    A = 4
    rng = np.random.RandomState(123)
    leaf_data = rng.randint(0, A, size=(3, 11))
    aln = np.full((5, 11), A, dtype=np.int32)
    aln[2] = leaf_data[0]
    aln[3] = leaf_data[1]
    aln[4] = leaf_data[2]
    aln = jnp.array(aln)

    base = jukes_cantor_model(A)
    models = [scale_model(base, r) for r in [0.5, 1.0, 2.0]]
    lw = jnp.log(jnp.ones(3) / 3.0)

    # Unpadded
    post_orig = MixturePosterior(aln, tree, models, lw)

    # Padded
    ptree, paln, R_real, C_real = pad_tree_and_alignment(tree, aln)
    post_padded = MixturePosterior(paln, ptree, models, lw)

    np.testing.assert_allclose(
        np.array(post_padded[:, :C_real]),
        np.array(post_orig),
        atol=1e-12,
    )


def test_distinct_bin_shapes_small():
    """Verify that distinct (R_pad, C_pad) shapes are manageable."""
    shapes = set()
    for R in range(3, 200):
        for C in range(5, 2000, 17):
            shapes.add((pad_to_geom_bin(R), pad_to_geom_bin(C)))
    # Should be much less than 200*118 = 23600 raw combinations
    # With ~30 node bins x ~30 col bins we get ~900 max; in practice ~600
    assert len(shapes) < 1000, f"Too many distinct shapes: {len(shapes)}"
