"""Tests for F81 fast path: must match general eigensub on F81/JC models."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.phylo.jax.types import Tree
from src.phylo.jax.models import jukes_cantor_model, f81_model
from src.phylo.jax import Counts


def _make_7node_tree():
    parentIndex = jnp.array([-1, 0, 0, 1, 1, 2, 2], dtype=jnp.int32)
    distanceToParent = jnp.array([0.0, 0.1, 0.2, 0.15, 0.25, 0.12, 0.18])
    return Tree(parentIndex=parentIndex, distanceToParent=distanceToParent)


class TestF81FastPath:

    def test_jc_fast_matches_general(self):
        """JC fast path should match general eigensub path."""
        tree = _make_7node_tree()
        R = 7
        C = 8
        A = 4
        alignment = jax.random.randint(jax.random.PRNGKey(42), (R, C), 0, A).astype(jnp.int32)

        model = jukes_cantor_model(A)
        counts_general = Counts(alignment, tree, model, f81_fast_flag=False)
        counts_fast = Counts(alignment, tree, model, f81_fast_flag=True)

        np.testing.assert_allclose(counts_fast, counts_general, atol=1e-3, rtol=1e-2)

    def test_f81_fast_matches_general(self):
        """F81 fast path should match general eigensub path."""
        tree = _make_7node_tree()
        R = 7
        C = 6
        A = 4
        pi = jnp.array([0.3, 0.2, 0.25, 0.25])
        alignment = jax.random.randint(jax.random.PRNGKey(7), (R, C), 0, A).astype(jnp.int32)

        model = f81_model(pi)
        counts_general = Counts(alignment, tree, model, f81_fast_flag=False)
        counts_fast = Counts(alignment, tree, model, f81_fast_flag=True)

        np.testing.assert_allclose(counts_fast, counts_general, atol=1e-3, rtol=1e-2)

    def test_jc64_fast_nonnegative(self):
        """JC(64) fast path should produce non-negative counts."""
        tree = _make_7node_tree()
        R = 7
        C = 5
        A = 64
        alignment = jax.random.randint(jax.random.PRNGKey(0), (R, C), 0, A).astype(jnp.int32)

        model = jukes_cantor_model(A)
        counts = Counts(alignment, tree, model, f81_fast_flag=True)

        assert jnp.all(counts >= -1e-6), f"Negative counts: min={jnp.min(counts)}"

    def test_dwell_sums_to_branch_length(self):
        """Sum of dwell times across states should approximate total ungapped branch length."""
        tree = _make_7node_tree()
        R = 7
        C = 4
        A = 4
        # All observed (no gaps) -> all branches active
        alignment = jax.random.randint(jax.random.PRNGKey(5), (R, C), 0, A).astype(jnp.int32)

        model = jukes_cantor_model(A)
        counts = Counts(alignment, tree, model, f81_fast_flag=True)

        # Total dwell per column = sum_i counts[i,i,c]
        total_dwell = sum(counts[i, i, :] for i in range(A))  # (C,)
        # Should be close to total branch length
        total_bl = jnp.sum(tree.distanceToParent[1:])
        np.testing.assert_allclose(total_dwell, total_bl, atol=0.2, rtol=0.1)
