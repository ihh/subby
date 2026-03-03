"""Integration tests for the public API: LogLike, Counts, RootProb."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.phylo.jax import LogLike, Counts, RootProb, MixturePosterior
from src.phylo.jax.types import Tree, RateModel
from src.phylo.jax.models import hky85_diag, jukes_cantor_model, f81_model, gamma_rate_categories, scale_model


def _make_medium_tree(n_leaves=10, key=None):
    """Build a random binary tree with n_leaves leaves."""
    if key is None:
        key = jax.random.PRNGKey(0)
    R = 2 * n_leaves - 1
    # Build tree by iteratively adding leaves to random internal nodes
    # Simple approach: balanced binary tree
    parentIndex = jnp.zeros(R, dtype=jnp.int32)
    parentIndex = parentIndex.at[0].set(-1)
    for i in range(1, R):
        parentIndex = parentIndex.at[i].set((i - 1) // 2)

    k1, k2 = jax.random.split(key)
    distances = jax.random.uniform(k1, (R,), minval=0.01, maxval=0.5)
    distances = distances.at[0].set(0.0)
    return Tree(parentIndex=parentIndex, distanceToParent=distances)


class TestLogLike:

    def test_shape(self):
        tree = _make_medium_tree(10)
        R = tree.parentIndex.shape[0]
        C = 20
        alignment = jax.random.randint(jax.random.PRNGKey(1), (R, C), -1, 4).astype(jnp.int32)
        model = jukes_cantor_model(4)
        ll = LogLike(alignment, tree, model)
        assert ll.shape == (C,)

    def test_loglike_finite(self):
        tree = _make_medium_tree(10)
        R = tree.parentIndex.shape[0]
        C = 15
        alignment = jax.random.randint(jax.random.PRNGKey(2), (R, C), 0, 4).astype(jnp.int32)
        model = hky85_diag(2.0, jnp.array([0.3, 0.2, 0.25, 0.25]))
        ll = LogLike(alignment, tree, model)
        assert jnp.all(jnp.isfinite(ll))
        assert jnp.all(ll <= 0)  # log-likelihoods should be non-positive

    def test_auto_diagonalize(self):
        """LogLike should accept RateModel and auto-diagonalize."""
        tree = _make_medium_tree(5)
        R = tree.parentIndex.shape[0]
        C = 5
        alignment = jax.random.randint(jax.random.PRNGKey(3), (R, C), 0, 4).astype(jnp.int32)

        A = 4
        subRate = jnp.ones((A, A)) / (A - 1)
        subRate = subRate - jnp.diag(jnp.sum(subRate, axis=-1))
        rootProb = jnp.ones(A) / A
        rate_model = RateModel(subRate=subRate, rootProb=rootProb)

        ll = LogLike(alignment, tree, rate_model)
        assert ll.shape == (C,)
        assert jnp.all(jnp.isfinite(ll))


class TestCounts:

    def test_shape(self):
        tree = _make_medium_tree(10)
        R = tree.parentIndex.shape[0]
        C = 15
        alignment = jax.random.randint(jax.random.PRNGKey(4), (R, C), 0, 4).astype(jnp.int32)
        model = jukes_cantor_model(4)
        counts = Counts(alignment, tree, model)
        assert counts.shape == (4, 4, C)

    def test_nonnegative(self):
        tree = _make_medium_tree(8)
        R = tree.parentIndex.shape[0]
        C = 10
        alignment = jax.random.randint(jax.random.PRNGKey(5), (R, C), 0, 4).astype(jnp.int32)
        model = jukes_cantor_model(4)
        counts = Counts(alignment, tree, model)
        assert jnp.all(counts >= -1e-4), f"Negative counts: {jnp.min(counts)}"

    def test_gradient_finite(self):
        """jax.grad of LogLike w.r.t. branch lengths should be finite."""
        tree = _make_medium_tree(5)
        R = tree.parentIndex.shape[0]
        C = 5
        alignment = jax.random.randint(jax.random.PRNGKey(6), (R, C), 0, 4).astype(jnp.int32)
        model = jukes_cantor_model(4)

        def loss(distances):
            t = Tree(parentIndex=tree.parentIndex, distanceToParent=distances)
            return jnp.sum(LogLike(alignment, t, model))

        grad = jax.grad(loss)(tree.distanceToParent)
        assert jnp.all(jnp.isfinite(grad[1:]))  # skip root (distance=0)


class TestRootProb:

    def test_sums_to_one(self):
        tree = _make_medium_tree(8)
        R = tree.parentIndex.shape[0]
        C = 10
        alignment = jax.random.randint(jax.random.PRNGKey(7), (R, C), 0, 4).astype(jnp.int32)
        model = jukes_cantor_model(4)
        rp = RootProb(alignment, tree, model)  # (A, C)
        assert rp.shape == (4, C)
        sums = jnp.sum(rp, axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4)

    def test_nonnegative(self):
        tree = _make_medium_tree(8)
        R = tree.parentIndex.shape[0]
        C = 10
        alignment = jax.random.randint(jax.random.PRNGKey(8), (R, C), 0, 4).astype(jnp.int32)
        model = jukes_cantor_model(4)
        rp = RootProb(alignment, tree, model)
        assert jnp.all(rp >= -1e-6)


class TestMixturePosterior:

    def test_posteriors_sum_to_one(self):
        tree = _make_medium_tree(5)
        R = tree.parentIndex.shape[0]
        C = 8
        alignment = jax.random.randint(jax.random.PRNGKey(9), (R, C), 0, 4).astype(jnp.int32)

        base = hky85_diag(2.0, jnp.array([0.25, 0.25, 0.25, 0.25]))
        models = [scale_model(base, r) for r in [0.5, 1.0, 2.0]]
        log_weights = jnp.log(jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))

        post = MixturePosterior(alignment, tree, models, log_weights)
        assert post.shape == (3, C)
        sums = jnp.sum(post, axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)
