"""Integration tests for the public API: LogLike, Counts, RootProb."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from subby.jax import LogLike, Counts, RootProb, MixturePosterior, LogLikeCustomGrad
from subby.jax.types import Tree, RateModel
from subby.jax.models import hky85_diag, jukes_cantor_model, f81_model, gamma_rate_categories, scale_model, irrev_model_from_rate_matrix


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


class TestCustomVJP:

    def test_grad_matches_autograd_reversible(self):
        """Custom VJP distance gradient should match autograd."""
        tree = _make_medium_tree(5)
        R = tree.parentIndex.shape[0]
        C = 5
        alignment = jax.random.randint(jax.random.PRNGKey(10), (R, C), 0, 4).astype(jnp.int32)
        model = jukes_cantor_model(4)

        def loss_auto(distances):
            t = Tree(parentIndex=tree.parentIndex, distanceToParent=distances)
            return jnp.sum(LogLike(alignment, t, model))

        def loss_custom(distances):
            t = Tree(parentIndex=tree.parentIndex, distanceToParent=distances)
            return jnp.sum(LogLikeCustomGrad(alignment, t, model))

        grad_auto = jax.grad(loss_auto)(tree.distanceToParent)
        grad_custom = jax.grad(loss_custom)(tree.distanceToParent)

        np.testing.assert_allclose(grad_custom[1:], grad_auto[1:], atol=1e-6)

    def test_grad_matches_autograd_hky(self):
        """Custom VJP with HKY85 model."""
        tree = _make_medium_tree(8)
        R = tree.parentIndex.shape[0]
        C = 10
        alignment = jax.random.randint(jax.random.PRNGKey(11), (R, C), 0, 4).astype(jnp.int32)
        model = hky85_diag(2.0, jnp.array([0.3, 0.2, 0.25, 0.25]))

        def loss_auto(distances):
            t = Tree(parentIndex=tree.parentIndex, distanceToParent=distances)
            return jnp.sum(LogLike(alignment, t, model))

        def loss_custom(distances):
            t = Tree(parentIndex=tree.parentIndex, distanceToParent=distances)
            return jnp.sum(LogLikeCustomGrad(alignment, t, model))

        grad_auto = jax.grad(loss_auto)(tree.distanceToParent)
        grad_custom = jax.grad(loss_custom)(tree.distanceToParent)

        np.testing.assert_allclose(grad_custom[1:], grad_auto[1:], atol=1e-6)

    def test_grad_matches_autograd_irreversible(self):
        """Custom VJP with irreversible model."""
        tree = _make_medium_tree(5)
        R = tree.parentIndex.shape[0]
        C = 5
        alignment = jax.random.randint(jax.random.PRNGKey(12), (R, C), 0, 4).astype(jnp.int32)

        A = 4
        rng = np.random.RandomState(42)
        rate = rng.uniform(0.01, 1.0, (A, A))
        np.fill_diagonal(rate, 0.0)
        np.fill_diagonal(rate, -rate.sum(axis=1))
        pi = np.ones(A) / A
        norm = -np.sum(pi * np.diag(rate))
        rate /= norm
        model = irrev_model_from_rate_matrix(jnp.array(rate), jnp.array(pi))

        def loss_auto(distances):
            t = Tree(parentIndex=tree.parentIndex, distanceToParent=distances)
            return jnp.sum(LogLike(alignment, t, model))

        def loss_custom(distances):
            t = Tree(parentIndex=tree.parentIndex, distanceToParent=distances)
            return jnp.sum(LogLikeCustomGrad(alignment, t, model))

        grad_auto = jax.grad(loss_auto)(tree.distanceToParent)
        grad_custom = jax.grad(loss_custom)(tree.distanceToParent)

        np.testing.assert_allclose(grad_custom[1:].real, grad_auto[1:].real, atol=1e-5)

    def test_forward_values_match(self):
        """Custom VJP forward pass should give same logLike values."""
        tree = _make_medium_tree(5)
        R = tree.parentIndex.shape[0]
        C = 8
        alignment = jax.random.randint(jax.random.PRNGKey(13), (R, C), 0, 4).astype(jnp.int32)
        model = jukes_cantor_model(4)

        ll_auto = LogLike(alignment, tree, model)
        ll_custom = LogLikeCustomGrad(alignment, tree, model)

        np.testing.assert_allclose(ll_custom, ll_auto, atol=1e-10)


class TestPerColumnModel:

    def test_single_model_list_matches(self):
        """[model] * C should give same result as single model."""
        tree = _make_medium_tree(5)
        R = tree.parentIndex.shape[0]
        C = 5
        alignment = jax.random.randint(jax.random.PRNGKey(14), (R, C), 0, 4).astype(jnp.int32)
        model = jukes_cantor_model(4)

        ll_single = LogLike(alignment, tree, model)
        ll_list = LogLike(alignment, tree, [model] * C)

        np.testing.assert_allclose(ll_list, ll_single, atol=1e-8)

    def test_different_rates_per_column(self):
        """Different rates per column vs column-by-column computation."""
        tree = _make_medium_tree(5)
        R = tree.parentIndex.shape[0]
        C = 4
        alignment = jax.random.randint(jax.random.PRNGKey(15), (R, C), 0, 4).astype(jnp.int32)

        base = hky85_diag(2.0, jnp.array([0.25, 0.25, 0.25, 0.25]))
        rates = [0.5, 1.0, 1.5, 2.0]
        models = [scale_model(base, r) for r in rates]

        # Per-column result
        ll_per_col = LogLike(alignment, tree, models)

        # Column-by-column reference
        ll_ref = jnp.array([
            LogLike(alignment[:, c:c+1], tree, models[c])[0]
            for c in range(C)
        ])

        np.testing.assert_allclose(ll_per_col, ll_ref, atol=1e-8)

    def test_rootprob_per_column(self):
        """RootProb with per-column models should sum to 1."""
        tree = _make_medium_tree(5)
        R = tree.parentIndex.shape[0]
        C = 3
        alignment = jax.random.randint(jax.random.PRNGKey(16), (R, C), 0, 4).astype(jnp.int32)

        base = jukes_cantor_model(4)
        models = [scale_model(base, r) for r in [0.5, 1.0, 2.0]]

        rp = RootProb(alignment, tree, models)
        assert rp.shape == (4, C)
        sums = jnp.sum(rp, axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-4)


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
