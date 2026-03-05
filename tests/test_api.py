"""Integration tests for the public API: LogLike, Counts, RootProb."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from subby.jax import LogLike, Counts, BranchCounts, RootProb, MixturePosterior, LogLikeCustomGrad, pad_alignment, unpad_columns, InsideOutside
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


class TestPadAlignment:

    def _make_setup(self, C=10, n_leaves=5, seed=42):
        tree = _make_medium_tree(n_leaves, key=jax.random.PRNGKey(seed))
        R = tree.parentIndex.shape[0]
        alignment = jax.random.randint(
            jax.random.PRNGKey(seed + 1), (R, C), 0, 4
        ).astype(jnp.int32)
        model = jukes_cantor_model(4)
        return alignment, tree, model

    def test_loglike_round_trip(self):
        alignment, tree, model = self._make_setup(C=10)
        padded, C_orig = pad_alignment(alignment, bin_size=8)
        ll_orig = LogLike(alignment, tree, model)
        ll_padded = unpad_columns(LogLike(padded, tree, model), C_orig)
        np.testing.assert_allclose(ll_padded, ll_orig, atol=1e-12)

    def test_counts_round_trip(self):
        alignment, tree, model = self._make_setup(C=10)
        padded, C_orig = pad_alignment(alignment, bin_size=8)
        counts_orig = Counts(alignment, tree, model)
        counts_padded = unpad_columns(Counts(padded, tree, model), C_orig)
        np.testing.assert_allclose(counts_padded, counts_orig, atol=1e-12)

    def test_rootprob_round_trip(self):
        alignment, tree, model = self._make_setup(C=10)
        padded, C_orig = pad_alignment(alignment, bin_size=8)
        rp_orig = RootProb(alignment, tree, model)
        rp_padded = unpad_columns(RootProb(padded, tree, model), C_orig)
        np.testing.assert_allclose(rp_padded, rp_orig, atol=1e-12)

    def test_padded_columns_loglike_zero(self):
        alignment, tree, model = self._make_setup(C=10)
        padded, C_orig = pad_alignment(alignment, bin_size=8)
        ll = LogLike(padded, tree, model)
        np.testing.assert_allclose(ll[C_orig:], 0.0, atol=1e-12)

    def test_noop_when_aligned(self):
        alignment, tree, model = self._make_setup(C=16)
        padded, C_orig = pad_alignment(alignment, bin_size=8)
        assert C_orig == 16
        assert padded.shape[1] == 16
        np.testing.assert_array_equal(padded, alignment)

    @pytest.mark.parametrize("bin_size", [8, 16, 32])
    def test_various_bin_sizes(self, bin_size):
        alignment, tree, model = self._make_setup(C=10)
        padded, C_orig = pad_alignment(alignment, bin_size=bin_size)
        assert padded.shape[1] % bin_size == 0
        assert C_orig == 10
        ll_orig = LogLike(alignment, tree, model)
        ll_padded = unpad_columns(LogLike(padded, tree, model), C_orig)
        np.testing.assert_allclose(ll_padded, ll_orig, atol=1e-12)

    def test_irreversible_model_round_trip(self):
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(50))
        R = tree.parentIndex.shape[0]
        C = 10
        alignment = jax.random.randint(
            jax.random.PRNGKey(51), (R, C), 0, 4
        ).astype(jnp.int32)

        A = 4
        rng = np.random.RandomState(99)
        rate = rng.uniform(0.01, 1.0, (A, A))
        np.fill_diagonal(rate, 0.0)
        np.fill_diagonal(rate, -rate.sum(axis=1))
        pi = np.ones(A) / A
        norm = -np.sum(pi * np.diag(rate))
        rate /= norm
        model = irrev_model_from_rate_matrix(jnp.array(rate), jnp.array(pi))

        padded, C_orig = pad_alignment(alignment, bin_size=8)
        ll_orig = LogLike(alignment, tree, model)
        ll_padded = unpad_columns(LogLike(padded, tree, model), C_orig)
        np.testing.assert_allclose(ll_padded.real, ll_orig.real, atol=1e-10)

    def test_single_column(self):
        """C=1 with large bin_size: almost all columns are padding."""
        alignment, tree, model = self._make_setup(C=1)
        padded, C_orig = pad_alignment(alignment, bin_size=32)
        assert C_orig == 1
        assert padded.shape[1] == 32
        ll_orig = LogLike(alignment, tree, model)
        ll_padded = LogLike(padded, tree, model)
        np.testing.assert_allclose(ll_padded[0], ll_orig[0], atol=1e-12)
        np.testing.assert_allclose(ll_padded[1:], 0.0, atol=1e-12)

    def test_nonuniform_pi_loglike_zero(self):
        """Padded columns logL=0 even with non-uniform pi (HKY85)."""
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(60))
        R = tree.parentIndex.shape[0]
        C = 7
        alignment = jax.random.randint(
            jax.random.PRNGKey(61), (R, C), 0, 4
        ).astype(jnp.int32)
        model = hky85_diag(2.5, jnp.array([0.1, 0.2, 0.3, 0.4]))
        padded, C_orig = pad_alignment(alignment, bin_size=16)
        ll = LogLike(padded, tree, model)
        np.testing.assert_allclose(ll[C_orig:], 0.0, atol=1e-12)

    def test_padded_rootprob_returns_prior(self):
        """All-gap columns should have posterior = prior pi."""
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(70))
        R = tree.parentIndex.shape[0]
        C = 5
        alignment = jax.random.randint(
            jax.random.PRNGKey(71), (R, C), 0, 4
        ).astype(jnp.int32)
        pi = jnp.array([0.1, 0.2, 0.3, 0.4])
        model = hky85_diag(2.0, pi)
        padded, C_orig = pad_alignment(alignment, bin_size=8)
        rp = RootProb(padded, tree, model)  # (A, C_padded)
        # Padded columns: posterior root = prior
        for c in range(C_orig, padded.shape[1]):
            np.testing.assert_allclose(rp[:, c], pi, atol=1e-10)

    def test_padded_counts_zero_with_auto_mask(self):
        """With branch_mask='auto', all-gap columns have empty Steiner tree -> zero counts."""
        alignment, tree, model = self._make_setup(C=10)
        padded, C_orig = pad_alignment(alignment, bin_size=16)
        counts = Counts(padded, tree, model, branch_mask="auto")
        # Padded columns (all gaps) should have all-zero counts
        np.testing.assert_allclose(counts[:, :, C_orig:], 0.0, atol=1e-12)

    def test_f81_fast_counts_round_trip(self):
        """F81 fast path should also be neutral for gap-padded columns."""
        alignment, tree, model = self._make_setup(C=10)
        padded, C_orig = pad_alignment(alignment, bin_size=16)
        counts_orig = Counts(alignment, tree, model, f81_fast_flag=True)
        counts_padded = unpad_columns(
            Counts(padded, tree, model, f81_fast_flag=True), C_orig
        )
        np.testing.assert_allclose(counts_padded, counts_orig, atol=1e-12)

    def test_custom_grad_round_trip(self):
        """LogLikeCustomGrad forward values must also be neutral for padded columns."""
        alignment, tree, model = self._make_setup(C=10)
        padded, C_orig = pad_alignment(alignment, bin_size=16)
        ll_orig = LogLikeCustomGrad(alignment, tree, model)
        ll_padded = LogLikeCustomGrad(padded, tree, model)
        np.testing.assert_allclose(
            unpad_columns(ll_padded, C_orig), ll_orig, atol=1e-12
        )
        np.testing.assert_allclose(ll_padded[C_orig:], 0.0, atol=1e-12)

    def test_bin_size_one_is_noop(self):
        """bin_size=1 should never pad, regardless of C."""
        for C in [1, 7, 13, 128]:
            alignment, tree, model = self._make_setup(C=C)
            padded, C_orig = pad_alignment(alignment, bin_size=1)
            assert C_orig == C
            assert padded.shape[1] == C
            np.testing.assert_array_equal(padded, alignment)

    def test_alignment_with_existing_gaps(self):
        """Padding must not corrupt columns that already contain gap tokens."""
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(80))
        R = tree.parentIndex.shape[0]
        C = 10
        alignment = jax.random.randint(
            jax.random.PRNGKey(81), (R, C), -1, 4
        ).astype(jnp.int32)  # includes -1 tokens
        model = jukes_cantor_model(4)
        padded, C_orig = pad_alignment(alignment, bin_size=16)
        ll_orig = LogLike(alignment, tree, model)
        ll_padded = unpad_columns(LogLike(padded, tree, model), C_orig)
        np.testing.assert_allclose(ll_padded, ll_orig, atol=1e-12)

    def test_dtype_preserved(self):
        """pad_alignment should preserve the input dtype."""
        alignment, tree, model = self._make_setup(C=10)
        assert alignment.dtype == jnp.int32
        padded, _ = pad_alignment(alignment, bin_size=8)
        assert padded.dtype == jnp.int32

    @pytest.mark.parametrize("C,bin_size,expected_padded", [
        (1, 8, 8),
        (8, 8, 8),
        (9, 8, 16),
        (15, 8, 16),
        (1, 128, 128),
        (127, 128, 128),
        (129, 128, 256),
    ])
    def test_padded_shape(self, C, bin_size, expected_padded):
        """Padded C = ceil(C / bin_size) * bin_size."""
        alignment, tree, model = self._make_setup(C=C)
        padded, C_orig = pad_alignment(alignment, bin_size=bin_size)
        assert C_orig == C
        assert padded.shape[1] == expected_padded

    def test_mixture_posterior_round_trip(self):
        """MixturePosterior should also be unaffected by padding."""
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(90))
        R = tree.parentIndex.shape[0]
        C = 10
        alignment = jax.random.randint(
            jax.random.PRNGKey(91), (R, C), 0, 4
        ).astype(jnp.int32)
        base = hky85_diag(2.0, jnp.array([0.25, 0.25, 0.25, 0.25]))
        models = [scale_model(base, r) for r in [0.5, 1.0, 2.0]]
        log_weights = jnp.log(jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))

        padded, C_orig = pad_alignment(alignment, bin_size=16)
        post_orig = MixturePosterior(alignment, tree, models, log_weights)
        post_padded = unpad_columns(
            MixturePosterior(padded, tree, models, log_weights), C_orig
        )
        np.testing.assert_allclose(post_padded, post_orig, atol=1e-10)


class TestInsideOutside:

    def _make_setup(self, C=10, n_leaves=5, seed=42):
        tree = _make_medium_tree(n_leaves, key=jax.random.PRNGKey(seed))
        R = tree.parentIndex.shape[0]
        alignment = jax.random.randint(
            jax.random.PRNGKey(seed + 1), (R, C), 0, 4
        ).astype(jnp.int32)
        model = jukes_cantor_model(4)
        return alignment, tree, model

    def test_loglike_matches(self):
        """InsideOutside.log_likelihood matches standalone LogLike."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        np.testing.assert_allclose(io.log_likelihood, LogLike(alignment, tree, model), atol=1e-12)

    def test_counts_matches(self):
        """InsideOutside.counts() matches standalone Counts."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        np.testing.assert_allclose(io.counts(), Counts(alignment, tree, model), atol=1e-12)

    def test_counts_f81_fast(self):
        """InsideOutside.counts(f81_fast_flag=True) matches standalone."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        np.testing.assert_allclose(
            io.counts(f81_fast_flag=True),
            Counts(alignment, tree, model, f81_fast_flag=True),
            atol=1e-12,
        )

    def test_node_posterior_root_matches_rootprob(self):
        """node_posterior(0) should match RootProb."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        np.testing.assert_allclose(
            io.node_posterior(0),
            RootProb(alignment, tree, model),
            atol=1e-10,
        )

    def test_node_posterior_sums_to_one(self):
        """Node posterior sums to 1 over states for every node and column."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        all_post = io.node_posterior()  # (R, A, C)
        sums = jnp.sum(all_post, axis=-2)  # (R, C)
        np.testing.assert_allclose(sums, 1.0, atol=1e-8)

    def test_node_posterior_nonnegative(self):
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        all_post = io.node_posterior()
        assert jnp.all(all_post >= -1e-10)

    def test_node_posterior_single_matches_all(self):
        """node_posterior(n) matches the n-th slice of node_posterior(None)."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        all_post = io.node_posterior()  # (R, A, C)
        for n in [0, 1, 3, 5]:
            single = io.node_posterior(n)  # (A, C)
            np.testing.assert_allclose(single, all_post[n], atol=1e-12)

    def test_branch_posterior_sums_to_one(self):
        """Branch posterior sums to 1 over (i,j) for every branch and column."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        R = tree.parentIndex.shape[0]
        for n in [1, 3, 5]:
            bp = io.branch_posterior(n)  # (A, A, C)
            sums = jnp.sum(bp, axis=(0, 1))  # (C,)
            np.testing.assert_allclose(sums, 1.0, atol=1e-8)

    def test_branch_posterior_nonnegative(self):
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        bp = io.branch_posterior(1)
        assert jnp.all(bp >= -1e-10)

    def test_branch_marginal_matches_node_posterior(self):
        """Summing branch_posterior over child states gives parent's node_posterior."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        R = tree.parentIndex.shape[0]
        parentIndex = tree.parentIndex

        for n in [1, 3, 5]:
            bp = io.branch_posterior(n)     # (A, A, C)
            parent = int(parentIndex[n])
            # Sum over child states (axis 1) to get parent marginal
            parent_marginal = jnp.sum(bp, axis=1)  # (A, C)
            parent_post = io.node_posterior(parent)  # (A, C)
            np.testing.assert_allclose(parent_marginal, parent_post, atol=1e-8)

    def test_branch_marginal_child_matches_node_posterior(self):
        """Summing branch_posterior over parent states gives child's node_posterior."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)

        for n in [1, 3, 5]:
            bp = io.branch_posterior(n)     # (A, A, C)
            # Sum over parent states (axis 0) to get child marginal
            child_marginal = jnp.sum(bp, axis=0)  # (A, C)
            child_post = io.node_posterior(n)       # (A, C)
            np.testing.assert_allclose(child_marginal, child_post, atol=1e-8)

    def test_branch_posterior_all(self):
        """branch_posterior(None) returns all branches; branch 0 is zeros."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        all_bp = io.branch_posterior()  # (R, A, A, C)
        R = tree.parentIndex.shape[0]
        assert all_bp.shape == (R, 4, 4, 10)
        # Branch 0 is all zeros
        np.testing.assert_allclose(all_bp[0], 0.0, atol=1e-15)
        # Other branches match individual queries
        for n in [1, 3, 5]:
            np.testing.assert_allclose(all_bp[n], io.branch_posterior(n), atol=1e-12)

    def test_irreversible_model(self):
        """InsideOutside works with irreversible model."""
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(100))
        R = tree.parentIndex.shape[0]
        C = 8
        alignment = jax.random.randint(
            jax.random.PRNGKey(101), (R, C), 0, 4
        ).astype(jnp.int32)

        A = 4
        rng = np.random.RandomState(42)
        rate = rng.uniform(0.01, 1.0, (A, A))
        np.fill_diagonal(rate, 0.0)
        np.fill_diagonal(rate, -rate.sum(axis=1))
        pi = np.ones(A) / A
        norm = -np.sum(pi * np.diag(rate))
        rate /= norm
        model = irrev_model_from_rate_matrix(jnp.array(rate), jnp.array(pi))

        io = InsideOutside(alignment, tree, model)
        np.testing.assert_allclose(
            io.log_likelihood.real,
            LogLike(alignment, tree, model).real,
            atol=1e-10,
        )
        # Node posterior should still sum to 1
        post = io.node_posterior(0)  # (A, C)
        sums = jnp.sum(post.real, axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_hky_model(self):
        """InsideOutside works with HKY85 model (non-uniform pi)."""
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(110))
        R = tree.parentIndex.shape[0]
        C = 8
        alignment = jax.random.randint(
            jax.random.PRNGKey(111), (R, C), 0, 4
        ).astype(jnp.int32)
        pi = jnp.array([0.1, 0.2, 0.3, 0.4])
        model = hky85_diag(2.5, pi)
        io = InsideOutside(alignment, tree, model)

        # Log-likelihood matches
        np.testing.assert_allclose(
            io.log_likelihood, LogLike(alignment, tree, model), atol=1e-12
        )
        # Root posterior matches
        np.testing.assert_allclose(
            io.node_posterior(0), RootProb(alignment, tree, model), atol=1e-10
        )
        # Branch posterior marginals consistent
        bp = io.branch_posterior(1)
        child_marginal = jnp.sum(bp, axis=0)
        np.testing.assert_allclose(child_marginal, io.node_posterior(1), atol=1e-8)

    def test_oracle_matches_jax(self):
        """Oracle InsideOutside matches JAX InsideOutside."""
        from subby.oracle import InsideOutside as OracleIO
        from subby.oracle import jukes_cantor_model as oracle_jc

        tree_jax = _make_medium_tree(5, key=jax.random.PRNGKey(120))
        R = tree_jax.parentIndex.shape[0]
        C = 8
        alignment_jax = jax.random.randint(
            jax.random.PRNGKey(121), (R, C), 0, 4
        ).astype(jnp.int32)

        model_jax = jukes_cantor_model(4)
        io_jax = InsideOutside(alignment_jax, tree_jax, model_jax)

        # Convert to oracle format
        alignment_np = np.array(alignment_jax)
        tree_np = {
            'parentIndex': np.array(tree_jax.parentIndex),
            'distanceToParent': np.array(tree_jax.distanceToParent),
        }
        model_np = oracle_jc(4)
        io_oracle = OracleIO(alignment_np, tree_np, model_np)

        # Log-likelihood
        np.testing.assert_allclose(
            np.array(io_jax.log_likelihood), io_oracle.log_likelihood, atol=1e-10
        )
        # Node posteriors
        for n in [0, 1, 3]:
            np.testing.assert_allclose(
                np.array(io_jax.node_posterior(n)),
                io_oracle.node_posterior(n),
                atol=1e-8,
            )
        # Branch posteriors
        for n in [1, 3, 5]:
            np.testing.assert_allclose(
                np.array(io_jax.branch_posterior(n)),
                io_oracle.branch_posterior(n),
                atol=1e-8,
            )
        # Counts
        np.testing.assert_allclose(
            np.array(io_jax.counts()),
            io_oracle.counts(),
            atol=1e-8,
        )

    def test_branch_counts_oracle_matches_jax(self):
        """Oracle InsideOutside.branch_counts matches JAX InsideOutside.branch_counts."""
        from subby.oracle import InsideOutside as OracleIO
        from subby.oracle import jukes_cantor_model as oracle_jc

        tree_jax = _make_medium_tree(5, key=jax.random.PRNGKey(120))
        R = tree_jax.parentIndex.shape[0]
        C = 8
        alignment_jax = jax.random.randint(
            jax.random.PRNGKey(121), (R, C), 0, 4
        ).astype(jnp.int32)

        model_jax = jukes_cantor_model(4)
        io_jax = InsideOutside(alignment_jax, tree_jax, model_jax)

        alignment_np = np.array(alignment_jax)
        tree_np = {
            'parentIndex': np.array(tree_jax.parentIndex),
            'distanceToParent': np.array(tree_jax.distanceToParent),
        }
        model_np = oracle_jc(4)
        io_oracle = OracleIO(alignment_np, tree_np, model_np)

        np.testing.assert_allclose(
            np.array(io_jax.branch_counts()),
            io_oracle.branch_counts(),
            atol=1e-8,
        )


class TestBranchCounts:

    def _make_setup(self, C=10, n_leaves=5, A=4, seed=42):
        tree = _make_medium_tree(n_leaves, key=jax.random.PRNGKey(seed))
        R = tree.parentIndex.shape[0]
        alignment = jax.random.randint(
            jax.random.PRNGKey(seed + 1), (R, C), 0, A
        ).astype(jnp.int32)
        model = jukes_cantor_model(A)
        return alignment, tree, model

    def test_shape(self):
        """BranchCounts returns (R, A, A, C)."""
        alignment, tree, model = self._make_setup()
        bc = BranchCounts(alignment, tree, model)
        R = tree.parentIndex.shape[0]
        assert bc.shape == (R, 4, 4, 10)

    def test_sum_matches_counts(self):
        """Sum of BranchCounts over branches matches Counts."""
        alignment, tree, model = self._make_setup()
        bc = BranchCounts(alignment, tree, model)
        c = Counts(alignment, tree, model)
        np.testing.assert_allclose(bc.sum(axis=-4), c, atol=1e-10)

    def test_branch_zero_is_zeros(self):
        """Branch 0 (root) should be all zeros."""
        alignment, tree, model = self._make_setup()
        bc = BranchCounts(alignment, tree, model)
        np.testing.assert_allclose(bc[0], 0.0, atol=1e-15)

    def test_f81_fast_sum_matches(self):
        """F81 fast path BranchCounts sum matches Counts."""
        alignment, tree, model = self._make_setup()
        bc = BranchCounts(alignment, tree, model, f81_fast_flag=True)
        c = Counts(alignment, tree, model, f81_fast_flag=True)
        np.testing.assert_allclose(bc.sum(axis=-4), c, atol=1e-10)

    def test_eigensub_vs_f81_fast_sum(self):
        """Eigensub and F81 fast BranchCounts sum to the same total."""
        alignment, tree, model = self._make_setup()
        bc_eig = BranchCounts(alignment, tree, model)
        bc_f81 = BranchCounts(alignment, tree, model, f81_fast_flag=True)
        np.testing.assert_allclose(
            bc_eig.sum(axis=-4), bc_f81.sum(axis=-4), atol=1e-6
        )

    def test_nonneg_off_diagonal(self):
        """Off-diagonal entries (substitution counts) should be non-negative."""
        alignment, tree, model = self._make_setup()
        bc = BranchCounts(alignment, tree, model)
        A = 4
        mask = ~jnp.eye(A, dtype=bool)
        off_diag = bc[:, mask]
        assert jnp.all(off_diag >= -1e-10)

    def test_nonneg_diagonal(self):
        """Diagonal entries (dwell times) should be non-negative."""
        alignment, tree, model = self._make_setup()
        bc = BranchCounts(alignment, tree, model)
        A = 4
        for i in range(A):
            assert jnp.all(bc[:, i, i, :] >= -1e-10)

    def test_hky85_model(self):
        """BranchCounts works with non-uniform pi (HKY85)."""
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(200))
        R = tree.parentIndex.shape[0]
        C = 8
        alignment = jax.random.randint(
            jax.random.PRNGKey(201), (R, C), 0, 4
        ).astype(jnp.int32)
        pi = jnp.array([0.1, 0.2, 0.3, 0.4])
        model = hky85_diag(2.5, pi)

        bc = BranchCounts(alignment, tree, model)
        c = Counts(alignment, tree, model)
        np.testing.assert_allclose(bc.sum(axis=-4), c, atol=1e-10)

    def test_irreversible_model(self):
        """BranchCounts works with irreversible model."""
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(300))
        R = tree.parentIndex.shape[0]
        C = 8
        alignment = jax.random.randint(
            jax.random.PRNGKey(301), (R, C), 0, 4
        ).astype(jnp.int32)

        A = 4
        rng = np.random.RandomState(42)
        rate = rng.uniform(0.01, 1.0, (A, A))
        np.fill_diagonal(rate, 0.0)
        np.fill_diagonal(rate, -rate.sum(axis=1))
        pi = np.ones(A) / A
        norm = -np.sum(pi * np.diag(rate))
        rate /= norm
        model = irrev_model_from_rate_matrix(jnp.array(rate), jnp.array(pi))

        bc = BranchCounts(alignment, tree, model)
        c = Counts(alignment, tree, model)
        np.testing.assert_allclose(
            bc.sum(axis=-4).real, c.real, atol=1e-8
        )

    def test_insideoutside_branch_counts_matches_standalone(self):
        """InsideOutside.branch_counts() matches standalone BranchCounts."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        bc_io = io.branch_counts()
        bc_standalone = BranchCounts(alignment, tree, model)
        np.testing.assert_allclose(bc_io, bc_standalone, atol=1e-12)

    def test_insideoutside_branch_counts_f81(self):
        """InsideOutside.branch_counts(f81_fast_flag=True) matches standalone."""
        alignment, tree, model = self._make_setup()
        io = InsideOutside(alignment, tree, model)
        bc_io = io.branch_counts(f81_fast_flag=True)
        bc_standalone = BranchCounts(alignment, tree, model, f81_fast_flag=True)
        np.testing.assert_allclose(bc_io, bc_standalone, atol=1e-12)

    def test_oracle_matches_jax(self):
        """Oracle BranchCounts matches JAX BranchCounts."""
        from subby.oracle import BranchCounts as OracleBranchCounts
        from subby.oracle import jukes_cantor_model as oracle_jc

        tree_jax = _make_medium_tree(5, key=jax.random.PRNGKey(400))
        R = tree_jax.parentIndex.shape[0]
        C = 10
        alignment_jax = jax.random.randint(
            jax.random.PRNGKey(401), (R, C), 0, 4
        ).astype(jnp.int32)
        model_jax = jukes_cantor_model(4)

        bc_jax = np.array(BranchCounts(alignment_jax, tree_jax, model_jax))

        alignment_np = np.array(alignment_jax)
        tree_np = {
            'parentIndex': np.array(tree_jax.parentIndex),
            'distanceToParent': np.array(tree_jax.distanceToParent),
        }
        model_np = oracle_jc(4)
        bc_oracle = OracleBranchCounts(alignment_np, tree_np, model_np)

        np.testing.assert_allclose(bc_jax, bc_oracle, atol=1e-8)

    def test_oracle_irrev_matches_jax(self):
        """Oracle irreversible BranchCounts matches JAX."""
        from subby.oracle import BranchCounts as OracleBranchCounts
        from subby.oracle import irrev_model_from_rate_matrix as oracle_irrev

        tree_jax = _make_medium_tree(5, key=jax.random.PRNGKey(500))
        R = tree_jax.parentIndex.shape[0]
        C = 8
        alignment_jax = jax.random.randint(
            jax.random.PRNGKey(501), (R, C), 0, 4
        ).astype(jnp.int32)

        A = 4
        rng = np.random.RandomState(99)
        rate = rng.uniform(0.01, 1.0, (A, A))
        np.fill_diagonal(rate, 0.0)
        np.fill_diagonal(rate, -rate.sum(axis=1))
        pi = np.ones(A) / A
        norm = -np.sum(pi * np.diag(rate))
        rate /= norm

        model_jax = irrev_model_from_rate_matrix(jnp.array(rate), jnp.array(pi))
        bc_jax = np.array(BranchCounts(alignment_jax, tree_jax, model_jax))

        alignment_np = np.array(alignment_jax)
        tree_np = {
            'parentIndex': np.array(tree_jax.parentIndex),
            'distanceToParent': np.array(tree_jax.distanceToParent),
        }
        model_np = oracle_irrev(rate, pi)
        bc_oracle = OracleBranchCounts(alignment_np, tree_np, model_np)

        np.testing.assert_allclose(bc_jax.real, bc_oracle.real, atol=1e-6)

    def test_large_alphabet(self):
        """BranchCounts works with A=20 (protein-sized)."""
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(600))
        R = tree.parentIndex.shape[0]
        C = 5
        alignment = jax.random.randint(
            jax.random.PRNGKey(601), (R, C), 0, 20
        ).astype(jnp.int32)
        model = jukes_cantor_model(20)

        bc = BranchCounts(alignment, tree, model)
        c = Counts(alignment, tree, model)
        assert bc.shape == (R, 20, 20, C)
        np.testing.assert_allclose(bc.sum(axis=-4), c, atol=1e-8)

    def test_per_column_model(self):
        """BranchCounts with per-column models."""
        tree = _make_medium_tree(5, key=jax.random.PRNGKey(700))
        R = tree.parentIndex.shape[0]
        C = 4
        alignment = jax.random.randint(
            jax.random.PRNGKey(701), (R, C), 0, 4
        ).astype(jnp.int32)

        models = [jukes_cantor_model(4) for _ in range(C)]
        bc = BranchCounts(alignment, tree, models)
        c = Counts(alignment, tree, models)
        assert bc.shape == (R, 4, 4, C)
        np.testing.assert_allclose(bc.sum(axis=-4), c, atol=1e-10)
