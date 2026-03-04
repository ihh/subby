"""Tests for HKY85 closed-form eigendecomposition."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax.numpy as jnp
import numpy as np
import pytest


from subby.jax.models import hky85_diag, jukes_cantor_model, f81_model
from subby.jax.diagonalize import reconstruct_rate_matrix, compute_sub_matrices
from subby.jax.types import Tree


class TestHKY85:

    def test_eigendecomp_reconstruct_roundtrip(self):
        """Diagonalize then reconstruct should give original rate matrix."""
        pi = jnp.array([0.3, 0.2, 0.25, 0.25])
        kappa = 2.0
        model = hky85_diag(kappa, pi)
        reconstructed = reconstruct_rate_matrix(model)

        # Build expected HKY rate matrix
        is_transition = jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=jnp.float64)
        R_expected = pi[None, :] * (1.0 + (kappa - 1.0) * is_transition)
        R_expected = R_expected - jnp.diag(jnp.sum(R_expected, axis=-1))
        expected_rate = -jnp.sum(pi * jnp.diag(R_expected))
        R_expected = R_expected / expected_rate

        np.testing.assert_allclose(reconstructed.subRate, R_expected, atol=1e-10)

    def test_eigenvectors_orthonormal(self):
        """Eigenvectors should be orthonormal."""
        pi = jnp.array([0.3, 0.2, 0.25, 0.25])
        model = hky85_diag(2.0, pi)
        V = model.eigenvectors  # (4, 4)
        VtV = V.T @ V
        np.testing.assert_allclose(VtV, jnp.eye(4), atol=1e-10)

    def test_eigenvalues_nonpositive(self):
        """All eigenvalues should be <= 0."""
        pi = jnp.array([0.1, 0.4, 0.2, 0.3])
        model = hky85_diag(3.0, pi)
        assert jnp.all(model.eigenvalues <= 1e-10)

    def test_zero_eigenvalue_exists(self):
        """One eigenvalue should be exactly 0 (stationary)."""
        model = hky85_diag(2.0, jnp.array([0.25, 0.25, 0.25, 0.25]))
        assert jnp.min(jnp.abs(model.eigenvalues)) < 1e-10

    def test_sub_matrix_rows_sum_to_one(self):
        """M(t) rows should sum to 1."""
        pi = jnp.array([0.3, 0.2, 0.25, 0.25])
        model = hky85_diag(2.0, pi)
        distances = jnp.array([0.0, 0.1, 0.5, 1.0, 5.0])
        M = compute_sub_matrices(model, distances)  # (5, 4, 4)
        row_sums = jnp.sum(M, axis=-1)  # (5, 4)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_sub_matrix_at_zero_is_identity(self):
        """M(0) should be identity."""
        model = hky85_diag(2.0, jnp.array([0.3, 0.2, 0.25, 0.25]))
        M = compute_sub_matrices(model, jnp.array([0.0]))
        np.testing.assert_allclose(M[0], jnp.eye(4), atol=1e-10)

    def test_sub_matrix_at_infinity_is_pi(self):
        """M(large t) rows should approach pi."""
        pi = jnp.array([0.3, 0.2, 0.25, 0.25])
        model = hky85_diag(2.0, pi)
        M = compute_sub_matrices(model, jnp.array([100.0]))
        for i in range(4):
            np.testing.assert_allclose(M[0, i, :], pi, atol=1e-6)

    def test_hky_reduces_to_jc_when_uniform(self):
        """With uniform pi and kappa=1, HKY85 should give JC-like behavior."""
        pi = jnp.array([0.25, 0.25, 0.25, 0.25])
        model_hky = hky85_diag(1.0, pi)
        model_jc = jukes_cantor_model(4)

        # Compare substitution matrices at t=0.3
        M_hky = compute_sub_matrices(model_hky, jnp.array([0.3]))
        M_jc = compute_sub_matrices(model_jc, jnp.array([0.3]))
        np.testing.assert_allclose(M_hky[0], M_jc[0], atol=1e-6)


class TestJC:

    def test_jc_eigenvalues(self):
        model = jukes_cantor_model(4)
        assert model.eigenvalues[0] == pytest.approx(0.0, abs=1e-10)
        expected_neg = -4.0 / 3.0
        for k in range(1, 4):
            assert model.eigenvalues[k] == pytest.approx(expected_neg, abs=1e-10)

    def test_jc_eigenvectors_orthonormal(self):
        model = jukes_cantor_model(4)
        VtV = model.eigenvectors.T @ model.eigenvectors
        np.testing.assert_allclose(VtV, jnp.eye(4), atol=1e-10)


class TestF81:

    def test_f81_eigenvalues(self):
        pi = jnp.array([0.3, 0.2, 0.25, 0.25])
        model = f81_model(pi)
        mu = 1.0 / (1.0 - jnp.sum(pi ** 2))
        assert model.eigenvalues[0] == pytest.approx(0.0, abs=1e-10)
        for k in range(1, 4):
            assert model.eigenvalues[k] == pytest.approx(-mu, abs=1e-10)

    def test_f81_sub_matrix(self):
        """M_ij(t) = delta_ij * e^{-mu*t} + pi_j * (1 - e^{-mu*t})."""
        pi = jnp.array([0.3, 0.2, 0.25, 0.25])
        model = f81_model(pi)
        t = 0.5
        M = compute_sub_matrices(model, jnp.array([t]))  # (1, 4, 4)
        mu = 1.0 / (1.0 - jnp.sum(pi ** 2))
        e_mt = jnp.exp(-mu * t)
        expected = jnp.eye(4) * e_mt + pi[None, :] * (1.0 - e_mt)
        np.testing.assert_allclose(M[0], expected, atol=1e-6)
