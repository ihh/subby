"""Tests for eigensubstitution accumulation against brute-force."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.phylo.jax.types import Tree
from src.phylo.jax.diagonalize import compute_sub_matrices
from src.phylo.jax.pruning import upward_pass
from src.phylo.jax.outside import downward_pass
from src.phylo.jax.eigensub import compute_J, eigenbasis_project, accumulate_C, back_transform
from src.phylo.jax.models import jukes_cantor_model, hky85_diag
from src.phylo.jax import Counts


def _brute_force_counts_single_branch(t, subRate, rootProb):
    """Brute-force expected counts for a 2-node tree (root->leaf).

    For each column, the joint posterior is:
    q_{ab} = pi_a * M_ab * delta(b, observed) / P(x)

    Expected counts from the I integral:
    I^{ab}_{ij}(T) = integral M_{ai}(s) M_{jb}(T-s) ds

    We compute this numerically.
    """
    from jax.scipy.linalg import expm
    A = subRate.shape[0]
    M_T = expm(subRate * t)  # (A, A)

    # Numerical integration of I^{ab}_{ij}(T) via quadrature
    n_steps = 1000
    ds = t / n_steps
    I = jnp.zeros((A, A, A, A))  # I[a,b,i,j]
    for step in range(n_steps):
        s = (step + 0.5) * ds
        M_s = expm(subRate * s)
        M_ts = expm(subRate * (t - s))
        # I^{ab}_{ij} += M_{ai}(s) * M_{jb}(T-s) * ds
        I += jnp.einsum('ai,jb->abij', M_s, M_ts) * ds

    return M_T, I


class TestEigensubstitution:

    def test_J_degenerate(self):
        """J^{kk}(T) = T * exp(mu_k * T) for degenerate eigenvalues."""
        mu = jnp.array([0.0, -1.0, -1.0, -2.0])
        t = jnp.array([0.5])
        J = compute_J(mu, t)  # (1, 4, 4)
        # Check diagonal
        for k in range(4):
            expected = 0.5 * jnp.exp(mu[k] * 0.5)
            np.testing.assert_allclose(J[0, k, k], expected, atol=1e-10)
        # Check degenerate pair (1,2): same eigenvalue -1
        expected_12 = 0.5 * jnp.exp(-1.0 * 0.5)  # T * exp(mu*T) since mu_1 = mu_2
        np.testing.assert_allclose(J[0, 1, 2], expected_12, atol=1e-6)

    def test_J_nondegenerate(self):
        """J^{kl}(T) = (exp(mu_k*T) - exp(mu_l*T)) / (mu_k - mu_l) for k != l."""
        mu = jnp.array([0.0, -1.0, -2.0, -3.0])
        t = jnp.array([0.5])
        J = compute_J(mu, t)
        # Check (0,1): (exp(0) - exp(-0.5)) / (0 - (-1)) = (1 - exp(-0.5)) / 1
        expected = (1.0 - jnp.exp(-0.5)) / 1.0
        np.testing.assert_allclose(J[0, 0, 1], expected, atol=1e-10)

    def test_counts_3node_tree(self):
        """Compare eigensub counts to brute-force on a 3-node binary tree.

        Uses root(gap) -> leaf1(observed), root -> leaf2(gap, zero-length branch).
        The zero-length gapped branch contributes zero counts, so only branch 1 matters.
        """
        A = 4
        t = 0.3

        # 3-node binary tree: root=0, leaf1=1 (observed), leaf2=2 (gap, zero branch)
        parentIndex = jnp.array([-1, 0, 0], dtype=jnp.int32)
        distanceToParent = jnp.array([0.0, t, 1e-10])
        tree = Tree(parentIndex=parentIndex, distanceToParent=distanceToParent)

        # leaf1 = state 0 (observed), root and leaf2 = gap (unobserved)
        alignment = jnp.array([[-1], [0], [-1]], dtype=jnp.int32)

        model = jukes_cantor_model(A)
        counts = Counts(alignment, tree, model)  # (A, A, 1)

        # Brute-force for single branch of length t
        subRate = jnp.ones((A, A)) / (A - 1)
        subRate = subRate - jnp.diag(jnp.sum(subRate, axis=-1))
        pi = jnp.ones(A) / A
        M_T, I = _brute_force_counts_single_branch(t, subRate, pi)

        # For root(gap)->leaf(0): q_{a,0} = pi_a * M_{a,0} / P(x)
        Px = jnp.sum(pi * M_T[:, 0])

        expected_dwell = jnp.zeros(A)
        expected_subs = jnp.zeros((A, A))
        for a in range(A):
            b = 0
            weight = pi[a] / Px
            for i in range(A):
                expected_dwell = expected_dwell.at[i].add(weight * I[a, b, i, i])
                for j in range(A):
                    if i != j:
                        expected_subs = expected_subs.at[i, j].add(
                            weight * subRate[i, j] * I[a, b, i, j]
                        )

        for i in range(A):
            np.testing.assert_allclose(counts[i, i, 0], expected_dwell[i], atol=1e-3, rtol=1e-2)
            for j in range(A):
                if i != j:
                    np.testing.assert_allclose(counts[i, j, 0], expected_subs[i, j], atol=1e-3, rtol=1e-2)

    def test_counts_nonnegative(self):
        """All count entries should be non-negative."""
        tree = Tree(
            parentIndex=jnp.array([-1, 0, 0, 1, 1, 2, 2], dtype=jnp.int32),
            distanceToParent=jnp.array([0.0, 0.1, 0.2, 0.15, 0.25, 0.12, 0.18]),
        )
        alignment = jax.random.randint(jax.random.PRNGKey(99), (7, 10), -1, 4).astype(jnp.int32)
        model = jukes_cantor_model(4)
        counts = Counts(alignment, tree, model)
        assert jnp.all(counts >= -1e-6), f"Negative counts found: min={jnp.min(counts)}"
