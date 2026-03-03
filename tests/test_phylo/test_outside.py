"""Tests for the downward (outside) pass."""
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
from src.phylo.jax.models import jukes_cantor_model, hky85_diag


def _make_simple_tree():
    parentIndex = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
    distanceToParent = jnp.array([0.0, 0.1, 0.2, 0.15, 0.25])
    return Tree(parentIndex=parentIndex, distanceToParent=distanceToParent)


def _make_7node_tree():
    """7-node tree:
          0
         / \\
        1   2
       / \\ / \\
      3  4 5  6
    """
    parentIndex = jnp.array([-1, 0, 0, 1, 1, 2, 2], dtype=jnp.int32)
    distanceToParent = jnp.array([0.0, 0.1, 0.2, 0.15, 0.25, 0.12, 0.18])
    return Tree(parentIndex=parentIndex, distanceToParent=distanceToParent)


class TestDownwardPass:

    def test_consistency_DM_U_equals_Px(self):
        """For each non-root node n, sum_{a,b} D_a * M_ab * U_b should equal P(x|theta).

        This is the fundamental consistency check for the sum-product algorithm.
        """
        tree = _make_7node_tree()
        R = 7
        C = 6
        A = 4
        key = jax.random.PRNGKey(123)
        alignment = jax.random.randint(key, (R, C), -1, A).astype(jnp.int32)

        model = jukes_cantor_model(A)
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)

        U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi)
        D, logNormD = downward_pass(U, logNormU, tree, subMatrices, model.pi, alignment)

        # For each non-root branch n:
        # sum_{a,b} D_true_a * M_ab * U_true_b = P(x)
        # = sum_{a,b} D_rescaled_a * M_ab * U_rescaled_b * exp(logNormD[n] + logNormU[n])
        for n in range(1, R):
            D_n = D[n]  # (C, A)
            U_n = U[n]  # (C, A)
            M_n = subMatrices[n]  # (A, A)

            # sum_{a,b} D_a M_ab U_b = sum_a D_a * (M @ U)_a
            MU = jnp.einsum('ij,cj->ci', M_n, U_n)  # (C, A)
            joint = jnp.sum(D_n * MU, axis=-1)  # (C,)

            log_joint = jnp.log(joint) + logNormD[n] + logNormU[n]

            np.testing.assert_allclose(log_joint, logLike, atol=1e-5, rtol=1e-4,
                                        err_msg=f"Node {n} inconsistency")

    def test_D_shapes(self):
        tree = _make_simple_tree()
        R = 5
        C = 4
        A = 4
        alignment = jax.random.randint(jax.random.PRNGKey(0), (R, C), -1, A).astype(jnp.int32)

        model = jukes_cantor_model(A)
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)
        U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi)
        D, logNormD = downward_pass(U, logNormU, tree, subMatrices, model.pi, alignment)

        assert D.shape == (R, C, A)
        assert logNormD.shape == (R, C)

    def test_consistency_hky(self):
        """Same consistency check with HKY85."""
        tree = _make_7node_tree()
        R = 7
        C = 5
        A = 4
        pi = jnp.array([0.3, 0.2, 0.25, 0.25])
        key = jax.random.PRNGKey(42)
        alignment = jax.random.randint(key, (R, C), 0, A).astype(jnp.int32)

        model = hky85_diag(2.0, pi)
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)

        U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi)
        D, logNormD = downward_pass(U, logNormU, tree, subMatrices, model.pi, alignment)

        for n in range(1, R):
            D_n = D[n]
            U_n = U[n]
            M_n = subMatrices[n]
            MU = jnp.einsum('ij,cj->ci', M_n, U_n)
            joint = jnp.sum(D_n * MU, axis=-1)
            log_joint = jnp.log(joint) + logNormD[n] + logNormU[n]
            np.testing.assert_allclose(log_joint, logLike, atol=1e-4, rtol=1e-3,
                                        err_msg=f"Node {n}")
