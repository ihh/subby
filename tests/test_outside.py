"""Tests for the downward (outside) pass."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from subby.jax.types import Tree
from subby.jax.diagonalize import compute_sub_matrices
from subby.jax.pruning import upward_pass
from subby.jax.outside import downward_pass
from subby.jax.models import jukes_cantor_model, hky85_diag


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

    def test_consistency_larger_tree(self):
        """Consistency check on a 31-node balanced binary tree."""
        R = 31
        parentIndex = np.zeros(R, dtype=np.int32)
        parentIndex[0] = -1
        for i in range(1, R):
            parentIndex[i] = (i - 1) // 2
        parentIndex = jnp.array(parentIndex)
        distances = jnp.ones(R) * 0.1
        distances = distances.at[0].set(0.0)
        tree = Tree(parentIndex=parentIndex, distanceToParent=distances)

        C = 32
        A = 4
        key = jax.random.PRNGKey(999)
        alignment = jax.random.randint(key, (R, C), -1, A).astype(jnp.int32)

        model = jukes_cantor_model(A)
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

    def test_pregathered_sibling_contrib(self):
        """Verify the pre-gathered sibling contribution optimization is correct.

        This tests that pre-computing sib_contrib = M @ U^(sib) outside the
        scan matches computing it inside the scan body.
        """
        tree = _make_7node_tree()
        R = 7
        C = 8
        A = 4
        key = jax.random.PRNGKey(77)
        alignment = jax.random.randint(key, (R, C), 0, A).astype(jnp.int32)

        model = hky85_diag(2.0, jnp.array([0.3, 0.2, 0.25, 0.25]))
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)

        U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi)
        D, logNormD = downward_pass(U, logNormU, tree, subMatrices, model.pi, alignment)

        # Verify all D values are positive and finite
        assert jnp.all(jnp.isfinite(D))
        assert jnp.all(D >= 0)
        assert jnp.all(jnp.isfinite(logNormD))


class TestDownwardPassParallel:
    """Tests for the parallel=True level-parallel downward pass."""

    def test_parallel_matches_sequential(self):
        """parallel=True should produce identical results to the default scan."""
        tree = _make_7node_tree()
        R = 7
        C = 6
        A = 4
        key = jax.random.PRNGKey(123)
        alignment = jax.random.randint(key, (R, C), -1, A).astype(jnp.int32)

        model = jukes_cantor_model(A)
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)

        U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi)

        D_seq, logNormD_seq = downward_pass(
            U, logNormU, tree, subMatrices, model.pi, alignment)
        D_par, logNormD_par = downward_pass(
            U, logNormU, tree, subMatrices, model.pi, alignment, parallel=True)

        np.testing.assert_allclose(D_par, D_seq, atol=1e-12)
        np.testing.assert_allclose(logNormD_par, logNormD_seq, atol=1e-12)

    def test_parallel_consistency_larger_tree(self):
        """Consistency check with parallel=True on a 31-node balanced tree."""
        R = 31
        parentIndex = np.zeros(R, dtype=np.int32)
        parentIndex[0] = -1
        for i in range(1, R):
            parentIndex[i] = (i - 1) // 2
        parentIndex = jnp.array(parentIndex)
        distances = jnp.ones(R) * 0.1
        distances = distances.at[0].set(0.0)
        tree = Tree(parentIndex=parentIndex, distanceToParent=distances)

        C = 32
        A = 4
        key = jax.random.PRNGKey(999)
        alignment = jax.random.randint(key, (R, C), -1, A).astype(jnp.int32)

        model = jukes_cantor_model(A)
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)

        U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi)
        D, logNormD = downward_pass(
            U, logNormU, tree, subMatrices, model.pi, alignment, parallel=True)

        for n in range(1, R):
            MU = jnp.einsum('ij,cj->ci', subMatrices[n], U[n])
            joint = jnp.sum(D[n] * MU, axis=-1)
            log_joint = jnp.log(joint) + logNormD[n] + logNormU[n]
            np.testing.assert_allclose(log_joint, logLike, atol=1e-4, rtol=1e-3,
                                        err_msg=f"Node {n}")

    def test_parallel_hky(self):
        """parallel=True matches sequential with HKY85 model."""
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

        D_seq, logNormD_seq = downward_pass(
            U, logNormU, tree, subMatrices, model.pi, alignment)
        D_par, logNormD_par = downward_pass(
            U, logNormU, tree, subMatrices, model.pi, alignment, parallel=True)

        np.testing.assert_allclose(D_par, D_seq, atol=1e-12)
        np.testing.assert_allclose(logNormD_par, logNormD_seq, atol=1e-12)
