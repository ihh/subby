"""Tests for the upward (pruning) pass against the toy implementation."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.phylo.jax.types import Tree, DiagModel
from src.phylo.jax.diagonalize import compute_sub_matrices
from src.phylo.jax.pruning import upward_pass
from src.phylo.jax.models import hky85_diag, jukes_cantor_model

# Import the toy implementation for reference comparison
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import toy_felsenstein_pruning as toy


def _make_simple_tree():
    """3-leaf binary tree:
          0 (root)
         / \\
        1   2
       / \\
      3   4
    Leaves: 2, 3, 4
    """
    parentIndex = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
    distanceToParent = jnp.array([0.0, 0.1, 0.2, 0.15, 0.25])
    return Tree(parentIndex=parentIndex, distanceToParent=distanceToParent)


def _make_random_alignment(R, C, A=4, key=None):
    if key is None:
        key = jax.random.PRNGKey(42)
    # Mix of observed tokens and gaps
    tokens = jax.random.randint(key, (R, C), -1, A)
    return tokens.astype(jnp.int32)


class TestUpwardPassVsToy:
    """Verify new pruning matches toy_felsenstein_pruning.subLogLike exactly."""

    def test_loglike_matches_toy_jc(self):
        """JC model on small tree, log-likelihoods must match."""
        tree = _make_simple_tree()
        R = 5
        C = 10
        A = 4
        alignment = _make_random_alignment(R, C, A)

        # Build JC rate matrix for toy
        subRate = jnp.ones((A, A)) / (A - 1)
        subRate = subRate - jnp.diag(jnp.sum(subRate, axis=-1))
        rootProb = jnp.ones(A) / A

        # Toy implementation
        toy_ll = toy.subLogLike(alignment, tree.distanceToParent, tree.parentIndex, subRate, rootProb)

        # New implementation
        model = jukes_cantor_model(A)
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)
        _, _, new_ll = upward_pass(alignment, tree, subMatrices, model.pi)

        np.testing.assert_allclose(new_ll, toy_ll, atol=1e-6, rtol=1e-5)

    def test_loglike_matches_toy_hky(self):
        """HKY85 model on small tree."""
        tree = _make_simple_tree()
        R = 5
        C = 8
        A = 4
        alignment = _make_random_alignment(R, C, A)

        pi = jnp.array([0.3, 0.2, 0.25, 0.25])
        kappa = 2.0

        # Build HKY rate matrix for toy
        # R_ij for HKY85: rate_ij = pi_j for transversions, kappa*pi_j for transitions
        is_transition = jnp.array([
            [0, 0, 1, 0],  # A->G
            [0, 0, 0, 1],  # C->T
            [1, 0, 0, 0],  # G->A
            [0, 1, 0, 0],  # T->C
        ], dtype=jnp.float64)
        subRate = pi[None, :] * (1.0 + (kappa - 1.0) * is_transition)
        subRate = subRate - jnp.diag(jnp.sum(subRate, axis=-1))
        # Normalize to expected rate 1
        expected_rate = -jnp.sum(pi * jnp.diag(subRate))
        subRate = subRate / expected_rate

        toy_ll = toy.subLogLike(alignment, tree.distanceToParent, tree.parentIndex, subRate, pi)

        model = hky85_diag(kappa, pi)
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)
        _, _, new_ll = upward_pass(alignment, tree, subMatrices, model.pi)

        np.testing.assert_allclose(new_ll, toy_ll, atol=1e-5, rtol=1e-4)

    def test_all_gaps_zero_loglike(self):
        """All-gap columns should have loglike = log(1) = 0."""
        tree = _make_simple_tree()
        R = 5
        C = 3
        alignment = jnp.full((R, C), -1, dtype=jnp.int32)

        model = jukes_cantor_model(4)
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)
        _, _, ll = upward_pass(alignment, tree, subMatrices, model.pi)

        np.testing.assert_allclose(ll, 0.0, atol=1e-10)

    def test_per_node_U_shapes(self):
        """Check U and logNormU shapes."""
        tree = _make_simple_tree()
        R = 5
        C = 4
        alignment = _make_random_alignment(R, C)

        model = jukes_cantor_model(4)
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)
        U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi)

        assert U.shape == (R, C, 4)
        assert logNormU.shape == (R, C)
        assert logLike.shape == (C,)

    def test_chunking_consistent(self):
        """Chunked and unchunked should produce same results."""
        tree = _make_simple_tree()
        R = 5
        C = 20
        alignment = _make_random_alignment(R, C)

        model = jukes_cantor_model(4)
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)

        U1, lnU1, ll1 = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize=999)
        U2, lnU2, ll2 = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize=5)

        np.testing.assert_allclose(ll1, ll2, atol=1e-10)
