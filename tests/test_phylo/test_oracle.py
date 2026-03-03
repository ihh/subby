"""Oracle vs JAX comparison tests.

Validates the imperative Python oracle against the JAX implementation
on identical inputs at atol=1e-8.
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.phylo.jax import LogLike as jax_LogLike
from src.phylo.jax import Counts as jax_Counts
from src.phylo.jax import RootProb as jax_RootProb
from src.phylo.jax import MixturePosterior as jax_MixturePosterior
from src.phylo.jax.types import Tree as JaxTree
from src.phylo.jax.models import (
    hky85_diag as jax_hky85_diag,
    jukes_cantor_model as jax_jukes_cantor_model,
    f81_model as jax_f81_model,
    gamma_rate_categories as jax_gamma_rate_categories,
    scale_model as jax_scale_model,
)
from src.phylo.jax.diagonalize import compute_sub_matrices as jax_compute_sub_matrices
from src.phylo.jax.pruning import upward_pass as jax_upward_pass
from src.phylo.jax.outside import downward_pass as jax_downward_pass
from src.phylo.jax.eigensub import (
    compute_J as jax_compute_J,
    eigenbasis_project as jax_eigenbasis_project,
    accumulate_C as jax_accumulate_C,
    back_transform as jax_back_transform,
)
from src.phylo.jax._utils import (
    token_to_likelihood as jax_token_to_likelihood,
    children_of as jax_children_of,
)
from src.phylo.jax.f81_fast import f81_counts as jax_f81_counts
from src.phylo.jax.mixture import mixture_posterior as jax_mixture_posterior

import src.phylo.oracle as oracle


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_tree(R, seed=0):
    """Build a balanced binary tree with R nodes."""
    rng = np.random.RandomState(seed)
    parentIndex = np.zeros(R, dtype=np.int32)
    parentIndex[0] = -1
    for i in range(1, R):
        parentIndex[i] = (i - 1) // 2
    distances = rng.uniform(0.01, 0.5, size=R).astype(np.float64)
    distances[0] = 0.0
    return parentIndex, distances


def _make_alignment(R, C, A, seed=1):
    """Random alignment with tokens in {0..A-1} and occasional gaps (-1)."""
    rng = np.random.RandomState(seed)
    alignment = rng.randint(0, A + 2, size=(R, C)).astype(np.int32)
    # Remap: A+1 -> -1 (gap), keep 0..A as-is
    alignment[alignment == A + 1] = -1
    return alignment


def _jax_tree(parentIndex, distances):
    return JaxTree(
        parentIndex=jnp.array(parentIndex),
        distanceToParent=jnp.array(distances),
    )


def _oracle_tree(parentIndex, distances):
    return {
        'parentIndex': parentIndex,
        'distanceToParent': distances,
    }


def _oracle_model_from_jax(jax_model):
    """Convert JAX DiagModel to oracle model dict."""
    return {
        'eigenvalues': np.array(jax_model.eigenvalues),
        'eigenvectors': np.array(jax_model.eigenvectors),
        'pi': np.array(jax_model.pi),
    }


# ---------------------------------------------------------------------------
# Tree sizes for parametrization
# ---------------------------------------------------------------------------

TREE_SIZES = [3, 5, 7, 19]


def _n_leaves_from_R(R):
    return (R + 1) // 2


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

def _make_jc4():
    return jax_jukes_cantor_model(4)


def _make_jc64():
    return jax_jukes_cantor_model(64)


def _make_f81():
    pi = jnp.array([0.35, 0.15, 0.25, 0.25])
    return jax_f81_model(pi)


def _make_hky85():
    return jax_hky85_diag(2.0, jnp.array([0.3, 0.2, 0.25, 0.25]))


# Parametrize as (model_name, model_factory, A)
MODEL_CONFIGS = [
    ('JC4', _make_jc4, 4),
    ('F81', _make_f81, 4),
    ('HKY85', _make_hky85, 4),
    ('JC64', _make_jc64, 64),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTokenToLikelihood:

    @pytest.mark.parametrize('A', [4, 64])
    def test_token_to_likelihood(self, A):
        R, C = 5, 10
        alignment = _make_alignment(R, C, A, seed=42)
        jax_L = np.array(jax_token_to_likelihood(jnp.array(alignment), A))
        oracle_L = oracle.token_to_likelihood(alignment, A)
        np.testing.assert_allclose(oracle_L, jax_L, atol=1e-12)


class TestChildrenOf:

    @pytest.mark.parametrize('R', TREE_SIZES)
    def test_children_of(self, R):
        parentIndex, _ = _make_tree(R)
        jax_lc, jax_rc, jax_sib = jax_children_of(jnp.array(parentIndex))
        ora_lc, ora_rc, ora_sib = oracle.children_of(parentIndex)
        np.testing.assert_array_equal(ora_lc, np.array(jax_lc))
        np.testing.assert_array_equal(ora_rc, np.array(jax_rc))
        np.testing.assert_array_equal(ora_sib, np.array(jax_sib))


class TestSubMatrices:

    @pytest.mark.parametrize('model_name,model_fn,A', MODEL_CONFIGS)
    def test_compute_sub_matrices(self, model_name, model_fn, A):
        R = 7
        parentIndex, distances = _make_tree(R, seed=10)
        jax_model = model_fn()
        ora_model = _oracle_model_from_jax(jax_model)

        jax_M = np.array(jax_compute_sub_matrices(jax_model, jnp.array(distances)))
        ora_M = oracle.compute_sub_matrices(ora_model, distances)
        np.testing.assert_allclose(ora_M, jax_M, atol=1e-8,
                                   err_msg=f"SubMatrices mismatch for {model_name}")


class TestUpwardPass:

    @pytest.mark.parametrize('R', TREE_SIZES)
    @pytest.mark.parametrize('model_name,model_fn,A',
                             [c for c in MODEL_CONFIGS if c[2] == 4])
    def test_upward_pass(self, R, model_name, model_fn, A):
        C = 8
        parentIndex, distances = _make_tree(R, seed=20)
        alignment = _make_alignment(R, C, A, seed=21)

        jax_model = model_fn()
        ora_model = _oracle_model_from_jax(jax_model)

        jax_tree = _jax_tree(parentIndex, distances)
        ora_tree = _oracle_tree(parentIndex, distances)

        jax_subM = jax_compute_sub_matrices(jax_model, jnp.array(distances))
        ora_subM = oracle.compute_sub_matrices(ora_model, distances)

        jax_U, jax_lnU, jax_ll = jax_upward_pass(
            jnp.array(alignment), jax_tree, jax_subM, jax_model.pi
        )
        ora_U, ora_lnU, ora_ll = oracle.upward_pass(
            alignment, ora_tree, ora_subM, ora_model['pi']
        )

        np.testing.assert_allclose(ora_ll, np.array(jax_ll), atol=1e-8,
                                   err_msg=f"logLike mismatch for R={R}, {model_name}")
        np.testing.assert_allclose(ora_lnU, np.array(jax_lnU), atol=1e-8,
                                   err_msg=f"logNormU mismatch for R={R}, {model_name}")
        np.testing.assert_allclose(ora_U, np.array(jax_U), atol=1e-8,
                                   err_msg=f"U mismatch for R={R}, {model_name}")


class TestDownwardPass:

    @pytest.mark.parametrize('R', TREE_SIZES)
    @pytest.mark.parametrize('model_name,model_fn,A',
                             [c for c in MODEL_CONFIGS if c[2] == 4])
    def test_downward_pass(self, R, model_name, model_fn, A):
        C = 8
        parentIndex, distances = _make_tree(R, seed=30)
        alignment = _make_alignment(R, C, A, seed=31)

        jax_model = model_fn()
        ora_model = _oracle_model_from_jax(jax_model)

        jax_tree = _jax_tree(parentIndex, distances)
        ora_tree = _oracle_tree(parentIndex, distances)

        jax_subM = jax_compute_sub_matrices(jax_model, jnp.array(distances))
        ora_subM = oracle.compute_sub_matrices(ora_model, distances)

        jax_U, jax_lnU, jax_ll = jax_upward_pass(
            jnp.array(alignment), jax_tree, jax_subM, jax_model.pi
        )
        ora_U, ora_lnU, ora_ll = oracle.upward_pass(
            alignment, ora_tree, ora_subM, ora_model['pi']
        )

        jax_D, jax_lnD = jax_downward_pass(
            jax_U, jax_lnU, jax_tree, jax_subM, jax_model.pi, jnp.array(alignment)
        )
        ora_D, ora_lnD = oracle.downward_pass(
            ora_U, ora_lnU, ora_tree, ora_subM, ora_model['pi'], alignment
        )

        np.testing.assert_allclose(ora_D, np.array(jax_D), atol=1e-8,
                                   err_msg=f"D mismatch for R={R}, {model_name}")
        np.testing.assert_allclose(ora_lnD, np.array(jax_lnD), atol=1e-8,
                                   err_msg=f"logNormD mismatch for R={R}, {model_name}")


class TestComputeJ:

    @pytest.mark.parametrize('model_name,model_fn,A', MODEL_CONFIGS)
    def test_compute_J(self, model_name, model_fn, A):
        R = 7
        _, distances = _make_tree(R, seed=40)
        jax_model = model_fn()
        ora_model = _oracle_model_from_jax(jax_model)

        jax_J = np.array(jax_compute_J(jax_model.eigenvalues, jnp.array(distances)))
        ora_J = oracle.compute_J(ora_model['eigenvalues'], distances)
        np.testing.assert_allclose(ora_J, jax_J, atol=1e-8,
                                   err_msg=f"J mismatch for {model_name}")


class TestEigenbasisProject:

    @pytest.mark.parametrize('R', [5, 7])
    @pytest.mark.parametrize('model_name,model_fn,A',
                             [c for c in MODEL_CONFIGS if c[2] == 4])
    def test_eigenbasis_project(self, R, model_name, model_fn, A):
        C = 6
        parentIndex, distances = _make_tree(R, seed=50)
        alignment = _make_alignment(R, C, A, seed=51)

        jax_model = model_fn()
        ora_model = _oracle_model_from_jax(jax_model)

        jax_tree = _jax_tree(parentIndex, distances)
        ora_tree = _oracle_tree(parentIndex, distances)

        jax_subM = jax_compute_sub_matrices(jax_model, jnp.array(distances))
        ora_subM = oracle.compute_sub_matrices(ora_model, distances)

        jax_U, jax_lnU, _ = jax_upward_pass(
            jnp.array(alignment), jax_tree, jax_subM, jax_model.pi
        )
        ora_U, ora_lnU, _ = oracle.upward_pass(
            alignment, ora_tree, ora_subM, ora_model['pi']
        )

        jax_D, _ = jax_downward_pass(
            jax_U, jax_lnU, jax_tree, jax_subM, jax_model.pi, jnp.array(alignment)
        )
        ora_D, _ = oracle.downward_pass(
            ora_U, ora_lnU, ora_tree, ora_subM, ora_model['pi'], alignment
        )

        jax_Ut, jax_Dt = jax_eigenbasis_project(jax_U, jax_D, jax_model)
        ora_Ut, ora_Dt = oracle.eigenbasis_project(ora_U, ora_D, ora_model)

        np.testing.assert_allclose(ora_Ut, np.array(jax_Ut), atol=1e-8,
                                   err_msg=f"U_tilde mismatch for R={R}, {model_name}")
        np.testing.assert_allclose(ora_Dt, np.array(jax_Dt), atol=1e-8,
                                   err_msg=f"D_tilde mismatch for R={R}, {model_name}")


class TestLogLike:

    @pytest.mark.parametrize('R', TREE_SIZES)
    @pytest.mark.parametrize('model_name,model_fn,A', MODEL_CONFIGS)
    def test_loglike(self, R, model_name, model_fn, A):
        C = 8
        parentIndex, distances = _make_tree(R, seed=60)
        alignment = _make_alignment(R, C, A, seed=61)

        jax_model = model_fn()
        ora_model = _oracle_model_from_jax(jax_model)
        jax_tree = _jax_tree(parentIndex, distances)
        ora_tree = _oracle_tree(parentIndex, distances)

        jax_ll = np.array(jax_LogLike(jnp.array(alignment), jax_tree, jax_model))
        ora_ll = oracle.LogLike(alignment, ora_tree, ora_model)

        np.testing.assert_allclose(ora_ll, jax_ll, atol=1e-8,
                                   err_msg=f"LogLike mismatch for R={R}, {model_name}")


class TestCounts:

    @pytest.mark.parametrize('R', TREE_SIZES)
    @pytest.mark.parametrize('model_name,model_fn,A',
                             [c for c in MODEL_CONFIGS if c[2] == 4])
    def test_counts_eigensub(self, R, model_name, model_fn, A):
        C = 6
        parentIndex, distances = _make_tree(R, seed=70)
        alignment = _make_alignment(R, C, A, seed=71)

        jax_model = model_fn()
        ora_model = _oracle_model_from_jax(jax_model)
        jax_tree = _jax_tree(parentIndex, distances)
        ora_tree = _oracle_tree(parentIndex, distances)

        jax_c = np.array(jax_Counts(jnp.array(alignment), jax_tree, jax_model))
        ora_c = oracle.Counts(alignment, ora_tree, ora_model)

        np.testing.assert_allclose(ora_c, jax_c, atol=1e-6,
                                   err_msg=f"Counts mismatch for R={R}, {model_name}")

    @pytest.mark.parametrize('R', TREE_SIZES)
    def test_counts_f81_fast(self, R):
        C = 6
        parentIndex, distances = _make_tree(R, seed=72)
        alignment = _make_alignment(R, C, 4, seed=73)

        jax_model = jax_jukes_cantor_model(4)
        ora_model = _oracle_model_from_jax(jax_model)
        jax_tree = _jax_tree(parentIndex, distances)
        ora_tree = _oracle_tree(parentIndex, distances)

        jax_c = np.array(jax_Counts(
            jnp.array(alignment), jax_tree, jax_model, f81_fast_flag=True
        ))
        ora_c = oracle.Counts(alignment, ora_tree, ora_model, f81_fast=True)

        np.testing.assert_allclose(ora_c, jax_c, atol=1e-6,
                                   err_msg=f"F81 fast Counts mismatch for R={R}")

    def test_counts_jc64(self):
        R = 7
        C = 4
        parentIndex, distances = _make_tree(R, seed=74)
        alignment = _make_alignment(R, C, 64, seed=75)

        jax_model = jax_jukes_cantor_model(64)
        ora_model = _oracle_model_from_jax(jax_model)
        jax_tree = _jax_tree(parentIndex, distances)
        ora_tree = _oracle_tree(parentIndex, distances)

        jax_c = np.array(jax_Counts(jnp.array(alignment), jax_tree, jax_model))
        ora_c = oracle.Counts(alignment, ora_tree, ora_model)

        np.testing.assert_allclose(ora_c, jax_c, atol=1e-5,
                                   err_msg="Counts mismatch for JC64")


class TestRootProb:

    @pytest.mark.parametrize('R', TREE_SIZES)
    @pytest.mark.parametrize('model_name,model_fn,A',
                             [c for c in MODEL_CONFIGS if c[2] == 4])
    def test_root_prob(self, R, model_name, model_fn, A):
        C = 8
        parentIndex, distances = _make_tree(R, seed=80)
        alignment = _make_alignment(R, C, A, seed=81)

        jax_model = model_fn()
        ora_model = _oracle_model_from_jax(jax_model)
        jax_tree = _jax_tree(parentIndex, distances)
        ora_tree = _oracle_tree(parentIndex, distances)

        jax_rp = np.array(jax_RootProb(jnp.array(alignment), jax_tree, jax_model))
        ora_rp = oracle.RootProb(alignment, ora_tree, ora_model)

        np.testing.assert_allclose(ora_rp, jax_rp, atol=1e-8,
                                   err_msg=f"RootProb mismatch for R={R}, {model_name}")


class TestMixturePosterior:

    @pytest.mark.parametrize('R', [5, 7])
    def test_mixture_posterior(self, R):
        C = 8
        parentIndex, distances = _make_tree(R, seed=90)
        alignment = _make_alignment(R, C, 4, seed=91)

        base_jax = jax_hky85_diag(2.0, jnp.array([0.25, 0.25, 0.25, 0.25]))
        rates = [0.5, 1.0, 2.0]
        jax_models = [jax_scale_model(base_jax, r) for r in rates]
        log_weights = jnp.log(jnp.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))

        jax_tree = _jax_tree(parentIndex, distances)
        jax_post = np.array(jax_MixturePosterior(
            jnp.array(alignment), jax_tree, jax_models, log_weights
        ))

        base_ora = oracle.hky85_diag(2.0, np.array([0.25, 0.25, 0.25, 0.25]))
        ora_models = [oracle.scale_model(base_ora, r) for r in rates]
        ora_log_weights = np.log(np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))
        ora_tree = _oracle_tree(parentIndex, distances)
        ora_post = oracle.MixturePosterior(alignment, ora_tree, ora_models, ora_log_weights)

        np.testing.assert_allclose(ora_post, jax_post, atol=1e-6,
                                   err_msg=f"MixturePosterior mismatch for R={R}")


class TestBranchMask:

    @pytest.mark.parametrize('R', TREE_SIZES)
    def test_branch_mask(self, R):
        C = 10
        A = 4
        parentIndex, _ = _make_tree(R, seed=100)
        alignment = _make_alignment(R, C, A, seed=101)

        from src.phylo.jax.components import compute_branch_mask as jax_branch_mask
        jax_bm = np.array(jax_branch_mask(jnp.array(alignment), jnp.array(parentIndex), A))
        ora_bm = oracle.compute_branch_mask(alignment, parentIndex, A)

        np.testing.assert_array_equal(ora_bm, jax_bm,
                                      err_msg=f"BranchMask mismatch for R={R}")


class TestModels:

    def test_hky85_eigenvalues(self):
        jax_m = jax_hky85_diag(2.0, jnp.array([0.3, 0.2, 0.25, 0.25]))
        ora_m = oracle.hky85_diag(2.0, np.array([0.3, 0.2, 0.25, 0.25]))
        np.testing.assert_allclose(ora_m['eigenvalues'], np.array(jax_m.eigenvalues), atol=1e-12)
        np.testing.assert_allclose(ora_m['eigenvectors'], np.array(jax_m.eigenvectors), atol=1e-12)

    def test_jc_eigenvalues(self):
        for A in [4, 64]:
            jax_m = jax_jukes_cantor_model(A)
            ora_m = oracle.jukes_cantor_model(A)
            np.testing.assert_allclose(ora_m['eigenvalues'], np.array(jax_m.eigenvalues), atol=1e-12)

    def test_f81_eigenvalues(self):
        pi = np.array([0.35, 0.15, 0.25, 0.25])
        jax_m = jax_f81_model(jnp.array(pi))
        ora_m = oracle.f81_model(pi)
        np.testing.assert_allclose(ora_m['eigenvalues'], np.array(jax_m.eigenvalues), atol=1e-12)

    def test_gamma_rate_categories(self):
        jax_rates, jax_weights = jax_gamma_rate_categories(0.5, 4)
        ora_rates, ora_weights = oracle.gamma_rate_categories(0.5, 4)
        np.testing.assert_allclose(ora_rates, np.array(jax_rates), atol=1e-4)
        np.testing.assert_allclose(ora_weights, np.array(jax_weights), atol=1e-12)
