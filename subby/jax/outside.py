from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from .types import Tree
from ._utils import children_of, token_to_likelihood


def downward_pass(
    U: jnp.ndarray,
    logNormU: jnp.ndarray,
    tree: Tree,
    subMatrices: jnp.ndarray,
    rootProb: jnp.ndarray,
    alignment: jnp.ndarray,
    per_column: bool = False,
    parallel: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute outside (D) vectors for all nodes.

    D^(n)_a = [sum_j M_{aj}(t_{p,s}) U^(s)_j] * L(x_p|a) *
              [pi_a if p=root else sum_i D^(p)_i M_{ia}(t_{g,p})]

    where s is n's sibling, p is n's parent, g is p's parent,
    and L(x_p|a) is the observation likelihood at the parent node.

    Optimization: sibling U/logNormU, substitution matrices, and observation
    likelihoods are pre-gathered before the scan so that each step only needs
    to index into the carry (D, logNormD) for the parent's values rather than
    indexing into multiple large arrays.

    Args:
        U: (*H, R, C, A) inside vectors from upward_pass (rescaled)
        logNormU: (*H, R, C) per-node subtree log-normalizers
        tree: Tree(parentIndex, distanceToParent)
        subMatrices: (*H, R, A, A) or (*H, R, C, A, A) if per_column=True
        rootProb: (*H, A) equilibrium frequencies
        alignment: (R, C) int32 token alignment
        per_column: if True, subMatrices has per-column substitution matrices
        parallel: if True, use level-parallel traversal (O(log R) iterations
            for balanced trees). Requires concrete parentIndex (not traced).
            Useful when the same tree topology is reused many times.

    Returns:
        D: (*H, R, C, A) outside vectors per node (rescaled)
        logNormD: (*H, R, C) per-node outside log-normalizers
    """
    if parallel:
        return _downward_pass_parallel(
            U, logNormU, tree, subMatrices, rootProb, alignment, per_column,
        )

    parentIndex = tree.parentIndex
    *H, R, C, A = U.shape

    _, _, sibling = children_of(parentIndex)

    # Observation likelihoods per node: (R, C, A)
    obs_like = token_to_likelihood(alignment, A)

    # Precompute per-step data (nodes 1..R-1)
    preorder_nodes = jnp.arange(1, R, dtype=jnp.int32)
    parent_of = parentIndex[1:]
    sibling_of = sibling[1:]

    # Pre-gather sibling contributions (avoids indexing U/logNormU in scan body)
    # Sibling's contribution: M @ U^(sib), pre-computed for all nodes
    if per_column:
        sib_M = subMatrices[..., sibling_of, :, :, :]    # (*H, R-1, C, A, A)
        sib_U = U[..., sibling_of, :, :]                 # (*H, R-1, C, A)
        sib_contrib_all = jnp.einsum('...ncij,...ncj->...nci', sib_M, sib_U)  # (*H, R-1, C, A)
        parent_M = subMatrices[..., parent_of, :, :, :]   # (*H, R-1, C, A, A)
    else:
        sib_M = subMatrices[..., sibling_of, :, :]        # (*H, R-1, A, A)
        sib_U = U[..., sibling_of, :, :]                  # (*H, R-1, C, A)
        sib_contrib_all = jnp.einsum('...nij,...ncj->...nci', sib_M, sib_U)  # (*H, R-1, C, A)
        parent_M = subMatrices[..., parent_of, :, :]      # (*H, R-1, A, A)

    sib_logNormU = logNormU[..., sibling_of, :]           # (*H, R-1, C)
    parent_obs = obs_like[parent_of]                       # (R-1, C, A)
    parent_is_root = (parent_of == 0)                      # (R-1,) bool

    D = jnp.zeros((*H, R, C, A))
    logNormD = jnp.zeros((*H, R, C))

    init = (D, logNormD)

    if per_column:
        step_fn = _downward_step_opt_per_column
    else:
        step_fn = _downward_step_opt

    (D, logNormD), _ = jax.lax.scan(
        lambda carry, xs: step_fn(carry, xs, rootProb),
        init,
        (preorder_nodes, parent_of, parent_is_root,
         sib_contrib_all, sib_logNormU, parent_M, parent_obs),
    )

    return D, logNormD


def _downward_step_opt(carry, xs, rootProb):
    """Optimized downward step with pre-computed sibling contributions."""
    D, logNormD = carry
    (node, parent, par_is_root,
     sib_contrib, sib_lnU, par_M, par_obs) = xs

    # Parent contribution (only thing that requires indexing into carry)
    parent_D = D[..., parent, :, :]  # (*H, C, A)
    prop_down = jnp.einsum('...ci,...ia->...ca', parent_D, par_M)

    root_contrib = jnp.broadcast_to(rootProb[..., None, :], prop_down.shape)
    parent_contrib = jnp.where(par_is_root, root_contrib, prop_down)
    parent_contrib = parent_contrib * par_obs

    D_raw = sib_contrib * parent_contrib

    # Log-norm
    log_norm_from_parent = logNormD[..., parent, :]
    log_norm_prior = jnp.where(par_is_root, 0.0, log_norm_from_parent)
    accumulated = sib_lnU + log_norm_prior

    # Rescale
    maxD = jnp.max(D_raw, axis=-1)
    maxD = jnp.maximum(maxD, 1e-300)
    D_rescaled = D_raw / maxD[..., None]
    log_rescale = jnp.log(maxD)

    logNormD_node = accumulated + log_rescale

    D = D.at[..., node, :, :].set(D_rescaled)
    logNormD = logNormD.at[..., node, :].set(logNormD_node)

    return (D, logNormD), None


def _downward_step_opt_per_column(carry, xs, rootProb):
    """Optimized downward step for per-column subMatrices."""
    D, logNormD = carry
    (node, parent, par_is_root,
     sib_contrib, sib_lnU, par_M, par_obs) = xs

    parent_D = D[..., parent, :, :]
    prop_down = jnp.einsum('...ci,...cia->...ca', parent_D, par_M)

    root_contrib = jnp.broadcast_to(rootProb[..., None, :], prop_down.shape)
    parent_contrib = jnp.where(par_is_root, root_contrib, prop_down)
    parent_contrib = parent_contrib * par_obs

    D_raw = sib_contrib * parent_contrib

    log_norm_from_parent = logNormD[..., parent, :]
    log_norm_prior = jnp.where(par_is_root, 0.0, log_norm_from_parent)
    accumulated = sib_lnU + log_norm_prior

    maxD = jnp.max(D_raw, axis=-1)
    maxD = jnp.maximum(maxD, 1e-300)
    D_rescaled = D_raw / maxD[..., None]
    log_rescale = jnp.log(maxD)

    logNormD_node = accumulated + log_rescale

    D = D.at[..., node, :, :].set(D_rescaled)
    logNormD = logNormD.at[..., node, :].set(logNormD_node)

    return (D, logNormD), None


def _downward_pass_parallel(
    U, logNormU, tree, subMatrices, rootProb, alignment, per_column,
):
    """Level-parallel downward pass.

    Groups nodes by tree depth and processes each level in one vectorized
    operation. For a balanced binary tree with R nodes, this reduces R-1
    sequential scan steps to O(log R) parallel iterations.

    Requires concrete (non-traced) parentIndex. Intended for use cases where
    the same tree topology is reused many times (e.g. genome-wide eigensub).
    """
    parentIndex = tree.parentIndex
    *H, R, C, A = U.shape

    _, _, sibling = children_of(parentIndex)

    obs_like = token_to_likelihood(alignment, A)

    # Precompute sibling contributions for all non-root nodes
    sibling_of = sibling[1:]

    if per_column:
        sib_M = subMatrices[..., sibling_of, :, :, :]
        sib_U = U[..., sibling_of, :, :]
        sib_contrib_all = jnp.einsum('...ncij,...ncj->...nci', sib_M, sib_U)
    else:
        sib_M = subMatrices[..., sibling_of, :, :]
        sib_U = U[..., sibling_of, :, :]
        sib_contrib_all = jnp.einsum('...nij,...ncj->...nci', sib_M, sib_U)

    sib_logNormU_all = logNormU[..., sibling_of, :]

    # Compute depths (requires concrete parentIndex)
    parent_np = np.asarray(parentIndex)
    depths = np.zeros(R, dtype=np.int32)
    for i in range(1, R):
        depths[i] = depths[parent_np[i]] + 1
    max_depth = int(depths.max())

    D = jnp.zeros((*H, R, C, A))
    logNormD = jnp.zeros((*H, R, C))

    for level in range(1, max_depth + 1):
        node_indices = np.where(depths == level)[0]
        idx = node_indices - 1  # index into pre-gathered arrays

        nodes = jnp.array(node_indices, dtype=jnp.int32)
        parents = parentIndex[nodes]
        par_is_root = (parents == 0)

        sc = sib_contrib_all[..., idx, :, :]
        slnU = sib_logNormU_all[..., idx, :]
        par_obs = obs_like[parents]

        parent_D = D[..., parents, :, :]

        if per_column:
            par_M = subMatrices[..., parents, :, :, :]
            prop_down = jnp.einsum('...nci,...ncia->...nca', parent_D, par_M)
        else:
            par_M = subMatrices[..., parents, :, :]
            prop_down = jnp.einsum('...nci,...nia->...nca', parent_D, par_M)

        root_contrib = jnp.broadcast_to(
            rootProb[..., None, None, :], prop_down.shape
        )
        parent_contrib = jnp.where(
            par_is_root[:, None, None], root_contrib, prop_down
        )
        parent_contrib = parent_contrib * par_obs

        D_raw = sc * parent_contrib

        log_norm_from_parent = logNormD[..., parents, :]
        log_norm_prior = jnp.where(par_is_root[:, None], 0.0, log_norm_from_parent)
        accumulated = slnU + log_norm_prior

        maxD = jnp.max(D_raw, axis=-1)
        maxD = jnp.maximum(maxD, 1e-300)
        D_rescaled = D_raw / maxD[..., None]
        log_rescale = jnp.log(maxD)

        logNormD_nodes = accumulated + log_rescale

        D = D.at[..., nodes, :, :].set(D_rescaled)
        logNormD = logNormD.at[..., nodes, :].set(logNormD_nodes)

    return D, logNormD
