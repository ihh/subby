from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from .types import Tree
from ._utils import children_of, token_to_likelihood


def _compute_depths(parentIndex: jnp.ndarray) -> np.ndarray:
    """Compute depth of each node in the tree.

    Args:
        parentIndex: (R,) int32 parent indices (parentIndex[0] = -1 for root)

    Returns:
        (R,) int32 numpy array of depths (root = 0)
    """
    parent_np = np.asarray(parentIndex)
    R = len(parent_np)
    depths = np.zeros(R, dtype=np.int32)
    for i in range(1, R):
        depths[i] = depths[parent_np[i]] + 1
    return depths


def downward_pass(
    U: jnp.ndarray,
    logNormU: jnp.ndarray,
    tree: Tree,
    subMatrices: jnp.ndarray,
    rootProb: jnp.ndarray,
    alignment: jnp.ndarray,
    per_column: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute outside (D) vectors for all nodes.

    Uses level-parallel computation when parentIndex is concrete: nodes are
    grouped by tree depth and all nodes at the same depth are computed in one
    vectorized operation. For a balanced binary tree this reduces O(R) sequential
    steps to O(log R).

    Falls back to sequential scan when parentIndex is traced (e.g. inside vmap).

    Args:
        U: (*H, R, C, A) inside vectors from upward_pass (rescaled)
        logNormU: (*H, R, C) per-node subtree log-normalizers
        tree: Tree(parentIndex, distanceToParent)
        subMatrices: (*H, R, A, A) or (*H, R, C, A, A) if per_column=True
        rootProb: (*H, A) equilibrium frequencies
        alignment: (R, C) int32 token alignment
        per_column: if True, subMatrices has per-column substitution matrices

    Returns:
        D: (*H, R, C, A) outside vectors per node (rescaled)
        logNormD: (*H, R, C) per-node outside log-normalizers
    """
    parentIndex = tree.parentIndex

    try:
        return _downward_pass_level_parallel(
            U, logNormU, parentIndex, subMatrices, rootProb, alignment, per_column
        )
    except jax.errors.TracerArrayConversionError:
        return _downward_pass_scan(
            U, logNormU, parentIndex, subMatrices, rootProb, alignment, per_column
        )


def _downward_pass_level_parallel(
    U, logNormU, parentIndex, subMatrices, rootProb, alignment, per_column,
):
    """Level-parallel downward pass for concrete (non-traced) parentIndex."""
    *H, R, C, A = U.shape

    _, _, sibling = children_of(parentIndex)

    obs_like = token_to_likelihood(alignment, A)

    # Precompute per-node data for all non-root nodes (indexed 0..R-2)
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

    # Compute depths and group nodes by level
    depths = _compute_depths(parentIndex)
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


def _downward_pass_scan(
    U, logNormU, parentIndex, subMatrices, rootProb, alignment, per_column,
):
    """Sequential scan fallback for traced parentIndex (e.g. inside vmap)."""
    *H, R, C, A = U.shape

    _, _, sibling = children_of(parentIndex)

    obs_like = token_to_likelihood(alignment, A)

    preorder_nodes = jnp.arange(1, R, dtype=jnp.int32)
    parent_of = parentIndex[1:]
    sibling_of = sibling[1:]

    if per_column:
        sib_M = subMatrices[..., sibling_of, :, :, :]
        sib_U = U[..., sibling_of, :, :]
        sib_contrib_all = jnp.einsum('...ncij,...ncj->...nci', sib_M, sib_U)
        parent_M = subMatrices[..., parent_of, :, :, :]
    else:
        sib_M = subMatrices[..., sibling_of, :, :]
        sib_U = U[..., sibling_of, :, :]
        sib_contrib_all = jnp.einsum('...nij,...ncj->...nci', sib_M, sib_U)
        parent_M = subMatrices[..., parent_of, :, :]

    sib_logNormU = logNormU[..., sibling_of, :]
    parent_obs = obs_like[parent_of]
    parent_is_root = (parent_of == 0)

    D = jnp.zeros((*H, R, C, A))
    logNormD = jnp.zeros((*H, R, C))

    init = (D, logNormD)

    if per_column:
        step_fn = _scan_step_per_column
    else:
        step_fn = _scan_step

    (D, logNormD), _ = jax.lax.scan(
        lambda carry, xs: step_fn(carry, xs, rootProb),
        init,
        (preorder_nodes, parent_of, parent_is_root,
         sib_contrib_all, sib_logNormU, parent_M, parent_obs),
    )

    return D, logNormD


def _scan_step(carry, xs, rootProb):
    """Sequential scan step (fallback for traced parentIndex)."""
    D, logNormD = carry
    (node, parent, par_is_root,
     sib_contrib, sib_lnU, par_M, par_obs) = xs

    parent_D = D[..., parent, :, :]
    prop_down = jnp.einsum('...ci,...ia->...ca', parent_D, par_M)

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


def _scan_step_per_column(carry, xs, rootProb):
    """Sequential scan step for per-column subMatrices (fallback)."""
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
