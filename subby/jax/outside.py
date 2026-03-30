from __future__ import annotations

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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute outside (D) vectors for all nodes.

    D^(n)_a = [sum_j M_{aj}(t_{p,s}) U^(s)_j] * L(x_p|a) *
              [pi_a if p=root else sum_i D^(p)_i M_{ia}(t_{g,p})]

    where s is n's sibling, p is n's parent, g is p's parent,
    and L(x_p|a) is the observation likelihood at the parent node.

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
    *H, R, C, A = U.shape

    _, _, sibling = children_of(parentIndex)

    # Observation likelihoods per node: (R, C, A)
    obs_like = token_to_likelihood(alignment, A)

    # Scan xs: just integer indices (tiny per step, like the upward pass).
    # All large arrays (U, subMatrices, obs_like, logNormU) are gathered
    # inside the step body from closure-captured constants.
    nodes = jnp.arange(1, R, dtype=jnp.int32)
    parent_of = parentIndex[1:]       # (R-1,)
    sibling_of = sibling[1:]          # (R-1,)

    # Move R-1 to leading axis for subMatrices (like upward pass does)
    if per_column:
        sub_mats = jnp.moveaxis(subMatrices, -4, 0)   # (R, *H, C, A, A)
    else:
        sub_mats = jnp.moveaxis(subMatrices, -3, 0)    # (R, *H, A, A)

    U_r = jnp.moveaxis(U, -3, 0)            # (R, *H, C, A)
    logNormU_r = jnp.moveaxis(logNormU, -1, 0)  # keep for indexing; shape depends on H

    D = jnp.zeros((*H, R, C, A))
    logNormD = jnp.zeros((*H, R, C))

    if per_column:
        step_fn = _make_step_per_column(sub_mats, U_r, logNormU, obs_like, rootProb)
    else:
        step_fn = _make_step(sub_mats, U_r, logNormU, obs_like, rootProb)

    (D, logNormD), _ = jax.lax.scan(
        step_fn, (D, logNormD),
        (nodes, parent_of, sibling_of),
    )

    return D, logNormD


def _make_step(sub_mats, U_r, logNormU, obs_like, rootProb):
    """Create step function with large arrays captured in closure."""

    def step(carry, xs):
        D, logNormD = carry
        node, parent, sib = xs

        # Sibling contribution: M^(sib) @ U^(sib)  (computed on the fly)
        sib_M = sub_mats[sib]                          # (*H, A, A)
        sib_U = U_r[sib]                               # (*H, C, A)
        sib_contrib = jnp.einsum('...ij,...cj->...ci', sib_M, sib_U)
        sib_lnU = logNormU[..., sib, :]                # (*H, C)

        # Parent contribution: D^(parent) @ M^(parent)
        parent_D = D[..., parent, :, :]                # (*H, C, A)
        par_M = sub_mats[parent]                       # (*H, A, A)
        prop_down = jnp.einsum('...ci,...ia->...ca', parent_D, par_M)

        is_root = (parent == 0)
        root_contrib = jnp.broadcast_to(rootProb[..., None, :], prop_down.shape)
        parent_contrib = jnp.where(is_root, root_contrib, prop_down)
        parent_contrib = parent_contrib * obs_like[parent]  # (C, A) broadcast

        D_raw = sib_contrib * parent_contrib

        # Log-norm accumulation
        log_norm_from_parent = logNormD[..., parent, :]
        log_norm_prior = jnp.where(is_root, 0.0, log_norm_from_parent)
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

    return step


def _make_step_per_column(sub_mats, U_r, logNormU, obs_like, rootProb):
    """Create step function for per-column subMatrices."""

    def step(carry, xs):
        D, logNormD = carry
        node, parent, sib = xs

        sib_M = sub_mats[sib]                          # (*H, C, A, A)
        sib_U = U_r[sib]                               # (*H, C, A)
        sib_contrib = jnp.einsum('...cij,...cj->...ci', sib_M, sib_U)
        sib_lnU = logNormU[..., sib, :]

        parent_D = D[..., parent, :, :]
        par_M = sub_mats[parent]                       # (*H, C, A, A)
        prop_down = jnp.einsum('...ci,...cia->...ca', parent_D, par_M)

        is_root = (parent == 0)
        root_contrib = jnp.broadcast_to(rootProb[..., None, :], prop_down.shape)
        parent_contrib = jnp.where(is_root, root_contrib, prop_down)
        parent_contrib = parent_contrib * obs_like[parent]

        D_raw = sib_contrib * parent_contrib

        log_norm_from_parent = logNormD[..., parent, :]
        log_norm_prior = jnp.where(is_root, 0.0, log_norm_from_parent)
        accumulated = sib_lnU + log_norm_prior

        maxD = jnp.max(D_raw, axis=-1)
        maxD = jnp.maximum(maxD, 1e-300)
        D_rescaled = D_raw / maxD[..., None]
        log_rescale = jnp.log(maxD)

        logNormD_node = accumulated + log_rescale

        D = D.at[..., node, :, :].set(D_rescaled)
        logNormD = logNormD.at[..., node, :].set(logNormD_node)

        return (D, logNormD), None

    return step
