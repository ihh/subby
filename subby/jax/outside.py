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

    Vmaps over columns so each scan step carries (R, A) instead of (R, C, A),
    reducing per-step copy cost by a factor of C.

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

    parent_of = parentIndex[1:]    # (R-1,)
    sibling_of = sibling[1:]       # (R-1,)
    parent_is_root = (parent_of == 0)  # (R-1,)
    nodes = jnp.arange(1, R, dtype=jnp.int32)

    # Pre-gather sibling contributions: (*H, R-1, C, A)
    if per_column:
        sib_M = subMatrices[..., sibling_of, :, :, :]
        sib_U = U[..., sibling_of, :, :]
        sib_contrib = jnp.einsum('...ncij,...ncj->...nci', sib_M, sib_U)
        par_M = subMatrices[..., parent_of, :, :, :]   # (*H, R-1, C, A, A)
    else:
        sib_M = subMatrices[..., sibling_of, :, :]
        sib_U = U[..., sibling_of, :, :]
        sib_contrib = jnp.einsum('...nij,...ncj->...nci', sib_M, sib_U)
        par_M = subMatrices[..., parent_of, :, :]       # (*H, R-1, A, A)

    sib_lnU = logNormU[..., sibling_of, :]              # (*H, R-1, C)
    par_obs = obs_like[parent_of]                        # (R-1, C, A)

    # Move C to axis 0 for vmapping over columns
    sc_c = jnp.moveaxis(sib_contrib, -2, 0)   # (C, *H, R-1, A)
    slnU_c = jnp.moveaxis(sib_lnU, -1, 0)     # (C, *H, R-1)
    pobs_c = jnp.moveaxis(par_obs, -2, 0)      # (C, R-1, A)

    if per_column:
        pM_c = jnp.moveaxis(par_M, -3, 0)     # (C, *H, R-1, A, A)

    def _one_col(sc, slnU, pobs, pM):
        """Downward pass for a single column.

        Args:
            sc: (*H, R-1, A) sibling contributions
            slnU: (*H, R-1) sibling log-norms
            pobs: (R-1, A) parent observation likelihoods
            pM: (*H, R-1, A, A) parent substitution matrices
        """
        # Move scan dim (R-1) to leading axis
        sc_s = jnp.moveaxis(sc, -2, 0)       # (R-1, *H, A)
        slnU_s = jnp.moveaxis(slnU, -1, 0)   # (R-1, *H)
        pM_s = jnp.moveaxis(pM, -3, 0)       # (R-1, *H, A, A)

        def step(carry, xs):
            D, logNormD = carry  # (*H, R, A), (*H, R)
            node, parent, is_root, sc_n, slnU_n, pM_n, pobs_n = xs

            parent_D = D[..., parent, :]  # (*H, A)
            prop_down = jnp.einsum('...i,...ia->...a', parent_D, pM_n)
            parent_contrib = jnp.where(is_root, rootProb, prop_down)
            parent_contrib = parent_contrib * pobs_n

            D_raw = sc_n * parent_contrib

            lnp = logNormD[..., parent]
            ln_prior = jnp.where(is_root, 0.0, lnp)
            accumulated = slnU_n + ln_prior

            maxD = jnp.max(D_raw, axis=-1)
            maxD = jnp.maximum(maxD, 1e-300)
            D_rescaled = D_raw / maxD[..., None]
            logNormD_node = accumulated + jnp.log(maxD)

            D = D.at[..., node, :].set(D_rescaled)
            logNormD = logNormD.at[..., node].set(logNormD_node)

            return (D, logNormD), None

        D = jnp.zeros((*H, R, A))
        logNormD = jnp.zeros((*H, R))

        (D, logNormD), _ = jax.lax.scan(
            step, (D, logNormD),
            (nodes, parent_of, parent_is_root,
             sc_s, slnU_s, pM_s, pobs),
        )
        return D, logNormD

    if per_column:
        D_c, logNormD_c = jax.vmap(_one_col)(sc_c, slnU_c, pobs_c, pM_c)
    else:
        D_c, logNormD_c = jax.vmap(
            _one_col, in_axes=(0, 0, 0, None),
        )(sc_c, slnU_c, pobs_c, par_M)

    # (C, *H, R, A) → (*H, R, C, A)
    D = jnp.moveaxis(D_c, 0, -2)
    # (C, *H, R) → (*H, R, C)
    logNormD = jnp.moveaxis(logNormD_c, 0, -1)

    return D, logNormD
