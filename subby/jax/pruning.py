from __future__ import annotations

import jax
import jax.numpy as jnp
from .types import Tree
from ._utils import token_to_likelihood


def upward_pass(
    alignment: jnp.ndarray,
    tree: Tree,
    subMatrices: jnp.ndarray,
    rootProb: jnp.ndarray,
    maxChunkSize: int = 128,
    per_column: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Felsenstein pruning: compute inside (U) vectors for all nodes.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree(parentIndex, distanceToParent)
        subMatrices: (*H, R, A, A) or (*H, R, C, A, A) if per_column=True
        rootProb: (*H, A) root/equilibrium frequencies
        maxChunkSize: chunk columns to limit memory
        per_column: if True, subMatrices has per-column substitution matrices

    Returns:
        U: (*H, R, C, A) inside vectors per node (rescaled)
        logNormU: (*H, R, C) per-node subtree log-normalizers
        logLike: (*H, C) log-likelihoods
    """
    parentIndex = tree.parentIndex
    R, C = alignment.shape
    if per_column:
        *H, _R, _C, A, _A = subMatrices.shape
        assert _R == R and _A == A and _C == C
    else:
        *H, _R, A, _A = subMatrices.shape
        assert _R == R and _A == A

    if C > maxChunkSize:
        results = []
        for i in range(0, C, maxChunkSize):
            chunk = alignment[:, i:i + maxChunkSize]
            if per_column:
                chunk_sub = subMatrices[..., :, i:i + maxChunkSize, :, :]
            else:
                chunk_sub = subMatrices
            U_c, lnU_c, ll_c = upward_pass(
                chunk, tree, chunk_sub, rootProb, maxChunkSize, per_column
            )
            results.append((U_c, lnU_c, ll_c))
        U = jnp.concatenate([r[0] for r in results], axis=-2)
        logNormU = jnp.concatenate([r[1] for r in results], axis=-1)
        logLike = jnp.concatenate([r[2] for r in results], axis=-1)
        return U, logNormU, logLike

    # Initialize likelihood from tokens
    likelihood = token_to_likelihood(alignment, A)  # (R, C, A)
    if len(H) > 0:
        likelihood = jnp.broadcast_to(likelihood, (*H, R, C, A)).copy()
    else:
        likelihood = likelihood.copy()

    logNormU = jnp.zeros((*H, R, C))

    # Postorder branches: child R-1 down to 1
    child_indices = jnp.arange(R - 1, 0, -1, dtype=jnp.int32)
    parent_indices = jnp.flip(parentIndex[1:])
    if per_column:
        # subMatrices: (*H, R, C, A, A) — move R axis to front then slice
        sub_mats = jnp.flip(jnp.moveaxis(subMatrices, -4, 0)[1:, ...], axis=0)
    else:
        sub_mats = jnp.flip(jnp.moveaxis(subMatrices, -3, 0)[1:, ...], axis=0)

    postorder = (child_indices, parent_indices, sub_mats)

    step_fn = _upward_step_per_column if per_column else _upward_step
    (likelihood, logNormU), _ = jax.lax.scan(
        step_fn, (likelihood, logNormU), postorder
    )

    # Log-likelihood: log(sum_b pi_b * U_root_b) + logNormU[root]
    root_like = likelihood[..., 0, :, :]  # (*H, C, A)
    logLike = logNormU[..., 0, :] + jnp.log(
        jnp.einsum('...ca,...a->...c', root_like, rootProb)
    )

    # U is just the final likelihood array (rescaled inside vectors)
    U = likelihood
    return U, logNormU, logLike


def _upward_step(carry, branch):
    """Single step of the upward (pruning) scan."""
    likelihood, logNormU = carry
    child, parent, subMatrix = branch

    # Multiply child's contribution into parent:
    # parent_b *= sum_j M_bj * child_j
    child_contrib = jnp.einsum(
        '...ij,...cj->...ci', subMatrix, likelihood[..., child, :, :]
    )
    likelihood = likelihood.at[..., parent, :, :].multiply(child_contrib)

    # Rescale parent to prevent underflow
    maxLike = jnp.max(likelihood[..., parent, :, :], axis=-1)  # (*H, C)
    maxLike = jnp.maximum(maxLike, 1e-300)
    likelihood = likelihood.at[..., parent, :, :].divide(maxLike[..., None])
    log_rescale = jnp.log(maxLike)

    # Accumulate per-node log-normalizer:
    # parent's subtree norm += child's subtree norm + this rescaling
    logNormU = logNormU.at[..., parent, :].add(
        logNormU[..., child, :] + log_rescale
    )

    return (likelihood, logNormU), None


def _upward_step_per_column(carry, branch):
    """Single step of the upward (pruning) scan with per-column subMatrices."""
    likelihood, logNormU = carry
    child, parent, subMatrix = branch
    # subMatrix: (*H, C, A, A) — per-column substitution matrix

    # Multiply child's contribution into parent:
    # parent_b(c) *= sum_j M_bj(c) * child_j(c)
    child_contrib = jnp.einsum(
        '...cij,...cj->...ci', subMatrix, likelihood[..., child, :, :]
    )
    likelihood = likelihood.at[..., parent, :, :].multiply(child_contrib)

    # Rescale parent to prevent underflow
    maxLike = jnp.max(likelihood[..., parent, :, :], axis=-1)  # (*H, C)
    maxLike = jnp.maximum(maxLike, 1e-300)
    likelihood = likelihood.at[..., parent, :, :].divide(maxLike[..., None])
    log_rescale = jnp.log(maxLike)

    # Accumulate per-node log-normalizer
    logNormU = logNormU.at[..., parent, :].add(
        logNormU[..., child, :] + log_rescale
    )

    return (likelihood, logNormU), None
