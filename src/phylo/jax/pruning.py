import jax
import jax.numpy as jnp
from ._utils import token_to_likelihood


def upward_pass(alignment, tree, subMatrices, rootProb, maxChunkSize=128):
    """Felsenstein pruning: compute inside (U) vectors for all nodes.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree(parentIndex, distanceToParent)
        subMatrices: (*H, R, A, A) substitution probability matrices
        rootProb: (*H, A) root/equilibrium frequencies
        maxChunkSize: chunk columns to limit memory

    Returns:
        U: (*H, R, C, A) inside vectors per node (rescaled)
        logNormU: (*H, R, C) per-node subtree log-normalizers
        logLike: (*H, C) log-likelihoods
    """
    parentIndex = tree.parentIndex
    R, C = alignment.shape
    *H, _R, A, _A = subMatrices.shape
    assert _R == R and _A == A

    if C > maxChunkSize:
        results = []
        for i in range(0, C, maxChunkSize):
            chunk = alignment[:, i:i + maxChunkSize]
            U_c, lnU_c, ll_c = upward_pass(chunk, tree, subMatrices, rootProb, maxChunkSize)
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
    sub_mats = jnp.flip(jnp.moveaxis(subMatrices, -3, 0)[1:, ...], axis=0)

    postorder = (child_indices, parent_indices, sub_mats)

    (likelihood, logNormU), _ = jax.lax.scan(
        _upward_step, (likelihood, logNormU), postorder
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
