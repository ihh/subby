import jax.numpy as jnp

from .types import Tree, DiagModel, RateModel
from .diagonalize import diagonalize_rate_matrix, compute_sub_matrices
from .pruning import upward_pass
from .outside import downward_pass
from .eigensub import compute_J, eigenbasis_project, accumulate_C, back_transform
from .f81_fast import f81_counts
from .mixture import mixture_posterior


def _ensure_diag(model):
    """Auto-diagonalize if given a RateModel."""
    if isinstance(model, RateModel):
        return diagonalize_rate_matrix(model.subRate, model.rootProb)
    return model


def LogLike(alignment, tree, model, maxChunkSize=128):
    """Compute per-column log-likelihoods.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel or RateModel
        maxChunkSize: chunk columns for memory

    Returns:
        (*H, C) log-likelihoods
    """
    model = _ensure_diag(model)
    subMatrices = compute_sub_matrices(model, tree.distanceToParent)
    _, _, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
    return logLike


def Counts(alignment, tree, model, maxChunkSize=128, f81_fast_flag=False):
    """Compute expected substitution counts and dwell times per column.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel or RateModel
        maxChunkSize: chunk columns for memory
        f81_fast_flag: if True, use O(CRA^2) F81/JC fast path

    Returns:
        (*H, A, A, C) counts tensor (diag=dwell, off-diag=substitution counts)
    """
    model = _ensure_diag(model)
    subMatrices = compute_sub_matrices(model, tree.distanceToParent)
    U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
    D, logNormD = downward_pass(U, logNormU, tree, subMatrices, model.pi, alignment)

    if f81_fast_flag:
        return f81_counts(
            U, D, logNormU, logNormD, logLike,
            tree.distanceToParent, model.pi, tree.parentIndex,
        )
    else:
        J = compute_J(model.eigenvalues, tree.distanceToParent)
        U_tilde, D_tilde = eigenbasis_project(U, D, model)
        C = accumulate_C(D_tilde, U_tilde, J, logNormU, logNormD, logLike, tree.parentIndex)
        return back_transform(C, model)


def RootProb(alignment, tree, model, maxChunkSize=128):
    """Compute posterior root state distribution per column.

    q_b = pi_b * U^(root)_b / P(x)

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel or RateModel
        maxChunkSize: chunk columns for memory

    Returns:
        (*H, A, C) posterior root distribution
    """
    model = _ensure_diag(model)
    subMatrices = compute_sub_matrices(model, tree.distanceToParent)
    U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)

    # q_b = pi_b * U_root_b / P(x)
    # With rescaling: q_b = pi_b * U_rescaled_root_b * exp(logNormU_root) / exp(logLike)
    root_U = U[..., 0, :, :]  # (*H, C, A)
    log_scale = logNormU[..., 0, :] - logLike  # (*H, C)
    scale = jnp.exp(log_scale)
    q = model.pi[..., None, :] * root_U * scale[..., None]  # (*H, C, A)
    # Transpose to (*H, A, C)
    q = jnp.moveaxis(q, -1, -2)
    return q


def MixturePosterior(alignment, tree, models, log_weights, maxChunkSize=128):
    """Compute posterior over mixture components per column.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        models: list of K DiagModel or RateModel instances
        log_weights: (K,) log prior weights
        maxChunkSize: chunk columns for memory

    Returns:
        (K, C) posterior probabilities
    """
    log_likes = jnp.stack([
        LogLike(alignment, tree, m, maxChunkSize) for m in models
    ])  # (K, C) or (K, *H, C)
    # If there are hidden dims H, sum over them for mixture
    while log_likes.ndim > 2:
        log_likes = jnp.sum(log_likes, axis=1)
    return mixture_posterior(log_likes, log_weights)
