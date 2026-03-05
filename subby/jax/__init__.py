from __future__ import annotations
from typing import Optional, Union

import jax.numpy as jnp

from .types import Tree, DiagModel, IrrevDiagModel, RateModel, AnyDiagModel, AnyModel
from .diagonalize import (
    diagonalize_rate_matrix, compute_sub_matrices,
    diagonalize_irreversible, compute_sub_matrices_irrev,
    check_detailed_balance, diagonalize_rate_matrix_auto,
)
from .pruning import upward_pass
from .outside import downward_pass
from .eigensub import (
    compute_J, eigenbasis_project, accumulate_C, back_transform,
    compute_J_complex, eigenbasis_project_irrev, accumulate_C_complex, back_transform_irrev,
)
from .f81_fast import f81_counts
from .mixture import mixture_posterior
from .vjp import make_loglike_custom_grad
from .models import (
    hky85_diag, jukes_cantor_model, f81_model,
    gamma_rate_categories, scale_model,
    irrev_model_from_rate_matrix, model_from_rate_matrix,
)


def _ensure_diag(model: AnyModel) -> AnyDiagModel:
    """Auto-diagonalize if given a RateModel."""
    if isinstance(model, RateModel):
        return diagonalize_rate_matrix(model.subRate, model.rootProb)
    if isinstance(model, IrrevDiagModel):
        return model
    return model


def _is_model_list(model) -> bool:
    """Check if model is a list/tuple of models (per-column)."""
    return isinstance(model, (list, tuple)) and len(model) > 0 and not isinstance(model, (Tree, DiagModel, IrrevDiagModel, RateModel))


def _stack_models(models: list[AnyModel]) -> tuple[AnyDiagModel, bool]:
    """Stack a list of models into a single model with leading C dimension.

    Returns:
        (stacked_model, is_irrev)
    """
    models = [_ensure_diag(m) for m in models]
    is_irrev = isinstance(models[0], IrrevDiagModel)
    if is_irrev:
        return IrrevDiagModel(
            eigenvalues=jnp.stack([m.eigenvalues for m in models]),
            eigenvectors=jnp.stack([m.eigenvectors for m in models]),
            eigenvectors_inv=jnp.stack([m.eigenvectors_inv for m in models]),
            pi=jnp.stack([m.pi for m in models]),
        ), True
    else:
        return DiagModel(
            eigenvalues=jnp.stack([m.eigenvalues for m in models]),
            eigenvectors=jnp.stack([m.eigenvectors for m in models]),
            pi=jnp.stack([m.pi for m in models]),
        ), False


def _compute_sub_matrices_per_column(
    stacked_model: AnyDiagModel, distances: jnp.ndarray, is_irrev: bool
) -> jnp.ndarray:
    """Compute per-column substitution matrices from a stacked model.

    stacked_model fields have leading C dimension.
    Returns: (R, C, A, A) substitution matrices.
    """
    if is_irrev:
        V = stacked_model.eigenvectors       # (C, A, A)
        V_inv = stacked_model.eigenvectors_inv  # (C, A, A)
        mu = stacked_model.eigenvalues        # (C, A)
        # exp(mu_k * t_r): (C, R, A)
        exp_mu_t = jnp.exp(mu[:, None, :] * distances[None, :, None])
        # M_ij(c,t) = sum_k V_ik(c) * exp(mu_k(c)*t) * V_inv_kj(c)
        M = jnp.einsum('cak,crk,ckj->craj', V, exp_mu_t, V_inv)
        # Transpose to (R, C, A, A)
        M = jnp.moveaxis(M, 0, 1)
        return M.real
    else:
        V = stacked_model.eigenvectors   # (C, A, A)
        mu = stacked_model.eigenvalues   # (C, A)
        pi = stacked_model.pi            # (C, A)
        exp_mu_t = jnp.exp(mu[:, None, :] * distances[None, :, None])
        S_t = jnp.einsum('cak,crk,cbk->crab', V, exp_mu_t, V)
        sqrt_pi = jnp.sqrt(pi)
        inv_sqrt_pi = 1.0 / sqrt_pi
        # M_ij = sqrt(pi_j/pi_i) * S_ij
        M = S_t * inv_sqrt_pi[:, None, :, None] * sqrt_pi[:, None, None, :]
        # Transpose to (R, C, A, A)
        M = jnp.moveaxis(M, 0, 1)
        return M


def LogLike(
    alignment: jnp.ndarray,
    tree: Tree,
    model,
    maxChunkSize: int = 128,
) -> jnp.ndarray:
    """Compute per-column log-likelihoods.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel, IrrevDiagModel, RateModel, or list of models
            (one per column, len must equal C)
        maxChunkSize: chunk columns for memory

    Returns:
        (*H, C) log-likelihoods
    """
    if _is_model_list(model):
        stacked, is_irrev = _stack_models(model)
        subMatrices = _compute_sub_matrices_per_column(
            stacked, tree.distanceToParent, is_irrev
        )
        # Use first model's pi for root (all should be same or broadcastable)
        rootProb = stacked.pi[0]
        _, _, logLike = upward_pass(
            alignment, tree, subMatrices, rootProb, maxChunkSize, per_column=True
        )
        return logLike
    model = _ensure_diag(model)
    if isinstance(model, IrrevDiagModel):
        subMatrices = compute_sub_matrices_irrev(model, tree.distanceToParent)
    else:
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)
    _, _, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
    return logLike


def LogLikeCustomGrad(
    alignment: jnp.ndarray,
    tree: Tree,
    model: AnyModel,
    maxChunkSize: int = 128,
) -> jnp.ndarray:
    """Like LogLike but with custom VJP for faster distance gradients.

    Uses the Fisher identity: the gradient of log-likelihood w.r.t. branch
    lengths equals a contraction of expected substitution counts, which can
    be computed via the downward pass + eigenbasis projection without tracing
    through the full computation graph.

    Same signature and output as LogLike. Only the backward pass differs.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel, IrrevDiagModel, or RateModel
        maxChunkSize: chunk columns for memory

    Returns:
        (*H, C) log-likelihoods
    """
    model = _ensure_diag(model)
    f = make_loglike_custom_grad(model, alignment, tree.parentIndex, maxChunkSize)
    return f(tree.distanceToParent)


def Counts(
    alignment: jnp.ndarray,
    tree: Tree,
    model,
    maxChunkSize: int = 128,
    f81_fast_flag: bool = False,
    branch_mask: Optional[Union[str, jnp.ndarray]] = "auto",
) -> jnp.ndarray:
    """Compute expected substitution counts and dwell times per column.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel, IrrevDiagModel, RateModel, or list of models
            (one per column, len must equal C)
        maxChunkSize: chunk columns for memory
        f81_fast_flag: if True, use O(CRA^2) F81/JC fast path
        branch_mask: "auto" (compute from alignment), None (no masking),
            or (*H, R, C) bool array

    Returns:
        (*H, A, A, C) counts tensor (diag=dwell, off-diag=substitution counts)
    """
    if _is_model_list(model):
        # Per-column models: compute column-by-column since the eigensub
        # pipeline requires model fields to align with *H batch dims,
        # which conflicts with the C (column) dimension.
        models_diag = [_ensure_diag(m) for m in model]
        C = alignment.shape[1]
        assert len(models_diag) == C, f"Expected {C} models, got {len(models_diag)}"
        results = []
        for c in range(C):
            col_counts = Counts(
                alignment[:, c:c+1], tree, models_diag[c],
                maxChunkSize=maxChunkSize,
                f81_fast_flag=f81_fast_flag,
                branch_mask=branch_mask[..., c:c+1] if branch_mask is not None and not isinstance(branch_mask, str) else branch_mask,
            )
            results.append(col_counts)
        return jnp.concatenate(results, axis=-1)

    model = _ensure_diag(model)
    is_irrev = isinstance(model, IrrevDiagModel)

    if isinstance(branch_mask, str) and branch_mask == "auto":
        from .components import compute_branch_mask
        A = model.pi.shape[-1]
        branch_mask = compute_branch_mask(alignment, tree.parentIndex, A)

    if is_irrev:
        assert not f81_fast_flag, "F81 fast path is reversible-only"
        subMatrices = compute_sub_matrices_irrev(model, tree.distanceToParent)
    else:
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)

    U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
    D, logNormD = downward_pass(U, logNormU, tree, subMatrices, model.pi, alignment)

    if f81_fast_flag:
        return f81_counts(
            U, D, logNormU, logNormD, logLike,
            tree.distanceToParent, model.pi, tree.parentIndex,
            branch_mask=branch_mask,
        )
    elif is_irrev:
        J = compute_J_complex(model.eigenvalues, tree.distanceToParent)
        U_tilde, D_tilde = eigenbasis_project_irrev(U, D, model)
        C = accumulate_C_complex(D_tilde, U_tilde, J, logNormU, logNormD, logLike,
                                 tree.parentIndex, branch_mask=branch_mask)
        return back_transform_irrev(C, model)
    else:
        J = compute_J(model.eigenvalues, tree.distanceToParent)
        U_tilde, D_tilde = eigenbasis_project(U, D, model)
        C = accumulate_C(D_tilde, U_tilde, J, logNormU, logNormD, logLike, tree.parentIndex,
                         branch_mask=branch_mask)
        return back_transform(C, model)


def RootProb(
    alignment: jnp.ndarray,
    tree: Tree,
    model,
    maxChunkSize: int = 128,
) -> jnp.ndarray:
    """Compute posterior root state distribution per column.

    q_b = pi_b * U^(root)_b / P(x)

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel, IrrevDiagModel, RateModel, or list of models
            (one per column, len must equal C)
        maxChunkSize: chunk columns for memory

    Returns:
        (*H, A, C) posterior root distribution
    """
    if _is_model_list(model):
        stacked, is_irrev = _stack_models(model)
        subMatrices = _compute_sub_matrices_per_column(
            stacked, tree.distanceToParent, is_irrev
        )
        rootProb = stacked.pi[0]
        U, logNormU, logLike = upward_pass(
            alignment, tree, subMatrices, rootProb, maxChunkSize, per_column=True
        )
        # Per-column pi: stacked.pi is (C, A)
        pi_per_col = stacked.pi  # (C, A)
        root_U = U[..., 0, :, :]  # (*H, C, A)
        log_scale = logNormU[..., 0, :] - logLike
        scale = jnp.exp(log_scale)
        q = pi_per_col * root_U * scale[..., None]
        q = jnp.moveaxis(q, -1, -2)
        return q

    model = _ensure_diag(model)
    if isinstance(model, IrrevDiagModel):
        subMatrices = compute_sub_matrices_irrev(model, tree.distanceToParent)
    else:
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


def MixturePosterior(
    alignment: jnp.ndarray,
    tree: Tree,
    models: list[AnyModel],
    log_weights: jnp.ndarray,
    maxChunkSize: int = 128,
) -> jnp.ndarray:
    """Compute posterior over mixture components per column.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        models: list of K DiagModel, IrrevDiagModel, or RateModel instances
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
