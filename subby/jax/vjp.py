"""Custom VJP for LogLike — fast distance gradients via Fisher identity.

The gradient of log-likelihood w.r.t. branch length t_n is:
  ∂logL_c/∂t_n = Σ_k D̃_k^(n,c) · μ_k · exp(μ_k·t_n) · Ũ_k^(n,c) · scale(n,c)

This avoids tracing through the full computation graph by reusing the
upward_pass from the forward pass and only adding the downward_pass +
eigenbasis projection in the backward pass.
"""
from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp

from .types import Tree, DiagModel, IrrevDiagModel, AnyDiagModel
from .diagonalize import compute_sub_matrices, compute_sub_matrices_irrev
from .pruning import upward_pass
from .outside import downward_pass
from .eigensub import eigenbasis_project, eigenbasis_project_irrev


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _loglike_custom_distances(distances, alignment, parentIndex, model, maxChunkSize):
    """LogLike parameterized by distances only (for custom VJP)."""
    tree = Tree(parentIndex=parentIndex, distanceToParent=distances)
    is_irrev = isinstance(model, IrrevDiagModel)
    if is_irrev:
        subMatrices = compute_sub_matrices_irrev(model, distances)
    else:
        subMatrices = compute_sub_matrices(model, distances)
    _, _, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
    return logLike


def _fwd(alignment, parentIndex, model, maxChunkSize, distances):
    tree = Tree(parentIndex=parentIndex, distanceToParent=distances)
    is_irrev = isinstance(model, IrrevDiagModel)
    if is_irrev:
        subMatrices = compute_sub_matrices_irrev(model, distances)
    else:
        subMatrices = compute_sub_matrices(model, distances)
    U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
    residuals = (distances, U, logNormU, logLike, subMatrices)
    return logLike, residuals


def _bwd(alignment, parentIndex, model, maxChunkSize, residuals, g):
    """Backward pass: compute distance gradient via eigenbasis projection.

    g: (*H, C) upstream gradient (cotangent for logLike)
    """
    distances, U, logNormU, logLike, subMatrices = residuals
    tree = Tree(parentIndex=parentIndex, distanceToParent=distances)
    is_irrev = isinstance(model, IrrevDiagModel)

    # Downward pass to get outside vectors
    D, logNormD = downward_pass(U, logNormU, tree, subMatrices, model.pi, alignment)

    # Project into eigenbasis
    if is_irrev:
        U_tilde, D_tilde = eigenbasis_project_irrev(U, D, model)
        mu = model.eigenvalues  # (*H, A) complex
    else:
        U_tilde, D_tilde = eigenbasis_project(U, D, model)
        mu = model.eigenvalues  # (*H, A) real

    # Per-branch distance gradient:
    # ∂logL_c/∂t_n = Σ_k D̃_k^(n,c) · μ_k · exp(μ_k·t_n) · Ũ_k^(n,c) · scale(n,c)
    #
    # This is the derivative of J^{kk}(t) = t·exp(μ_k·t) w.r.t. t evaluated
    # at the diagonal, contracted with D̃ and Ũ. More precisely, it's the
    # diagonal (k=l) contribution of ∂/∂t_n [Σ_{kl} D̃_k J^{kl} Ũ_l · scale].

    # exp(μ_k · t_n): (*H, R, A)
    exp_mu_t = jnp.exp(mu[..., None, :] * distances[..., None])

    # μ_k · exp(μ_k · t_n): (*H, R, A) — derivative of exp(μ_k·t) w.r.t. t
    mu_exp = mu[..., None, :] * exp_mu_t  # (*H, R, A)

    # Scale factors: exp(logNormD + logNormU - logLike) for branches 1..R-1
    log_scale = logNormD[..., 1:, :] + logNormU[..., 1:, :] - logLike[..., None, :]
    scale = jnp.exp(log_scale)  # (*H, R-1, C)

    # Non-root branches only
    Dt = D_tilde[..., 1:, :, :]   # (*H, R-1, C, A)
    Ut = U_tilde[..., 1:, :, :]   # (*H, R-1, C, A)
    mu_exp_nr = mu_exp[..., 1:, :]  # (*H, R-1, A)

    # per_branch[n,c] = Σ_k Dt[n,c,k] * mu_exp[n,k] * Ut[n,c,k]
    per_branch = jnp.einsum('...nck,...nk,...nck->...nc', Dt, mu_exp_nr, Ut)
    per_branch = per_branch * scale  # (*H, R-1, C)
    if is_irrev:
        per_branch = per_branch.real

    # Contract with upstream gradient g over columns: Σ_c g_c * per_branch[n,c]
    dist_grad_nonroot = jnp.einsum('...nc,...c->...n', per_branch, g)  # (*H, R-1)

    # Pad with 0 for root (root has no branch)
    root_shape = (*dist_grad_nonroot.shape[:-1], 1)
    dist_grad = jnp.concatenate(
        [jnp.zeros(root_shape, dtype=dist_grad_nonroot.dtype), dist_grad_nonroot],
        axis=-1,
    )  # (*H, R)

    # If there are H batch dims, sum them out — distances is flat (R,)
    while dist_grad.ndim > distances.ndim:
        dist_grad = dist_grad.sum(axis=0)

    if is_irrev:
        dist_grad = dist_grad.real

    return (dist_grad,)


_loglike_custom_distances.defvjp(_fwd, _bwd)
