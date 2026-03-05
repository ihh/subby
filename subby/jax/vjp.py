"""Custom VJP for LogLike — fast distance gradients via Fisher identity.

The gradient of log-likelihood w.r.t. branch length t_n is:
  ∂logL_c/∂t_n = Σ_k D̃_k^(n,c) · μ_k · exp(μ_k·t_n) · Ũ_k^(n,c) · scale(n,c)

This avoids tracing through the full computation graph by reusing the
upward_pass from the forward pass and only adding the downward_pass +
eigenbasis projection in the backward pass.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import Tree, DiagModel, IrrevDiagModel, AnyDiagModel
from .diagonalize import compute_sub_matrices, compute_sub_matrices_irrev
from .pruning import upward_pass
from .outside import downward_pass
from .eigensub import eigenbasis_project, eigenbasis_project_irrev


def _is_irrev(model):
    return isinstance(model, IrrevDiagModel)


def make_loglike_custom_grad(model, alignment, parentIndex, maxChunkSize=128):
    """Create a LogLike function with custom VJP, closed over non-differentiable args.

    Returns a function f(distances) -> logLike with custom backward pass.
    """
    is_irrev = _is_irrev(model)

    @jax.custom_vjp
    def f(distances):
        tree = Tree(parentIndex=parentIndex, distanceToParent=distances)
        if is_irrev:
            subMatrices = compute_sub_matrices_irrev(model, distances)
        else:
            subMatrices = compute_sub_matrices(model, distances)
        _, _, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
        return logLike

    def f_fwd(distances):
        tree = Tree(parentIndex=parentIndex, distanceToParent=distances)
        if is_irrev:
            subMatrices = compute_sub_matrices_irrev(model, distances)
        else:
            subMatrices = compute_sub_matrices(model, distances)
        U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
        residuals = (distances, U, logNormU, logLike, subMatrices)
        return logLike, residuals

    def f_bwd(residuals, g):
        """Backward pass: compute distance gradient via eigenbasis projection.

        g: (*H, C) upstream gradient (cotangent for logLike)
        """
        distances, U, logNormU, logLike, subMatrices = residuals
        tree = Tree(parentIndex=parentIndex, distanceToParent=distances)

        # Downward pass to get outside vectors
        D, logNormD = downward_pass(U, logNormU, tree, subMatrices, model.pi, alignment)

        # Project into eigenbasis for distance gradient.
        # For the distance gradient ∂logL/∂t_n, we need:
        #   Σ_k μ_k exp(μ_k t) · [Σ_a D_a V_{ak}] · [Σ_b V^{-1}_{kb} U_b]
        # For reversible models (V orthogonal, V^{-1} = V^T with sqrt(pi) weighting),
        # this equals Σ_k D̃_k · μ_k · exp(μ_k t) · Ũ_k from eigenbasis_project.
        # For irreversible models, the projections are swapped relative to
        # eigenbasis_project_irrev (which uses V^{-1} for D and V for U).
        if is_irrev:
            V = model.eigenvectors          # (*H, A, A) complex
            V_inv = model.eigenvectors_inv  # (*H, A, A) complex
            # D_proj_k = Σ_a D_a V_{ak}  (project D with right eigenvectors)
            D_tilde = jnp.einsum('...rca,...ak->...rck', D, V)
            # U_proj_k = Σ_b V^{-1}_{kb} U_b  (project U with left eigenvectors)
            U_tilde = jnp.einsum('...kb,...rcb->...rck', V_inv, U)
            mu = model.eigenvalues  # (*H, A) complex
        else:
            U_tilde, D_tilde = eigenbasis_project(U, D, model)
            mu = model.eigenvalues  # (*H, A) real

        # Per-branch distance gradient:
        # ∂logL_c/∂t_n = Σ_k D_proj_k · μ_k · exp(μ_k·t_n) · U_proj_k · scale(n,c)

        # exp(μ_k · t_n): (*H, R, A)
        exp_mu_t = jnp.exp(mu[..., None, :] * distances[..., None])

        # μ_k · exp(μ_k · t_n): (*H, R, A)
        mu_exp = mu[..., None, :] * exp_mu_t

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

        # Contract with upstream gradient g over columns
        dist_grad_nonroot = jnp.einsum('...nc,...c->...n', per_branch, g)  # (*H, R-1)

        # Pad with 0 for root
        root_shape = (*dist_grad_nonroot.shape[:-1], 1)
        dist_grad = jnp.concatenate(
            [jnp.zeros(root_shape, dtype=dist_grad_nonroot.dtype), dist_grad_nonroot],
            axis=-1,
        )

        # Sum out H batch dims if present
        while dist_grad.ndim > distances.ndim:
            dist_grad = dist_grad.sum(axis=0)

        if is_irrev:
            dist_grad = dist_grad.real

        return (dist_grad,)

    f.defvjp(f_fwd, f_bwd)
    return f
