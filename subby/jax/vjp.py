"""Custom VJP for LogLike — fast distance gradients via Fisher identity.

Two methods are provided:

1. **eigen** (default): The gradient of log-likelihood w.r.t. branch length t_n is:
     ∂logL_c/∂t_n = Σ_k D̃_k^(n,c) · μ_k · exp(μ_k·t_n) · Ũ_k^(n,c) · scale(n,c)
   This uses eigenbasis projection of the inside/outside vectors.

2. **pade**: Uses the identity ∂M(t)/∂t = Q·M(t) to compute:
     ∂logL_c/∂t_n = scale · Σ_a D_a · Σ_b (Q·M(t_n))_ab · U_b
   The forward pass computes M(t) via JAX's Padé matrix exponential
   (jax.scipy.linalg.expm), avoiding eigendecomposition entirely.

Both methods have the same asymptotic complexity O(CRA²) for the backward
pass. They differ only in constant factors and numerical characteristics.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .types import Tree, DiagModel, IrrevDiagModel, RateModel, AnyDiagModel
from .diagonalize import (
    compute_sub_matrices, compute_sub_matrices_irrev, reconstruct_rate_matrix,
)
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


# ---------------------------------------------------------------------------
# Padé-based custom VJP (no eigendecomposition)
# ---------------------------------------------------------------------------


def _extract_rate_matrix(model):
    """Extract the rate matrix Q and pi from any model type.

    Returns:
        (Q, pi) where Q is (*H, A, A) and pi is (*H, A).
    """
    if isinstance(model, RateModel):
        return model.subRate, model.rootProb
    elif isinstance(model, IrrevDiagModel):
        Q = jnp.einsum(
            '...ik,...k,...kj->...ij',
            model.eigenvectors, model.eigenvalues, model.eigenvectors_inv,
        ).real
        return Q, model.pi
    elif isinstance(model, DiagModel):
        rm = reconstruct_rate_matrix(model)
        return rm.subRate, rm.rootProb
    else:
        raise ValueError(f"Cannot extract rate matrix from {type(model)}")


def _compute_sub_matrices_pade(Q, distances):
    """Compute substitution matrices M(t) = expm(Q*t) via Padé approximation.

    Args:
        Q: (*H, A, A) rate matrix
        distances: (R,) branch lengths

    Returns:
        (*H, R, A, A) substitution matrices
    """
    # Q*t for each branch: (*H, R, A, A)
    Q_t = Q[..., None, :, :] * distances[..., None, None]

    # Flatten batch+branch dims, apply expm, reshape back
    orig_shape = Q_t.shape
    A = orig_shape[-1]
    flat = Q_t.reshape(-1, A, A)
    M_flat = jax.vmap(jax.scipy.linalg.expm)(flat)
    return M_flat.reshape(orig_shape)


def make_loglike_pade_grad(Q, pi, alignment, parentIndex, maxChunkSize=128):
    """Create a LogLike function using Padé expm with custom VJP.

    Forward pass: M(t) = expm(Q*t) via JAX's Padé approximation.
    Backward pass: ∂logL/∂t_n = scale · D^T · (Q·M(t_n)) · U.

    Avoids eigendecomposition entirely. Same asymptotic complexity as the
    eigenbasis approach O(CRA²), but uses different matrix operations
    that may have different constant factors.

    Args:
        Q: (A, A) or (*H, A, A) rate matrix
        pi: (A,) or (*H, A) equilibrium distribution
        alignment: (R, C) int32 tokens
        parentIndex: (R,) parent indices
        maxChunkSize: chunk columns for memory

    Returns:
        A function f(distances) -> logLike with custom VJP.
    """

    @jax.custom_vjp
    def f(distances):
        tree = Tree(parentIndex=parentIndex, distanceToParent=distances)
        subMatrices = _compute_sub_matrices_pade(Q, distances)
        _, _, logLike = upward_pass(alignment, tree, subMatrices, pi, maxChunkSize)
        return logLike

    def f_fwd(distances):
        tree = Tree(parentIndex=parentIndex, distanceToParent=distances)
        subMatrices = _compute_sub_matrices_pade(Q, distances)
        U, logNormU, logLike = upward_pass(
            alignment, tree, subMatrices, pi, maxChunkSize
        )
        residuals = (distances, U, logNormU, logLike, subMatrices)
        return logLike, residuals

    def f_bwd(residuals, g):
        """Backward pass: compute distance gradient via Q·M(t) contraction.

        ∂logL_c/∂t_n = scale(n,c) · Σ_a D_a^(n,c) · (Q·M(t_n)·U^(n,c))_a

        g: (*H, C) upstream gradient (cotangent for logLike)
        """
        distances, U, logNormU, logLike, subMatrices = residuals
        tree = Tree(parentIndex=parentIndex, distanceToParent=distances)

        # Downward pass to get outside vectors
        D, logNormD = downward_pass(U, logNormU, tree, subMatrices, pi, alignment)

        # Non-root branches only
        Mt = subMatrices[..., 1:, :, :]   # (*H, R-1, A, A)
        Dt = D[..., 1:, :, :]             # (*H, R-1, C, A)
        Ut = U[..., 1:, :, :]             # (*H, R-1, C, A)

        # Step 1: MU = M(t_n) @ U for each branch and column
        # MU[..., n, c, a] = Σ_b M[..., n, a, b] · U[..., n, c, b]
        MU = jnp.einsum('...nab,...ncb->...nca', Mt, Ut)  # (*H, R-1, C, A)

        # Step 2: QMU = Q @ MU
        # QMU[..., n, c, a] = Σ_b Q[..., a, b] · MU[..., n, c, b]
        QMU = jnp.einsum('...ab,...ncb->...nca', Q, MU)   # (*H, R-1, C, A)

        # Scale factors: exp(logNormD + logNormU - logLike) for branches 1..R-1
        log_scale = (
            logNormD[..., 1:, :] + logNormU[..., 1:, :] - logLike[..., None, :]
        )
        scale = jnp.exp(log_scale)  # (*H, R-1, C)

        # per_branch[n,c] = Σ_a D[n,c,a] · QMU[n,c,a] · scale[n,c]
        per_branch = jnp.sum(Dt * QMU, axis=-1) * scale  # (*H, R-1, C)

        # Contract with upstream gradient g over columns
        dist_grad_nonroot = jnp.einsum('...nc,...c->...n', per_branch, g)

        # Pad with 0 for root
        root_shape = (*dist_grad_nonroot.shape[:-1], 1)
        dist_grad = jnp.concatenate(
            [jnp.zeros(root_shape, dtype=dist_grad_nonroot.dtype),
             dist_grad_nonroot],
            axis=-1,
        )

        # Sum out H batch dims if present
        while dist_grad.ndim > distances.ndim:
            dist_grad = dist_grad.sum(axis=0)

        return (dist_grad,)

    f.defvjp(f_fwd, f_bwd)
    return f
