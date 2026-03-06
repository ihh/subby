"""Standalone CTMC branch integral functions.

Computes expected substitution counts and dwell times for a single
continuous-time Markov chain (CTMC) branch, independent of any alignment
or tree.

For a CTMC with rate matrix Q and branch length t:
    result[a, b, i, j] = E[N_{i->j}(t) | X(0)=a, X(t)=b]   (i != j)
    result[a, b, i, i] = E[T_i(t) | X(0)=a, X(t)=b]         (dwell time)
"""

from __future__ import annotations

import jax.numpy as jnp

from .types import DiagModel, IrrevDiagModel, RateModel, AnyDiagModel, AnyModel
from .diagonalize import diagonalize_rate_matrix_auto
from .eigensub import compute_J, compute_J_complex


def expected_counts_eigen(
    eigenvalues: jnp.ndarray,
    eigenvectors: jnp.ndarray,
    pi: jnp.ndarray,
    t: float,
) -> jnp.ndarray:
    """Expected counts from pre-computed eigendecomposition (reversible).

    For a reversible CTMC with symmetrized rate matrix S = V diag(mu) V^T
    and rate matrix Q = diag(1/sqrt(pi)) S diag(sqrt(pi)):

    W[a,b,i,j] = sum_{kl} P[a,i,k] * J[k,l](t) * P[b,j,l]
    where P[x,y,k] = V[x,k] * V[y,k]

    result[a,b,i,i] = W[a,b,i,i] / S_t[a,b]                    (dwell)
    result[a,b,i,j] = S[i,j] * W[a,b,i,j] / S_t[a,b]          (subs, i!=j)

    Args:
        eigenvalues: (*H, A) real eigenvalues of symmetrized rate matrix
        eigenvectors: (*H, A, A) real orthogonal eigenvectors
        pi: (*H, A) equilibrium distribution
        t: scalar branch length

    Returns:
        (*H, A, A, A, A) tensor: result[..., a, b, i, j]
        Diagonal [a,b,i,i] = dwell time; off-diagonal [a,b,i,j] = sub count.
    """
    mu = eigenvalues   # (*H, A)
    V = eigenvectors   # (*H, A, A)
    A = V.shape[-1]

    # J[k,l](t): (*H, A, A) — single branch
    distances = jnp.array([t])
    J = compute_J(mu, distances)  # (*H, 1, A, A)
    J = J[..., 0, :, :]          # (*H, A, A)

    # Symmetrized transition matrix: S_t[a,b] = sum_k V[a,k] exp(mu_k*t) V[b,k]
    exp_mu_t = jnp.exp(mu * t)  # (*H, A)
    S_t = jnp.einsum('...ak,...k,...bk->...ab', V, exp_mu_t, V)  # (*H, A, A)

    # W[a,b,i,j] = sum_{kl} V[a,k]*V[i,k] * J[k,l] * V[b,l]*V[j,l]
    # = sum_{kl} P[a,i,k] * J[k,l] * P[b,j,l]
    # where P[x,y,k] = V[x,k] * V[y,k]
    #
    # Compute: first contract J with V on both sides for the (i,k) and (j,l) indices
    # VJV[a,b,i,j] = sum_{kl} V[a,k] V[i,k] J[k,l] V[b,l] V[j,l]
    W = jnp.einsum('...ak,...ik,...kl,...bl,...jl->...abij', V, V, J, V, V)

    # Symmetrized rate matrix: S[i,j] = sum_k V[i,k] mu_k V[j,k]
    S = jnp.einsum('...ik,...k,...jk->...ij', V, mu, V)  # (*H, A, A)

    # Build result: dwell on diagonal, S[i,j]*W/M on off-diagonal
    diag_mask = jnp.eye(A, dtype=bool)

    # Safe division by S_t (not M): the correct denominator in the symmetrized basis
    S_t_safe = jnp.where(jnp.abs(S_t) < 1e-300, 1.0, S_t)
    S_t_mask = jnp.abs(S_t) >= 1e-300  # (*H, A, A) bool

    # result[a,b,i,j]:
    #   i==j: W[a,b,i,i] / S_t[a,b]
    #   i!=j: S[i,j] * W[a,b,i,j] / S_t[a,b]
    W_scaled = jnp.where(
        diag_mask[None, None, :, :],  # broadcast over (a,b)
        W,
        S[..., None, None, :, :] * W,  # S broadcasts over (a,b)
    )

    result = W_scaled / S_t_safe[..., :, :, None, None]
    result = jnp.where(S_t_mask[..., :, :, None, None], result, 0.0)

    return result


def expected_counts_eigen_irrev(
    eigenvalues: jnp.ndarray,
    eigenvectors: jnp.ndarray,
    eigenvectors_inv: jnp.ndarray,
    pi: jnp.ndarray,
    t: float,
) -> jnp.ndarray:
    """Expected counts from pre-computed eigendecomposition (irreversible).

    For an irreversible CTMC with Q = V diag(mu) V^{-1}:

    L[a,i,k] = V[a,k] * V_inv[k,i]
    R[b,j,l] = V[j,l] * V_inv[l,b]
    W[a,b,i,j] = sum_{kl} L[a,i,k] * J[k,l](t) * R[b,j,l]

    result[a,b,i,i] = W[a,b,i,i] / M[a,b]
    result[a,b,i,j] = Q[i,j] * W[a,b,i,j] / M[a,b]          (i!=j)

    Args:
        eigenvalues: (*H, A) complex eigenvalues
        eigenvectors: (*H, A, A) complex right eigenvectors V
        eigenvectors_inv: (*H, A, A) complex V^{-1}
        pi: (*H, A) real equilibrium distribution
        t: scalar branch length

    Returns:
        (*H, A, A, A, A) real tensor: result[..., a, b, i, j]
    """
    mu = eigenvalues       # (*H, A) complex
    V = eigenvectors       # (*H, A, A) complex
    V_inv = eigenvectors_inv  # (*H, A, A) complex
    A = V.shape[-1]

    # J[k,l](t): (*H, A, A) complex — single branch
    distances = jnp.array([t])
    J = compute_J_complex(mu, distances)  # (*H, 1, A, A)
    J = J[..., 0, :, :]                  # (*H, A, A)

    # Transition matrix: M[a,b] = Re(sum_k V[a,k] exp(mu_k*t) V_inv[k,b])
    exp_mu_t = jnp.exp(mu * t)  # (*H, A) complex
    M = jnp.einsum('...ak,...k,...kb->...ab', V, exp_mu_t, V_inv)  # complex
    M = M.real  # (*H, A, A)

    # W[a,b,i,j] = sum_{kl} V[a,k]*V_inv[k,i] * J[k,l] * V[j,l]*V_inv[l,b]
    W = jnp.einsum('...ak,...ki,...kl,...jl,...lb->...abij', V, V_inv, J, V, V_inv)

    # Rate matrix: Q[i,j] = Re(sum_k V[i,k] mu_k V_inv[k,j])
    Q = jnp.einsum('...ik,...k,...kj->...ij', V, mu, V_inv)  # complex

    # Build result
    diag_mask = jnp.eye(A, dtype=bool)

    M_safe = jnp.where(jnp.abs(M) < 1e-300, 1.0, M)
    M_mask = jnp.abs(M) >= 1e-300

    W_scaled = jnp.where(
        diag_mask[None, None, :, :],
        W,
        Q[..., None, None, :, :] * W,
    )

    result = W_scaled / M_safe[..., :, :, None, None]
    result = jnp.where(M_mask[..., :, :, None, None], result, 0.0)

    return result.real


def ExpectedCounts(model: AnyModel, t: float) -> jnp.ndarray:
    """Expected substitution counts and dwell times for a single CTMC branch.

    Computes E[N_{i->j}(t) | X(0)=a, X(t)=b] (off-diagonal) and
    E[T_i(t) | X(0)=a, X(t)=b] (diagonal) for all (a,b,i,j).

    Outer function: accepts DiagModel, IrrevDiagModel, or RateModel.
    Auto-diagonalizes RateModel. Dispatches to reversible or irreversible.

    Args:
        model: DiagModel, IrrevDiagModel, or RateModel
        t: scalar branch length

    Returns:
        (*H, A, A, A, A) tensor: result[..., a, b, i, j]
        Diagonal [a,b,i,i] = dwell time; off-diagonal [a,b,i,j] = sub count.
    """
    if isinstance(model, RateModel):
        model = diagonalize_rate_matrix_auto(model.subRate, model.rootProb)

    if isinstance(model, IrrevDiagModel):
        return expected_counts_eigen_irrev(
            model.eigenvalues, model.eigenvectors,
            model.eigenvectors_inv, model.pi, t,
        )
    else:
        return expected_counts_eigen(
            model.eigenvalues, model.eigenvectors, model.pi, t,
        )
