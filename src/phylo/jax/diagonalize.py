import jax.numpy as jnp
from .types import DiagModel, IrrevDiagModel, RateModel


def diagonalize_rate_matrix(subRate, rootProb):
    """Eigendecompose a reversible rate matrix.

    S_ij = R_ij * sqrt(pi_i / pi_j) is symmetric.
    S = V diag(mu) V^T.

    Args:
        subRate: (*H, A, A) rate matrix
        rootProb: (*H, A) equilibrium distribution

    Returns:
        DiagModel(eigenvalues, eigenvectors, pi)
    """
    sqrt_pi = jnp.sqrt(rootProb)
    inv_sqrt_pi = 1.0 / sqrt_pi
    # S_ij = R_ij * sqrt(pi_i / pi_j) = R_ij * sqrt_pi_i * inv_sqrt_pi_j
    S = subRate * sqrt_pi[..., :, None] * inv_sqrt_pi[..., None, :]
    # Symmetrize to clean up numerical noise
    S = 0.5 * (S + jnp.swapaxes(S, -2, -1))
    eigenvalues, eigenvectors = jnp.linalg.eigh(S)
    return DiagModel(eigenvalues=eigenvalues, eigenvectors=eigenvectors, pi=rootProb)


def reconstruct_rate_matrix(model):
    """Reconstruct rate matrix from DiagModel. For testing.

    R_ij = sqrt(pi_j/pi_i) * sum_k v_ik * mu_k * v_jk

    Returns:
        RateModel(subRate, rootProb)
    """
    V = model.eigenvectors   # (*H, A, A)
    mu = model.eigenvalues   # (*H, A)
    pi = model.pi            # (*H, A)
    sqrt_pi = jnp.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi
    # S = V diag(mu) V^T
    S = jnp.einsum('...ak,...k,...bk->...ab', V, mu, V)
    # R_ij = S_ij * sqrt(pi_j / pi_i) = S_ij * inv_sqrt_pi_i * sqrt_pi_j
    subRate = S * inv_sqrt_pi[..., :, None] * sqrt_pi[..., None, :]
    return RateModel(subRate=subRate, rootProb=pi)


def diagonalize_irreversible(subRate, rootProb):
    """Eigendecompose an irreversible rate matrix.

    R = V diag(mu) V^{-1} directly (no symmetrization).

    Args:
        subRate: (*H, A, A) rate matrix
        rootProb: (*H, A) stationary distribution

    Returns:
        IrrevDiagModel(eigenvalues, eigenvectors, eigenvectors_inv, pi)
    """
    eigenvalues, eigenvectors = jnp.linalg.eig(subRate)
    eigenvectors_inv = jnp.linalg.inv(eigenvectors)
    return IrrevDiagModel(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigenvectors_inv=eigenvectors_inv,
        pi=rootProb,
    )


def compute_sub_matrices_irrev(model, distances):
    """Compute substitution probability matrices M(t) for irreversible model.

    M_ij(t) = Re(sum_k V_ik exp(mu_k t) V_inv_kj)

    Args:
        model: IrrevDiagModel
        distances: (R,) branch lengths

    Returns:
        (*H, R, A, A) substitution matrices (real)
    """
    V = model.eigenvectors       # (*H, A, A) complex
    V_inv = model.eigenvectors_inv  # (*H, A, A) complex
    mu = model.eigenvalues       # (*H, A) complex

    # exp(mu_k * t_r): (*H, R, A) complex
    exp_mu_t = jnp.exp(
        mu[..., None, :] * distances[..., None]
    )

    # M_ij(t) = sum_k V_ik * exp(mu_k*t) * V_inv_kj
    M = jnp.einsum('...ak,...rk,...kj->...raj', V, exp_mu_t, V_inv)

    return M.real


def compute_sub_matrices(model, distances):
    """Compute substitution probability matrices M(t) from eigendecomposition.

    M_ij(t) = sqrt(pi_j/pi_i) * sum_k v_ik * exp(mu_k * t) * v_jk

    Args:
        model: DiagModel
        distances: (R,) branch lengths

    Returns:
        (*H, R, A, A) substitution matrices
    """
    V = model.eigenvectors   # (*H, A, A)
    mu = model.eigenvalues   # (*H, A)
    pi = model.pi            # (*H, A)

    # exp(mu_k * t_r): (*H, R, A)
    # mu is (*H, A), distances is (R,). We need (*H, R, A).
    # Broadcast: mu[..., None, :] is (*H, 1, A), distances[None, ..., None] handles dims
    exp_mu_t = jnp.exp(
        mu[..., None, :] * distances[..., None]
    )  # (*H, R, A) — mu broadcasts over R, distances broadcasts over A

    # S(t) = V diag(exp(mu*t)) V^T: sum_k v_ak exp(mu_k t) v_bk
    # (*H, R, A, A)
    S_t = jnp.einsum('...ak,...rk,...bk->...rab', V, exp_mu_t, V)

    # M_ij(t) = sqrt(pi_j/pi_i) * S_ij(t)
    sqrt_pi = jnp.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi
    M = S_t * inv_sqrt_pi[..., None, :, None] * sqrt_pi[..., None, None, :]
    return M
