from __future__ import annotations
import jax.numpy as jnp
from .types import DiagModel, IrrevDiagModel, AnyDiagModel
from .diagonalize import diagonalize_irreversible, diagonalize_rate_matrix_auto


def hky85_diag(kappa: float, pi: jnp.ndarray) -> DiagModel:
    """HKY85 model with closed-form eigendecomposition.

    Args:
        kappa: transition/transversion ratio (scalar)
        pi: (4,) equilibrium frequencies [A, C, G, T]

    Returns:
        DiagModel with closed-form eigenvalues and eigenvectors
    """
    pi = jnp.asarray(pi, dtype=jnp.float64)
    pi_A, pi_C, pi_G, pi_T = pi[0], pi[1], pi[2], pi[3]
    pi_R = pi_A + pi_G  # purines
    pi_Y = pi_C + pi_T  # pyrimidines

    # Normalization: expected rate = 1
    # -sum_i pi_i * Q_ii = 1 gives:
    beta = 1.0 / (2.0 * pi_R * pi_Y + 2.0 * kappa * (pi_A * pi_G + pi_C * pi_T))

    # Eigenvalues of Q (verified analytically):
    mu_0 = 0.0
    mu_1 = -beta
    mu_2 = -beta * (pi_R + kappa * pi_Y)   # within-pyrimidine mode
    mu_3 = -beta * (pi_Y + kappa * pi_R)   # within-purine mode
    eigenvalues = jnp.array([mu_0, mu_1, mu_2, mu_3])

    # Eigenvectors of symmetrized matrix S = diag(sqrt(pi)) Q diag(1/sqrt(pi)).
    # If Q*v = lambda*v (right eigenvector), then S*w = lambda*w where w = diag(sqrt(pi))*v.
    sqrt_pi = jnp.sqrt(pi)

    # v_0 = (1,1,1,1) right eigenvector of Q for lambda=0 => w_0 = sqrt(pi)
    w0 = sqrt_pi

    # v_1 = (pi_Y, -pi_R, pi_Y, -pi_R) for lambda_1 = -beta
    w1 = jnp.array([
        jnp.sqrt(pi_A) * pi_Y,
        -jnp.sqrt(pi_C) * pi_R,
        jnp.sqrt(pi_G) * pi_Y,
        -jnp.sqrt(pi_T) * pi_R,
    ])
    # ||w1||^2 = pi_R * pi_Y
    w1 = w1 / jnp.sqrt(pi_R * pi_Y)

    # v_2 = (0, pi_T, 0, -pi_C) for lambda_2 = -beta*(pi_R + kappa*pi_Y)
    w2 = jnp.array([
        0.0,
        jnp.sqrt(pi_C) * pi_T,
        0.0,
        -jnp.sqrt(pi_T) * pi_C,
    ])
    # ||w2||^2 = pi_C * pi_T * pi_Y
    w2 = w2 / jnp.sqrt(pi_C * pi_T * pi_Y)

    # v_3 = (pi_G, 0, -pi_A, 0) for lambda_3 = -beta*(pi_Y + kappa*pi_R)
    w3 = jnp.array([
        jnp.sqrt(pi_A) * pi_G,
        0.0,
        -jnp.sqrt(pi_G) * pi_A,
        0.0,
    ])
    # ||w3||^2 = pi_A * pi_G * pi_R
    w3 = w3 / jnp.sqrt(pi_A * pi_G * pi_R)

    # Stack: eigenvectors[a, k] = component a of eigenvector k
    eigenvectors = jnp.stack([w0, w1, w2, w3], axis=-1)  # (4, 4)

    return DiagModel(eigenvalues=eigenvalues, eigenvectors=eigenvectors, pi=pi)


def jukes_cantor_model(A: int) -> DiagModel:
    """Jukes-Cantor model for an A-state alphabet.

    R_ij = mu * (1/A) for i!=j, normalized so expected rate = 1.
    mu = A/(A-1).

    Returns:
        DiagModel
    """
    pi = jnp.ones(A) / A
    mu = A / (A - 1.0)
    # Eigenvalues: 0 (once), -mu (A-1 times)
    eigenvalues = jnp.concatenate([jnp.zeros(1), jnp.full(A - 1, -mu)])
    # Eigenvectors: v^(0) = sqrt(pi) = 1/sqrt(A) * ones
    # Remaining eigenvectors: any orthonormal basis of the subspace orthogonal to ones
    # Use Householder or Gram-Schmidt on identity columns
    v0 = jnp.ones(A) / jnp.sqrt(A)
    # Build orthonormal complement via QR of [v0 | I]
    basis = jnp.eye(A)
    Q, _ = jnp.linalg.qr(jnp.concatenate([v0[:, None], basis], axis=1))
    eigenvectors = Q[:, :A]  # (A, A), first column is v0
    return DiagModel(eigenvalues=eigenvalues, eigenvectors=eigenvectors, pi=pi)


def f81_model(pi: jnp.ndarray) -> DiagModel:
    """F81 model: R_ij = mu * pi_j for i!=j, normalized to expected rate = 1.

    Args:
        pi: (A,) equilibrium frequencies

    Returns:
        DiagModel
    """
    pi = jnp.asarray(pi, dtype=jnp.float64)
    A = pi.shape[0]
    # mu = 1 / (1 - sum(pi^2))
    mu = 1.0 / (1.0 - jnp.sum(pi ** 2))
    # Eigenvalues: 0 (once), -mu (A-1 times)
    eigenvalues = jnp.concatenate([jnp.zeros(1), jnp.full(A - 1, -mu)])
    # Eigenvectors of S where S_ij = R_ij * sqrt(pi_i/pi_j)
    # v^(0) = sqrt(pi)
    sqrt_pi = jnp.sqrt(pi)
    # Remaining eigenvectors: orthonormal in subspace perp to sqrt(pi)
    basis = jnp.eye(A)
    augmented = jnp.concatenate([sqrt_pi[:, None], basis], axis=1)
    Q, _ = jnp.linalg.qr(augmented)
    eigenvectors = Q[:, :A]  # (A, A)
    return DiagModel(eigenvalues=eigenvalues, eigenvectors=eigenvectors, pi=pi)


def gamma_rate_categories(alpha: float, K: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Yang (1994) discretized gamma rate categories using quantile medians.

    Args:
        alpha: shape parameter (scalar)
        K: number of categories

    Returns:
        (rates, weights): each (K,) — rates are mean-normalized, weights are uniform
    """
    from jax.scipy.special import gammainc
    # Quantile boundaries
    boundaries = jnp.linspace(0, 1, K + 1)
    midpoints = 0.5 * (boundaries[:-1] + boundaries[1:])
    rates = _gamma_quantiles(alpha, midpoints)
    # Normalize so mean rate = 1
    rates = rates * K / jnp.sum(rates)
    weights = jnp.ones(K) / K
    return rates, weights


def _gamma_quantiles(alpha: float, probs: jnp.ndarray) -> jnp.ndarray:
    """Compute quantiles of Gamma(alpha, 1/alpha) distribution via bisection."""
    from jax.scipy.special import gammainc
    lo = jnp.zeros_like(probs)
    hi = jnp.ones_like(probs) * jnp.maximum(50.0, 10.0 / alpha)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        cdf = gammainc(alpha, mid * alpha)
        lo = jnp.where(cdf < probs, mid, lo)
        hi = jnp.where(cdf >= probs, mid, hi)
    return 0.5 * (lo + hi)


def scale_model(model: DiagModel, rate_multiplier: float | jnp.ndarray) -> DiagModel:
    """Scale eigenvalues by a rate multiplier.

    Args:
        model: DiagModel
        rate_multiplier: scalar or (K,) array

    Returns:
        DiagModel with scaled eigenvalues. If rate_multiplier is (K,),
        adds K as a leading batch dimension.
    """
    rate_multiplier = jnp.asarray(rate_multiplier)
    if rate_multiplier.ndim == 0:
        return DiagModel(
            eigenvalues=model.eigenvalues * rate_multiplier,
            eigenvectors=model.eigenvectors,
            pi=model.pi,
        )
    # rate_multiplier is (K,) — broadcast to add K as leading dim
    K = rate_multiplier.shape[0]
    eigenvalues = model.eigenvalues[None, ...] * rate_multiplier[:, None]  # (K, *H, A)
    eigenvectors = jnp.broadcast_to(
        model.eigenvectors[None, ...],
        (K, *model.eigenvectors.shape)
    )
    pi = jnp.broadcast_to(model.pi[None, ...], (K, *model.pi.shape))
    return DiagModel(eigenvalues=eigenvalues, eigenvectors=eigenvectors, pi=pi)


def irrev_model_from_rate_matrix(subRate: jnp.ndarray, pi: jnp.ndarray) -> IrrevDiagModel:
    """Construct an IrrevDiagModel from a (possibly irreversible) rate matrix.

    Always uses irreversible decomposition (eig, not eigh).

    Args:
        subRate: (*H, A, A) rate matrix
        pi: (*H, A) stationary distribution

    Returns:
        IrrevDiagModel
    """
    subRate = jnp.asarray(subRate, dtype=jnp.float64)
    pi = jnp.asarray(pi, dtype=jnp.float64)
    return diagonalize_irreversible(subRate, pi)


def model_from_rate_matrix(
    subRate: jnp.ndarray, pi: jnp.ndarray,
    reversible: bool | None = None,
    tol: float = 1e-10,
) -> AnyDiagModel:
    """Construct a diagonalized model from a rate matrix.

    Auto-detects reversibility via detailed balance when reversible=None.

    Args:
        subRate: (*H, A, A) rate matrix
        pi: (*H, A) stationary distribution
        reversible: True → reversible (eigh), False → irreversible (eig),
                    None → auto-detect via detailed balance check
        tol: tolerance for detailed balance check when reversible=None

    Returns:
        DiagModel if reversible, IrrevDiagModel if irreversible.
    """
    subRate = jnp.asarray(subRate, dtype=jnp.float64)
    pi = jnp.asarray(pi, dtype=jnp.float64)
    return diagonalize_rate_matrix_auto(subRate, pi, reversible=reversible, tol=tol)
