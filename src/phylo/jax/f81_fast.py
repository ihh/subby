import jax.numpy as jnp


def f81_counts(U, D, logNormU, logNormD, logLike, distances, pi, parentIndex):
    """O(CRA^2) direct computation of expected counts for F81/JC models.

    For F81: M_ij(t) = delta_ij * e^{-mu*t} + pi_j * (1 - e^{-mu*t})

    Uses closed-form I^{ab}_{ij}(t) integrals to avoid eigenbasis.

    Args:
        U: (*H, R, C, A) inside vectors (rescaled)
        D: (*H, R, C, A) outside vectors (rescaled)
        logNormU: (*H, R, C) per-node inside log-normalizers
        logNormD: (*H, R, C) per-node outside log-normalizers
        logLike: (*H, C) log-likelihoods
        distances: (R,) branch lengths
        pi: (*H, A) equilibrium frequencies
        parentIndex: (R,) parent indices

    Returns:
        (*H, A, A, C) counts tensor (diag=dwell, off-diag=substitutions)
    """
    *H, R, C, A = U.shape

    # mu = 1 / (1 - sum(pi^2)) for F81
    mu = 1.0 / (1.0 - jnp.sum(pi ** 2, axis=-1))  # (*H,)

    # Per-branch quantities for nodes 1..R-1
    t = distances[1:]  # (R-1,)
    mu_t = mu[..., None] * t  # (*H, R-1)
    e_t = jnp.exp(-mu_t)     # (*H, R-1)  = e^{-mu*t}
    p = 1.0 - e_t            # (*H, R-1)  = 1 - e^{-mu*t}

    # alpha = t * e_t
    # beta = p/mu - t*e_t = (1-e_t)/mu - t*e_t
    # gamma = t*(1+e_t) - 2*p/mu = t + t*e_t - 2*(1-e_t)/mu
    alpha = t * e_t                                          # (*H, R-1)
    beta = p / mu[..., None] - t * e_t                       # (*H, R-1)
    gamma = t * (1.0 + e_t) - 2.0 * p / mu[..., None]       # (*H, R-1)

    # Scale factor per branch per column
    log_scale = (
        logNormD[..., 1:, :] + logNormU[..., 1:, :] - logLike[..., None, :]
    )  # (*H, R-1, C)
    scale = jnp.exp(log_scale)  # (*H, R-1, C)

    # Per-branch inside/outside vectors (nodes 1..R-1)
    D_b = D[..., 1:, :, :]  # (*H, R-1, C, A)
    U_b = U[..., 1:, :, :]  # (*H, R-1, C, A)

    # Scaled D and U: multiply by scale
    # D_i = D_b[..., i] * scale
    D_scaled = D_b * scale[..., None]  # (*H, R-1, C, A)
    U_scaled = U_b                      # (*H, R-1, C, A), not scaled yet (scale already in D)

    # Key aggregates per branch per column:
    # piU = sum_b pi_b * U_b:  (*H, R-1, C)
    piU = jnp.einsum('...a,...nca->...nc', pi, U_scaled)

    # Dsum = sum_a D_a (scaled):  (*H, R-1, C)
    Dsum = jnp.sum(D_scaled, axis=-1)

    # sum_{ab} D_a U_b I^{ab}_{ij}(t)
    # = alpha * D_i * U_j + beta * (D_i * piU + pi_i * Dsum * U_j) + gamma * pi_i * Dsum * piU

    # For dwell (i == j): result[i,i,c] = (1/P(x)) * sum_n above with i=j
    # For subs (i != j): result[i,j,c] = (mu*pi_j/P(x)) * sum_n above

    # Term 1: alpha * D_i * U_j  -> sum over branches -> (*H, A, A, C)
    # alpha is (*H, R-1), D_scaled is (*H, R-1, C, A), U_scaled is (*H, R-1, C, A)
    term1 = jnp.einsum('...n,...nci,...ncj->...ijc', alpha, D_scaled, U_scaled)

    # Term 2: beta * (D_i * piU + pi_i * Dsum * U_j)
    # Part a: beta * D_i * piU -> sum_n beta * D_scaled_i * piU
    term2a = jnp.einsum('...n,...nci,...nc->...ic', beta, D_scaled, piU)  # (*H, A, C)
    # Part b: beta * pi_i * Dsum * U_j -> sum_n beta * pi_i * Dsum * U_j
    term2b = jnp.einsum('...n,...i,...nc,...ncj->...ijc', beta, pi, Dsum, U_scaled)
    # term2a contributes to all j equally for fixed i (broadcast)
    term2 = term2a[..., :, None, :] + term2b  # (*H, A, A, C)

    # Term 3: gamma * pi_i * Dsum * piU
    term3_scalar = jnp.einsum('...n,...nc,...nc->...c', gamma, Dsum, piU)  # (*H, C)
    term3 = pi[..., :, None, None] * term3_scalar[..., None, None, :]   # (*H, A, 1, C)
    # broadcast to (*H, A, A, C) — same for all j
    # Actually term3 contributes equally to all j, so need to think about this
    # gamma * pi_i * Dsum * piU is independent of j
    # For dwell (i=j): contributes to diag
    # For subs (i!=j): gets multiplied by mu*pi_j
    # So we keep term3 as (*H, A, C) = pi_i * scalar
    term3_per_i = pi[..., :, None] * term3_scalar[..., None, :]  # (*H, A, C)

    # Build I_sum = sum_n sum_{ab} D_a U_b I^{ab}_{ij}
    # For the diagonal part (dwell time w_i):
    # w_i = sum_{ab} D_a U_b I^{ab}_{ii} = alpha*D_i*U_i + beta*(D_i*piU + pi_i*Dsum*U_i) + gamma*pi_i*Dsum*piU
    # Extract diagonal of term1:
    diag_term1 = jnp.einsum('...iic->...ic', term1)  # (*H, A, C)
    diag_term2 = jnp.einsum('...iic->...ic', term2)  # (*H, A, C)
    dwell = diag_term1 + diag_term2 + term3_per_i     # (*H, A, C)

    # For the off-diagonal (substitution u_{ij}):
    # u_{ij} = mu * pi_j * [alpha*D_i*U_j + beta*(D_i*piU + pi_i*Dsum*U_j) + gamma*pi_i*Dsum*piU]
    I_sum_ij = term1 + term2 + term3_per_i[..., :, None, :]  # (*H, A, A, C) — approximate
    # Actually term3 is the same for all j, so I_sum_ij[i,j,c] for i!=j:
    # = term1[i,j] + term2[i,j] + term3_per_i[i]
    # We already have this in I_sum_ij (term3_per_i broadcast to all j)

    subs = mu[..., None, None, None] * pi[..., None, :, None] * I_sum_ij  # (*H, A, A, C)

    # Combine: diagonal = dwell, off-diagonal = subs
    diag_mask = jnp.eye(A, dtype=bool)
    counts = jnp.where(
        diag_mask[..., None],
        dwell[..., :, None, :] * jnp.eye(A)[..., None],  # place dwell on diagonal
        subs,
    )

    # More carefully: build the result tensor
    # result[i, j, c] = dwell[i, c] if i == j, else subs[i, j, c]
    result = subs * (1.0 - jnp.eye(A)[..., None])  # zero out diagonal of subs
    # Add dwell on diagonal
    result = result + jnp.einsum('...ic,ij->...ijc', dwell, jnp.eye(A))

    return result
