import jax.numpy as jnp


def compute_J(eigenvalues, distances):
    """Compute J^{kl}(T) interaction matrix between decay modes.

    J^{kl}(T) = T * exp(mu_k * T)           if mu_k ≈ mu_l
                (exp(mu_k*T) - exp(mu_l*T)) / (mu_k - mu_l)  otherwise

    Args:
        eigenvalues: (*H, A) eigenvalues
        distances: (R,) branch lengths

    Returns:
        (*H, R, A, A) J matrices per branch
    """
    mu = eigenvalues  # (*H, A)
    t = distances     # (R,)

    # mu_k * t_r: (*H, R, A)
    mu_t = mu[..., None, :] * t[..., None]  # broadcast: (*H, 1, A) * (R, 1) -> (*H, R, A)

    # exp(mu_k * t): (*H, R, A)
    exp_mu_t = jnp.exp(mu_t)

    # mu_k - mu_l: (*H, A, A)
    mu_diff = mu[..., :, None] - mu[..., None, :]  # (*H, A, A)

    # Degenerate case: J = T * exp(mu_k * T)
    # Non-degenerate: J = (exp(mu_k*T) - exp(mu_l*T)) / (mu_k - mu_l)

    # exp(mu_k*T) and exp(mu_l*T): (*H, R, A, 1) and (*H, R, 1, A)
    exp_k = exp_mu_t[..., :, None]   # (*H, R, A, 1)
    exp_l = exp_mu_t[..., None, :]   # (*H, R, 1, A)

    # mu_diff: (*H, 1, A, A) broadcast with t: (R, 1, 1)
    mu_diff_expanded = mu_diff[..., None, :, :]  # (*H, 1, A, A)

    # Non-degenerate case
    J_nondeg = (exp_k - exp_l) / (mu_diff_expanded + 1e-30)

    # Degenerate case: T * exp(mu_k * T)
    t_expanded = t[..., None, None]  # (R, 1, 1) -> broadcast to (*H, R, A, A)
    J_deg = t_expanded * exp_k

    # Use degenerate where |mu_k - mu_l| is small
    is_degenerate = jnp.abs(mu_diff_expanded) < 1e-8
    J = jnp.where(is_degenerate, J_deg, J_nondeg)

    return J


def eigenbasis_project(U, D, model):
    """Project inside/outside vectors into eigenbasis.

    U_tilde^(n)_l = sum_b U_b * v_{bl} * sqrt(pi_b)
    D_tilde^(n)_k = sum_a D_a * v_{ak} / sqrt(pi_a)

    Args:
        U: (*H, R, C, A) inside vectors (rescaled)
        D: (*H, R, C, A) outside vectors (rescaled)
        model: DiagModel

    Returns:
        U_tilde: (*H, R, C, A) projected inside
        D_tilde: (*H, R, C, A) projected outside
    """
    V = model.eigenvectors  # (*H, A, A)
    pi = model.pi           # (*H, A)
    sqrt_pi = jnp.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi

    # U_tilde_l = sum_b U_b * v_{bl} * sqrt(pi_b)
    # = sum_b (U_b * sqrt(pi_b)) * v_{bl}
    U_weighted = U * sqrt_pi[..., None, None, :]        # (*H, R, C, A)
    U_tilde = jnp.einsum('...rcb,...bk->...rck', U_weighted, V)  # (*H, R, C, A)

    # D_tilde_k = sum_a D_a * v_{ak} / sqrt(pi_a)
    D_weighted = D * inv_sqrt_pi[..., None, None, :]    # (*H, R, C, A)
    D_tilde = jnp.einsum('...rca,...ak->...rck', D_weighted, V)  # (*H, R, C, A)

    return U_tilde, D_tilde


def accumulate_C(D_tilde, U_tilde, J, logNormU, logNormD, logLike, parentIndex,
                 branch_mask=None):
    """Accumulate eigenbasis counts C_{kl} over branches.

    C_{kl,c} = sum_{n>0} D_tilde_k^(n) * J^{kl}(t_n) * U_tilde_l^(n) * scale[n,c]

    where scale[n,c] = exp(logNormD[n,c] + logNormU[n,c] - logLike[c])

    Args:
        D_tilde: (*H, R, C, A) projected outside vectors
        U_tilde: (*H, R, C, A) projected inside vectors
        J: (*H, R, A, A) J matrices per branch
        logNormU: (*H, R, C) per-node inside log-normalizers
        logNormD: (*H, R, C) per-node outside log-normalizers
        logLike: (*H, C) log-likelihoods
        parentIndex: (R,) parent indices
        branch_mask: (*H, R, C) bool or None — if provided, zeros out
            contributions from inactive branches

    Returns:
        C: (*H, A, A, C) eigenbasis counts
    """
    *H, R, C, A = U_tilde.shape

    # Scale factors per branch per column: exp(logNormD + logNormU - logLike)
    # Only for branches n > 0 (non-root)
    log_scale = (
        logNormD[..., 1:, :] + logNormU[..., 1:, :] - logLike[..., None, :]
    )  # (*H, R-1, C)
    scale = jnp.exp(log_scale)  # (*H, R-1, C)
    if branch_mask is not None:
        scale = scale * branch_mask[..., 1:, :]

    # Branch contributions: D_tilde_k * J_kl * U_tilde_l * scale
    # D_tilde for branches: (*H, R-1, C, A) — nodes 1..R-1
    Dt = D_tilde[..., 1:, :, :]  # (*H, R-1, C, A)
    Ut = U_tilde[..., 1:, :, :]  # (*H, R-1, C, A)
    Jb = J[..., 1:, :, :]        # (*H, R-1, A, A)

    # C_{kl,c} = sum_n (Dt_{n,c,k} * scale_{n,c}) * J_{n,k,l} * Ut_{n,c,l}
    # = sum_n Dt_scaled_{n,c,k} * J_{n,k,l} * Ut_{n,c,l}
    Dt_scaled = Dt * scale[..., None]  # (*H, R-1, C, A)

    C = jnp.einsum('...nck,...nkl,...ncl->...klc', Dt_scaled, Jb, Ut)

    return C


def back_transform(C, model):
    """Transform eigenbasis counts to natural basis.

    w_i = sum_{kl} v_{ik} * v_{il} * C_{kl}
    u_{ij} = S_{ij} * sum_{kl} v_{ik} * v_{jl} * C_{kl}   (i != j)

    Returns tensor where diag = dwell times, off-diag = substitution counts.

    Args:
        C: (*H, A, A, C) eigenbasis counts
        model: DiagModel

    Returns:
        (*H, A, A, C) counts tensor
    """
    V = model.eigenvectors  # (*H, A, A)
    pi = model.pi           # (*H, A)

    # VCV = V C V^T per column: (*H, A, A, C)
    # V^T C V in matrix notation, but here C is per-column
    # result_{ij,c} = sum_{kl} V_{ik} C_{kl,c} V_{jl}
    VCV = jnp.einsum('...ik,...klc,...jl->...ijc', V, C, V)

    # Dwell times: diag of VCV
    # Substitution counts: S_{ij} * VCV_{ij} for i != j
    # S_{ij} = R_{ij} * sqrt(pi_i/pi_j)
    # For reversible model: S is symmetric, S_{ij} = mu_ij * sqrt(pi_i * pi_j) / normalization
    # Actually S_{ij} = R_{ij} * sqrt(pi_i/pi_j), and R is the rate matrix

    # Reconstruct S from eigendecomposition: S = V diag(mu) V^T
    mu = model.eigenvalues
    S = jnp.einsum('...ak,...k,...bk->...ab', V, mu, V)  # (*H, A, A)

    # Off-diagonal: u_{ij} = S_{ij} * VCV_{ij}
    # Diagonal: w_i = VCV_{ii}
    # Build result
    diag_mask = jnp.eye(V.shape[-1], dtype=bool)
    counts = jnp.where(
        diag_mask[..., None],
        VCV,           # diagonal = dwell times
        S[..., None] * VCV  # off-diagonal = S_ij * VCV_ij
    )

    return counts


def compute_J_complex(eigenvalues, distances):
    """Compute J^{kl}(T) interaction matrix for complex eigenvalues.

    Same formula as compute_J but operates on complex eigenvalues.

    Args:
        eigenvalues: (*H, A) complex eigenvalues
        distances: (R,) branch lengths (real)

    Returns:
        (*H, R, A, A) complex J matrices per branch
    """
    mu = eigenvalues  # (*H, A) complex
    t = distances     # (R,) real

    mu_t = mu[..., None, :] * t[..., None]
    exp_mu_t = jnp.exp(mu_t)

    mu_diff = mu[..., :, None] - mu[..., None, :]

    exp_k = exp_mu_t[..., :, None]
    exp_l = exp_mu_t[..., None, :]

    mu_diff_expanded = mu_diff[..., None, :, :]

    J_nondeg = (exp_k - exp_l) / (mu_diff_expanded + 1e-30)

    t_expanded = t[..., None, None]
    J_deg = t_expanded * exp_k

    is_degenerate = jnp.abs(mu_diff_expanded) < 1e-8
    J = jnp.where(is_degenerate, J_deg, J_nondeg)

    return J


def eigenbasis_project_irrev(U, D, model):
    """Project inside/outside vectors into eigenbasis for irreversible model.

    U_tilde_k = sum_b U_b * V_bk         (no sqrt(pi) weighting)
    D_tilde_k = sum_a D_a * V_inv_ka      (using V^{-1}, transposed)

    Args:
        U: (*H, R, C, A) inside vectors (rescaled, real)
        D: (*H, R, C, A) outside vectors (rescaled, real)
        model: IrrevDiagModel

    Returns:
        U_tilde: (*H, R, C, A) complex projected inside
        D_tilde: (*H, R, C, A) complex projected outside
    """
    V = model.eigenvectors       # (*H, A, A) complex
    V_inv = model.eigenvectors_inv  # (*H, A, A) complex

    # U_tilde_k = sum_b U_b * V_bk
    U_tilde = jnp.einsum('...rcb,...bk->...rck', U, V)

    # D_tilde_k = sum_a D_a * V_inv_ka
    # V_inv is (A,A) where V_inv[k,a], so V_inv transposed gives V_inv[a,k]
    # We want sum_a D_a * V_inv[k,a] = sum_a D_a * V_inv^T[a,k]
    V_inv_T = jnp.swapaxes(V_inv, -2, -1)  # (*H, A, A)
    D_tilde = jnp.einsum('...rca,...ak->...rck', D, V_inv_T)

    return U_tilde, D_tilde


def accumulate_C_complex(D_tilde, U_tilde, J, logNormU, logNormD, logLike,
                         parentIndex, branch_mask=None):
    """Accumulate eigenbasis counts C_{kl} over branches (complex variant).

    Same structure as accumulate_C but D_tilde, U_tilde, J are complex.
    Scale remains real.

    Args:
        D_tilde: (*H, R, C, A) complex projected outside vectors
        U_tilde: (*H, R, C, A) complex projected inside vectors
        J: (*H, R, A, A) complex J matrices per branch
        logNormU: (*H, R, C) real per-node inside log-normalizers
        logNormD: (*H, R, C) real per-node outside log-normalizers
        logLike: (*H, C) real log-likelihoods
        parentIndex: (R,) parent indices
        branch_mask: (*H, R, C) bool or None

    Returns:
        C: (*H, A, A, C) complex eigenbasis counts
    """
    log_scale = (
        logNormD[..., 1:, :] + logNormU[..., 1:, :] - logLike[..., None, :]
    )
    scale = jnp.exp(log_scale)
    if branch_mask is not None:
        scale = scale * branch_mask[..., 1:, :]

    Dt = D_tilde[..., 1:, :, :]
    Ut = U_tilde[..., 1:, :, :]
    Jb = J[..., 1:, :, :]

    Dt_scaled = Dt * scale[..., None]

    C = jnp.einsum('...nck,...nkl,...ncl->...klc', Dt_scaled, Jb, Ut)

    return C


def back_transform_irrev(C, model):
    """Transform eigenbasis counts to natural basis for irreversible model.

    VCV = V C V^{-1} per column
    R = V diag(mu) V^{-1} (the rate matrix)
    Off-diagonal: R_ij * VCV_ij, diagonal: VCV_ii
    Takes .real of final result.

    Args:
        C: (*H, A, A, C) complex eigenbasis counts
        model: IrrevDiagModel

    Returns:
        (*H, A, A, C) real counts tensor
    """
    V = model.eigenvectors       # (*H, A, A) complex
    V_inv = model.eigenvectors_inv  # (*H, A, A) complex

    # VCV_{ij,c} = sum_{kl} V_{ik} C_{kl,c} V_inv_{lj}
    VCV = jnp.einsum('...ik,...klc,...lj->...ijc', V, C, V_inv)

    # R = V diag(mu) V^{-1}
    mu = model.eigenvalues  # (*H, A) complex
    R = jnp.einsum('...ak,...k,...kj->...aj', V, mu, V_inv)

    diag_mask = jnp.eye(V.shape[-1], dtype=bool)
    counts = jnp.where(
        diag_mask[..., None],
        VCV,
        R[..., None] * VCV
    )

    return counts.real
