"""Imperative Python oracle for phylogenetic sufficient statistics.

Every algorithm is implemented as explicit nested for-loops over indices.
No vectorization, no einsum, no JAX. Uses only numpy (and scipy.linalg.qr
for eigenvector construction). Purpose: be obviously correct at the expense
of speed, serving as the cross-language test oracle for WebGPU and WASM.

All arrays are plain numpy ndarrays with float64 precision.
"""

import numpy as np
from scipy.special import gammainc as _scipy_gammainc


# ---------------------------------------------------------------------------
# 1. Tree utilities
# ---------------------------------------------------------------------------

def children_of(parentIndex):
    """Compute left_child, right_child, sibling arrays for a binary tree.

    Args:
        parentIndex: (R,) int array, preorder (parentIndex[0] == -1).

    Returns:
        left_child:  (R,) int, -1 for leaves
        right_child: (R,) int, -1 for leaves
        sibling:     (R,) int, -1 for root
    """
    R = len(parentIndex)
    left_child = np.full(R, -1, dtype=np.intp)
    right_child = np.full(R, -1, dtype=np.intp)

    for n in range(1, R):
        p = parentIndex[n]
        if left_child[p] == -1:
            left_child[p] = n
        else:
            right_child[p] = n

    sibling = np.full(R, -1, dtype=np.intp)
    for n in range(1, R):
        p = parentIndex[n]
        if left_child[p] == n:
            sibling[n] = right_child[p]
        else:
            sibling[n] = left_child[p]

    return left_child, right_child, sibling


def validate_binary_tree(parentIndex):
    """Assert every non-leaf node has exactly 2 children."""
    R = len(parentIndex)
    child_count = np.zeros(R, dtype=int)
    for n in range(1, R):
        child_count[parentIndex[n]] += 1
    for n in range(R):
        if child_count[n] != 0 and child_count[n] != 2:
            raise ValueError(
                f"Node {n} has {child_count[n]} children; expected 0 or 2."
            )


# ---------------------------------------------------------------------------
# 2. Token to likelihood
# ---------------------------------------------------------------------------

def token_to_likelihood(alignment, A):
    """Convert integer token alignment to likelihood vectors.

    Token encoding:
        0..A-1 : observed (one-hot)
        A      : ungapped-unobserved (all ones)
        A+1    : gapped (all ones)
        -1     : gap (legacy, all ones)

    Args:
        alignment: (R, C) int array
        A: alphabet size

    Returns:
        (R, C, A) float64 likelihood vectors
    """
    R, C = alignment.shape
    L = np.zeros((R, C, A), dtype=np.float64)
    for r in range(R):
        for c in range(C):
            tok = alignment[r, c]
            if 0 <= tok < A:
                L[r, c, tok] = 1.0
            else:
                # gap (-1), ungapped-unobserved (A), gapped (A+1)
                for a in range(A):
                    L[r, c, a] = 1.0
    return L


# ---------------------------------------------------------------------------
# 3-6. Model construction
# ---------------------------------------------------------------------------

def hky85_diag(kappa, pi):
    """HKY85 model with closed-form eigendecomposition.

    Args:
        kappa: transition/transversion ratio (scalar)
        pi: (4,) equilibrium frequencies [A, C, G, T]

    Returns:
        dict with 'eigenvalues' (4,), 'eigenvectors' (4,4), 'pi' (4,)
    """
    pi = np.asarray(pi, dtype=np.float64)
    pi_A, pi_C, pi_G, pi_T = pi[0], pi[1], pi[2], pi[3]
    pi_R = pi_A + pi_G
    pi_Y = pi_C + pi_T

    beta = 1.0 / (2.0 * pi_R * pi_Y + 2.0 * kappa * (pi_A * pi_G + pi_C * pi_T))

    eigenvalues = np.array([
        0.0,
        -beta,
        -beta * (pi_R + kappa * pi_Y),
        -beta * (pi_Y + kappa * pi_R),
    ])

    sqrt_pi = np.sqrt(pi)

    # w0 = sqrt(pi) (stationary mode)
    w0 = sqrt_pi.copy()

    # w1: purine-pyrimidine split
    w1 = np.array([
        sqrt_pi[0] * pi_Y,
        -sqrt_pi[1] * pi_R,
        sqrt_pi[2] * pi_Y,
        -sqrt_pi[3] * pi_R,
    ])
    w1 /= np.sqrt(pi_R * pi_Y)

    # w2: within-pyrimidine
    w2 = np.array([
        0.0,
        sqrt_pi[1] * pi_T,
        0.0,
        -sqrt_pi[3] * pi_C,
    ])
    w2 /= np.sqrt(pi_C * pi_T * pi_Y)

    # w3: within-purine
    w3 = np.array([
        sqrt_pi[0] * pi_G,
        0.0,
        -sqrt_pi[2] * pi_A,
        0.0,
    ])
    w3 /= np.sqrt(pi_A * pi_G * pi_R)

    eigenvectors = np.column_stack([w0, w1, w2, w3])  # (4, 4)

    return {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors, 'pi': pi}


def jukes_cantor_model(A):
    """Jukes-Cantor model for an A-state alphabet.

    Returns:
        dict with 'eigenvalues' (A,), 'eigenvectors' (A,A), 'pi' (A,)
    """
    pi = np.ones(A, dtype=np.float64) / A
    mu = float(A) / (A - 1.0)
    eigenvalues = np.zeros(A, dtype=np.float64)
    eigenvalues[1:] = -mu

    v0 = np.ones(A, dtype=np.float64) / np.sqrt(A)
    basis = np.eye(A, dtype=np.float64)
    augmented = np.column_stack([v0, basis])
    Q, _ = np.linalg.qr(augmented)
    eigenvectors = Q[:, :A]

    return {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors, 'pi': pi}


def f81_model(pi):
    """F81 model: R_ij = mu * pi_j for i!=j, normalized to expected rate = 1.

    Args:
        pi: (A,) equilibrium frequencies

    Returns:
        dict with 'eigenvalues' (A,), 'eigenvectors' (A,A), 'pi' (A,)
    """
    pi = np.asarray(pi, dtype=np.float64)
    A = len(pi)
    mu = 1.0 / (1.0 - np.sum(pi ** 2))
    eigenvalues = np.zeros(A, dtype=np.float64)
    eigenvalues[1:] = -mu

    sqrt_pi = np.sqrt(pi)
    basis = np.eye(A, dtype=np.float64)
    augmented = np.column_stack([sqrt_pi, basis])
    Q, _ = np.linalg.qr(augmented)
    eigenvectors = Q[:, :A]

    return {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors, 'pi': pi}


def gamma_rate_categories(alpha, K):
    """Yang (1994) discretized gamma rate categories using quantile medians.

    Args:
        alpha: shape parameter (scalar)
        K: number of categories

    Returns:
        (rates, weights): each (K,) — rates are mean-normalized, weights are uniform
    """
    boundaries = np.linspace(0, 1, K + 1)
    midpoints = 0.5 * (boundaries[:-1] + boundaries[1:])

    rates = np.zeros(K, dtype=np.float64)
    for i in range(K):
        rates[i] = _gamma_quantile(alpha, midpoints[i])

    rates = rates * K / np.sum(rates)
    weights = np.ones(K, dtype=np.float64) / K
    return rates, weights


def _gamma_quantile(alpha, p):
    """Compute quantile of Gamma(alpha, 1/alpha) distribution via bisection."""
    lo = 0.0
    hi = max(50.0, 10.0 / alpha)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        cdf = float(_scipy_gammainc(alpha, mid * alpha))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def scale_model(model, rate_multiplier):
    """Scale eigenvalues by a rate multiplier.

    Args:
        model: dict with 'eigenvalues', 'eigenvectors', 'pi'
        rate_multiplier: scalar

    Returns:
        New model dict with scaled eigenvalues.
    """
    return {
        'eigenvalues': model['eigenvalues'] * rate_multiplier,
        'eigenvectors': model['eigenvectors'].copy(),
        'pi': model['pi'].copy(),
    }


# ---------------------------------------------------------------------------
# 6b. Irreversible model construction
# ---------------------------------------------------------------------------

def diagonalize_irreversible(subRate, rootProb):
    """Eigendecompose an irreversible rate matrix.

    R = V diag(mu) V^{-1} via numpy eig.

    Args:
        subRate: (A, A) rate matrix
        rootProb: (A,) stationary distribution

    Returns:
        dict with complex eigenvalues, eigenvectors, eigenvectors_inv, pi,
        and 'reversible': False
    """
    subRate = np.asarray(subRate, dtype=np.float64)
    rootProb = np.asarray(rootProb, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(subRate)
    eigenvectors_inv = np.linalg.inv(eigenvectors)
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'eigenvectors_inv': eigenvectors_inv,
        'pi': rootProb,
        'reversible': False,
    }


def irrev_model_from_rate_matrix(subRate, pi):
    """Construct an irreversible model from a rate matrix.

    Args:
        subRate: (A, A) rate matrix
        pi: (A,) stationary distribution

    Returns:
        dict with complex arrays and 'reversible': False
    """
    return diagonalize_irreversible(subRate, pi)


def diagonalize_rate_matrix(subRate, rootProb):
    """Eigendecompose a reversible rate matrix.

    S_ij = R_ij * sqrt(pi_i / pi_j) is symmetric.
    S = V diag(mu) V^T.

    Args:
        subRate: (A, A) rate matrix
        rootProb: (A,) equilibrium distribution

    Returns:
        dict with 'eigenvalues', 'eigenvectors', 'pi', 'reversible': True
    """
    subRate = np.asarray(subRate, dtype=np.float64)
    rootProb = np.asarray(rootProb, dtype=np.float64)
    A = len(rootProb)
    sqrt_pi = np.sqrt(rootProb)
    inv_sqrt_pi = 1.0 / sqrt_pi
    # S_ij = R_ij * sqrt(pi_i / pi_j)
    S = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            S[i, j] = subRate[i, j] * sqrt_pi[i] * inv_sqrt_pi[j]
    # Symmetrize
    S = 0.5 * (S + S.T)
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'pi': rootProb,
        'reversible': True,
    }


def check_detailed_balance(subRate, pi, tol=1e-10):
    """Check whether a rate matrix satisfies detailed balance: pi_i R_ij = pi_j R_ji.

    Args:
        subRate: (A, A) rate matrix
        pi: (A,) equilibrium distribution
        tol: absolute tolerance

    Returns:
        True if detailed balance holds.
    """
    subRate = np.asarray(subRate, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    A = len(pi)
    for i in range(A):
        for j in range(A):
            if abs(pi[i] * subRate[i, j] - pi[j] * subRate[j, i]) > tol:
                return False
    return True


def model_from_rate_matrix(subRate, pi, reversible=None, tol=1e-10):
    """Construct a diagonalized model from a rate matrix.

    Auto-detects reversibility via detailed balance when reversible=None.

    Args:
        subRate: (A, A) rate matrix
        pi: (A,) equilibrium distribution
        reversible: True → reversible (eigh), False → irreversible (eig),
                    None → auto-detect via detailed balance check
        tol: tolerance for detailed balance check when reversible=None

    Returns:
        dict — reversible model if reversible, irreversible otherwise.
    """
    subRate = np.asarray(subRate, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    if reversible is None:
        reversible = check_detailed_balance(subRate, pi, tol=tol)
    if reversible:
        return diagonalize_rate_matrix(subRate, pi)
    else:
        return diagonalize_irreversible(subRate, pi)


# ---------------------------------------------------------------------------
# 6b2. Goldman-Yang (GY94) codon substitution model
# ---------------------------------------------------------------------------

def _gy94_codon_neighbors():
    """Precompute single-nucleotide codon neighbor relationships.

    Returns:
        list of (i, j, is_transition, is_nonsynonymous) for all pairs of
        sense codons differing at exactly one nucleotide position.
    """
    from subby.formats import genetic_code
    gc = genetic_code()
    sense_indices = gc['sense_indices']
    codons = gc['codons']
    amino_acids = gc['amino_acids']

    transitions = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

    neighbors = []
    for si, idx_i in enumerate(sense_indices):
        codon_i = codons[idx_i]
        aa_i = amino_acids[idx_i]
        for sj, idx_j in enumerate(sense_indices):
            if si == sj:
                continue
            codon_j = codons[idx_j]
            # Count nucleotide differences
            diffs = [(p, codon_i[p], codon_j[p]) for p in range(3) if codon_i[p] != codon_j[p]]
            if len(diffs) != 1:
                continue
            _, nuc_i, nuc_j = diffs[0]
            is_ts = (nuc_i, nuc_j) in transitions
            aa_j = amino_acids[idx_j]
            is_nonsyn = aa_i != aa_j
            neighbors.append((si, sj, is_ts, is_nonsyn))
    return neighbors


def gy94_model(omega, kappa, pi=None):
    """Goldman-Yang (1994) codon substitution model.

    Operates on 61 sense codons. Rate matrix:
      Q_ij = 0 if codons differ at more than 1 nucleotide position
      Q_ij = pi_j * kappa^(is_transition) * omega^(is_nonsynonymous)
      Q_ii = -sum_{j != i} Q_ij
      Normalized so -sum_i pi_i Q_ii = 1.

    This is reversible (pi_i Q_ij = pi_j Q_ji), so uses symmetric
    eigendecomposition.

    Args:
        omega: dN/dS ratio (Ka/Ks)
        kappa: transition/transversion ratio
        pi: (61,) codon equilibrium frequencies (default: uniform 1/61)

    Returns:
        dict with 'eigenvalues', 'eigenvectors', 'pi', 'reversible': True
    """
    A = 61
    if pi is None:
        pi = np.ones(A, dtype=np.float64) / A
    pi = np.asarray(pi, dtype=np.float64)

    neighbors = _gy94_codon_neighbors()

    Q = np.zeros((A, A), dtype=np.float64)
    for si, sj, is_ts, is_nonsyn in neighbors:
        rate = pi[sj]
        if is_ts:
            rate *= kappa
        if is_nonsyn:
            rate *= omega
        Q[si, sj] = rate

    # Set diagonal
    for i in range(A):
        Q[i, i] = -np.sum(Q[i, :])

    # Normalize: -sum_i pi_i Q_ii = 1
    expected_rate = -np.sum(pi * np.diag(Q))
    Q /= expected_rate

    return diagonalize_rate_matrix(Q, pi)


# ---------------------------------------------------------------------------
# 6c. Irreversible substitution matrices
# ---------------------------------------------------------------------------

def compute_sub_matrices_irrev(model, distances):
    """Compute substitution probability matrices M(t) for irreversible model.

    M_ij(t) = Re(sum_k V_ik * exp(mu_k * t) * V_inv_kj)

    Args:
        model: dict with complex 'eigenvalues', 'eigenvectors', 'eigenvectors_inv', 'pi'
        distances: (R,) branch lengths

    Returns:
        (R, A, A) substitution matrices (real)
    """
    V = model['eigenvectors']         # (A, A) complex
    V_inv = model['eigenvectors_inv']  # (A, A) complex
    mu = model['eigenvalues']         # (A,) complex
    R = len(distances)
    A = len(mu)

    M = np.zeros((R, A, A), dtype=np.float64)
    for r in range(R):
        t = distances[r]
        for i in range(A):
            for j in range(A):
                s = 0.0 + 0.0j
                for k in range(A):
                    s += V[i, k] * np.exp(mu[k] * t) * V_inv[k, j]
                M[r, i, j] = s.real
    return M


# ---------------------------------------------------------------------------
# 6d. Irreversible eigensub functions
# ---------------------------------------------------------------------------

def compute_J_complex(eigenvalues, distances):
    """Compute J^{kl}(T) interaction matrix for complex eigenvalues.

    Args:
        eigenvalues: (A,) complex eigenvalues
        distances: (R,) branch lengths

    Returns:
        (R, A, A) complex J matrices per branch
    """
    A = len(eigenvalues)
    R = len(distances)
    J = np.zeros((R, A, A), dtype=np.complex128)

    for r in range(R):
        t = distances[r]
        for k in range(A):
            mu_k = eigenvalues[k]
            exp_k = np.exp(mu_k * t)
            for l in range(A):
                mu_l = eigenvalues[l]
                diff = mu_k - mu_l
                if abs(diff) < 1e-8:
                    J[r, k, l] = t * exp_k
                else:
                    J[r, k, l] = (exp_k - np.exp(mu_l * t)) / diff
    return J


def eigenbasis_project_irrev(U, D, model):
    """Project inside/outside vectors into eigenbasis for irreversible model.

    U_tilde_k = sum_b U_b * V_bk         (no pi weighting)
    D_tilde_k = sum_a D_a * V_inv_ka     (using V^{-1})

    Args:
        U: (R, C, A) inside vectors (real)
        D: (R, C, A) outside vectors (real)
        model: dict with complex 'eigenvectors', 'eigenvectors_inv'

    Returns:
        U_tilde: (R, C, A) complex projected inside
        D_tilde: (R, C, A) complex projected outside
    """
    V = model['eigenvectors']         # (A, A) complex
    V_inv = model['eigenvectors_inv']  # (A, A) complex
    R, C, A = U.shape

    U_tilde = np.zeros((R, C, A), dtype=np.complex128)
    D_tilde = np.zeros((R, C, A), dtype=np.complex128)

    for r in range(R):
        for c in range(C):
            for k in range(A):
                su = 0.0 + 0.0j
                sd = 0.0 + 0.0j
                for b in range(A):
                    su += U[r, c, b] * V[b, k]
                    sd += D[r, c, b] * V_inv[k, b]
                U_tilde[r, c, k] = su
                D_tilde[r, c, k] = sd

    return U_tilde, D_tilde


def accumulate_C_complex(D_tilde, U_tilde, J, logNormU, logNormD, logLike,
                         parentIndex, branch_mask=None):
    """Accumulate eigenbasis counts C_{kl} over branches (complex).

    Args:
        D_tilde: (R, C, A) complex projected outside vectors
        U_tilde: (R, C, A) complex projected inside vectors
        J: (R, A, A) complex J matrices per branch
        logNormU: (R, C) real
        logNormD: (R, C) real
        logLike: (C,) real
        parentIndex: (R,) parent indices
        branch_mask: (R, C) bool or None

    Returns:
        C: (A, A, C) complex eigenbasis counts
    """
    R, C, A = U_tilde.shape
    Cout = np.zeros((A, A, C), dtype=np.complex128)

    for n in range(1, R):
        for c in range(C):
            if branch_mask is not None and not branch_mask[n, c]:
                continue
            log_s = logNormD[n, c] + logNormU[n, c] - logLike[c]
            scale = np.exp(log_s)
            for k in range(A):
                for l in range(A):
                    Cout[k, l, c] += D_tilde[n, c, k] * J[n, k, l] * U_tilde[n, c, l] * scale

    return Cout


def accumulate_C_complex_per_branch(D_tilde, U_tilde, J, logNormU, logNormD,
                                    logLike, parentIndex, branch_mask=None):
    """Accumulate eigenbasis counts C_{kl} per branch (complex, not summed).

    Same as accumulate_C_complex but returns (R, A, A, C) instead of (A, A, C).
    Branch 0 (root) is all zeros.

    Args:
        D_tilde: (R, C, A) complex projected outside vectors
        U_tilde: (R, C, A) complex projected inside vectors
        J: (R, A, A) complex J matrices per branch
        logNormU: (R, C) real
        logNormD: (R, C) real
        logLike: (C,) real
        parentIndex: (R,) parent indices
        branch_mask: (R, C) bool or None

    Returns:
        Cout: (R, A, A, C) complex eigenbasis counts per branch
    """
    R, C, A = U_tilde.shape
    Cout = np.zeros((R, A, A, C), dtype=np.complex128)

    for n in range(1, R):
        for c in range(C):
            if branch_mask is not None and not branch_mask[n, c]:
                continue
            log_s = logNormD[n, c] + logNormU[n, c] - logLike[c]
            scale = np.exp(log_s)
            for k in range(A):
                for l in range(A):
                    Cout[n, k, l, c] = D_tilde[n, c, k] * J[n, k, l] * U_tilde[n, c, l] * scale

    return Cout


def back_transform_irrev(C, model):
    """Transform eigenbasis counts to natural basis for irreversible model.

    VCV_{ij,c} = sum_{kl} V_{ik} C_{kl,c} V_inv_{lj}
    R = V diag(mu) V^{-1}
    Off-diagonal: R_ij * VCV_ij, diagonal: VCV_ii
    Takes .real of final result.

    Args:
        C: (A, A, C) complex eigenbasis counts
        model: dict with complex 'eigenvalues', 'eigenvectors', 'eigenvectors_inv'

    Returns:
        (A, A, C) real counts tensor
    """
    V = model['eigenvectors']         # (A, A) complex
    V_inv = model['eigenvectors_inv']  # (A, A) complex
    mu = model['eigenvalues']         # (A,) complex
    A = len(mu)
    Ccols = C.shape[2]

    # Reconstruct R = V diag(mu) V^{-1}
    R_mat = np.zeros((A, A), dtype=np.complex128)
    for i in range(A):
        for j in range(A):
            s = 0.0 + 0.0j
            for k in range(A):
                s += V[i, k] * mu[k] * V_inv[k, j]
            R_mat[i, j] = s

    # VCV = V C V^{-1} per column
    VCV = np.zeros((A, A, Ccols), dtype=np.complex128)
    for col in range(Ccols):
        for i in range(A):
            for j in range(A):
                s = 0.0 + 0.0j
                for k in range(A):
                    for l in range(A):
                        s += V[i, k] * C[k, l, col] * V_inv[l, j]
                VCV[i, j, col] = s

    # Build result
    counts = np.zeros((A, A, Ccols), dtype=np.float64)
    for col in range(Ccols):
        for i in range(A):
            for j in range(A):
                if i == j:
                    counts[i, j, col] = VCV[i, j, col].real
                else:
                    counts[i, j, col] = (R_mat[i, j] * VCV[i, j, col]).real

    return counts


def back_transform_irrev_per_branch(C, model):
    """Transform eigenbasis counts to natural basis for irreversible model, per branch.

    Same as back_transform_irrev but works on (R, A, A, C) input.

    Args:
        C: (R, A, A, C) complex eigenbasis counts per branch
        model: dict with complex 'eigenvalues', 'eigenvectors', 'eigenvectors_inv'

    Returns:
        (R, A, A, C) real counts tensor per branch
    """
    V = model['eigenvectors']         # (A, A) complex
    V_inv = model['eigenvectors_inv']  # (A, A) complex
    mu = model['eigenvalues']         # (A,) complex
    A = len(mu)
    R = C.shape[0]
    Ccols = C.shape[3]

    # Reconstruct R = V diag(mu) V^{-1}
    R_mat = np.zeros((A, A), dtype=np.complex128)
    for i in range(A):
        for j in range(A):
            s = 0.0 + 0.0j
            for k in range(A):
                s += V[i, k] * mu[k] * V_inv[k, j]
            R_mat[i, j] = s

    counts = np.zeros((R, A, A, Ccols), dtype=np.float64)
    for n in range(R):
        # VCV = V C[n] V^{-1} per column
        for col in range(Ccols):
            for i in range(A):
                for j in range(A):
                    s = 0.0 + 0.0j
                    for k in range(A):
                        for l in range(A):
                            s += V[i, k] * C[n, k, l, col] * V_inv[l, j]
                    vcv = s
                    if i == j:
                        counts[n, i, j, col] = vcv.real
                    else:
                        counts[n, i, j, col] = (R_mat[i, j] * vcv).real

    return counts


# ---------------------------------------------------------------------------
# 7. Substitution matrices
# ---------------------------------------------------------------------------

def compute_sub_matrices(model, distances):
    """Compute substitution probability matrices M(t) from eigendecomposition.

    M_ij(t) = sqrt(pi_j/pi_i) * sum_k v_ik * exp(mu_k * t) * v_jk

    Args:
        model: dict with 'eigenvalues' (A,), 'eigenvectors' (A,A), 'pi' (A,)
        distances: (R,) branch lengths

    Returns:
        (R, A, A) substitution matrices
    """
    V = model['eigenvectors']   # (A, A)
    mu = model['eigenvalues']   # (A,)
    pi = model['pi']            # (A,)
    R = len(distances)
    A = len(mu)

    sqrt_pi = np.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi

    M = np.zeros((R, A, A), dtype=np.float64)
    for r in range(R):
        t = distances[r]
        for i in range(A):
            for j in range(A):
                s = 0.0
                for k in range(A):
                    s += V[i, k] * np.exp(mu[k] * t) * V[j, k]
                M[r, i, j] = inv_sqrt_pi[i] * sqrt_pi[j] * s
    return M


# ---------------------------------------------------------------------------
# 8. Upward pass (Felsenstein pruning)
# ---------------------------------------------------------------------------

def upward_pass(alignment, tree, subMatrices, rootProb):
    """Compute inside (U) vectors for all nodes.

    Args:
        alignment: (R, C) int32 tokens
        tree: dict with 'parentIndex' (R,), 'distanceToParent' (R,)
        subMatrices: (R, A, A) substitution probability matrices
        rootProb: (A,) root/equilibrium frequencies

    Returns:
        U: (R, C, A) inside vectors per node (rescaled)
        logNormU: (R, C) per-node subtree log-normalizers
        logLike: (C,) log-likelihoods
    """
    parentIndex = tree['parentIndex']
    R, C = alignment.shape
    A = subMatrices.shape[-1]

    # Initialize likelihood from tokens
    U = token_to_likelihood(alignment, A)  # (R, C, A)
    logNormU = np.zeros((R, C), dtype=np.float64)

    # Postorder: children R-1 down to 1
    for n in range(R - 1, 0, -1):
        p = parentIndex[n]
        M = subMatrices[n]  # (A, A)
        # parent_b *= sum_j M_bj * child_j
        for c in range(C):
            child_contrib = np.zeros(A, dtype=np.float64)
            for b in range(A):
                s = 0.0
                for j in range(A):
                    s += M[b, j] * U[n, c, j]
                child_contrib[b] = s
            for b in range(A):
                U[p, c, b] *= child_contrib[b]

            # Rescale parent
            max_val = max(U[p, c, :])
            if max_val < 1e-300:
                max_val = 1e-300
            for b in range(A):
                U[p, c, b] /= max_val
            log_rescale = np.log(max_val)
            logNormU[p, c] += logNormU[n, c] + log_rescale

    # Log-likelihood
    logLike = np.zeros(C, dtype=np.float64)
    for c in range(C):
        s = 0.0
        for a in range(A):
            s += rootProb[a] * U[0, c, a]
        logLike[c] = logNormU[0, c] + np.log(s)

    return U, logNormU, logLike


# ---------------------------------------------------------------------------
# 9. Downward pass (outside algorithm)
# ---------------------------------------------------------------------------

def downward_pass(U, logNormU, tree, subMatrices, rootProb, alignment):
    """Compute outside (D) vectors for all nodes.

    Args:
        U: (R, C, A) inside vectors (rescaled)
        logNormU: (R, C) per-node log-normalizers
        tree: dict with 'parentIndex' (R,), 'distanceToParent' (R,)
        subMatrices: (R, A, A) substitution probability matrices
        rootProb: (A,) root/equilibrium frequencies
        alignment: (R, C) int32 tokens

    Returns:
        D: (R, C, A) outside vectors (rescaled)
        logNormD: (R, C) per-node outside log-normalizers
    """
    parentIndex = tree['parentIndex']
    R, C, A = U.shape

    _, _, sibling = children_of(parentIndex)
    obs_like = token_to_likelihood(alignment, A)  # (R, C, A)

    D = np.zeros((R, C, A), dtype=np.float64)
    logNormD = np.zeros((R, C), dtype=np.float64)

    # Preorder: nodes 1, 2, ..., R-1
    for n in range(1, R):
        p = parentIndex[n]
        sib = sibling[n]
        M_sib = subMatrices[sib]  # (A, A) — sibling's branch matrix
        M_p = subMatrices[p]       # (A, A) — parent's branch matrix

        for c in range(C):
            # Sibling contribution: sum_j M_{sib}[a,j] * U[sib,c,j]
            sib_contrib = np.zeros(A, dtype=np.float64)
            for a in range(A):
                s = 0.0
                for j in range(A):
                    s += M_sib[a, j] * U[sib, c, j]
                sib_contrib[a] = s

            # Parent contribution
            parent_contrib = np.zeros(A, dtype=np.float64)
            if p == 0:
                # Parent is root: use pi
                for a in range(A):
                    parent_contrib[a] = rootProb[a]
            else:
                # sum_i D[p,c,i] * M_p[i,a]
                for a in range(A):
                    s = 0.0
                    for i in range(A):
                        s += D[p, c, i] * M_p[i, a]
                    parent_contrib[a] = s

            # Include parent's observation likelihood
            for a in range(A):
                parent_contrib[a] *= obs_like[p, c, a]

            # D[n] = sib_contrib * parent_contrib
            D_raw = np.zeros(A, dtype=np.float64)
            for a in range(A):
                D_raw[a] = sib_contrib[a] * parent_contrib[a]

            # logNormD computation
            log_norm_from_sib = logNormU[sib, c]
            if p == 0:
                log_norm_prior = 0.0
            else:
                log_norm_prior = logNormD[p, c]
            accumulated = log_norm_from_sib + log_norm_prior

            # Rescale
            max_val = max(D_raw)
            if max_val < 1e-300:
                max_val = 1e-300
            for a in range(A):
                D[n, c, a] = D_raw[a] / max_val
            logNormD[n, c] = accumulated + np.log(max_val)

    return D, logNormD


# ---------------------------------------------------------------------------
# 10. Compute J (eigenvalue interaction matrix)
# ---------------------------------------------------------------------------

def compute_J(eigenvalues, distances):
    """Compute J^{kl}(T) interaction matrix between decay modes.

    J^{kl}(T) = T * exp(mu_k * T)                      if |mu_k - mu_l| < 1e-8
                (exp(mu_k*T) - exp(mu_l*T)) / (mu_k - mu_l)  otherwise

    Args:
        eigenvalues: (A,) eigenvalues
        distances: (R,) branch lengths

    Returns:
        (R, A, A) J matrices per branch
    """
    A = len(eigenvalues)
    R = len(distances)
    J = np.zeros((R, A, A), dtype=np.float64)

    for r in range(R):
        t = distances[r]
        for k in range(A):
            for l in range(A):
                mu_k = eigenvalues[k]
                mu_l = eigenvalues[l]
                diff = mu_k - mu_l
                if abs(diff) < 1e-8:
                    J[r, k, l] = t * np.exp(mu_k * t)
                else:
                    J[r, k, l] = (np.exp(mu_k * t) - np.exp(mu_l * t)) / diff
    return J


# ---------------------------------------------------------------------------
# 11. Eigenbasis projection
# ---------------------------------------------------------------------------

def eigenbasis_project(U, D, model):
    """Project inside/outside vectors into eigenbasis.

    U_tilde_l = sum_b U_b * v_{bl} * sqrt(pi_b)
    D_tilde_k = sum_a D_a * v_{ak} / sqrt(pi_a)

    Args:
        U: (R, C, A) inside vectors (rescaled)
        D: (R, C, A) outside vectors (rescaled)
        model: dict

    Returns:
        U_tilde: (R, C, A) projected inside
        D_tilde: (R, C, A) projected outside
    """
    V = model['eigenvectors']  # (A, A)
    pi = model['pi']           # (A,)
    R, C, A = U.shape

    sqrt_pi = np.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi

    U_tilde = np.zeros((R, C, A), dtype=np.float64)
    D_tilde = np.zeros((R, C, A), dtype=np.float64)

    for r in range(R):
        for c in range(C):
            for k in range(A):
                # U_tilde_l = sum_b U_b * v_{bl} * sqrt(pi_b)
                su = 0.0
                sd = 0.0
                for b in range(A):
                    su += U[r, c, b] * V[b, k] * sqrt_pi[b]
                    sd += D[r, c, b] * V[b, k] * inv_sqrt_pi[b]
                U_tilde[r, c, k] = su
                D_tilde[r, c, k] = sd

    return U_tilde, D_tilde


# ---------------------------------------------------------------------------
# 12. Accumulate eigenbasis counts
# ---------------------------------------------------------------------------

def accumulate_C(D_tilde, U_tilde, J, logNormU, logNormD, logLike, parentIndex,
                 branch_mask=None):
    """Accumulate eigenbasis counts C_{kl} over branches.

    C_{kl,c} = sum_{n>0} D_tilde_k^(n) * J^{kl}(t_n) * U_tilde_l^(n) * scale[n,c]

    where scale[n,c] = exp(logNormD[n,c] + logNormU[n,c] - logLike[c])

    Args:
        D_tilde: (R, C, A) projected outside vectors
        U_tilde: (R, C, A) projected inside vectors
        J: (R, A, A) J matrices per branch
        logNormU: (R, C) per-node inside log-normalizers
        logNormD: (R, C) per-node outside log-normalizers
        logLike: (C,) log-likelihoods
        parentIndex: (R,) parent indices
        branch_mask: (R, C) bool or None — if provided, skip inactive branches

    Returns:
        C: (A, A, C) eigenbasis counts
    """
    R, C, A = U_tilde.shape
    Cout = np.zeros((A, A, C), dtype=np.float64)

    for n in range(1, R):
        for c in range(C):
            if branch_mask is not None and not branch_mask[n, c]:
                continue
            log_s = logNormD[n, c] + logNormU[n, c] - logLike[c]
            scale = np.exp(log_s)
            for k in range(A):
                for l in range(A):
                    Cout[k, l, c] += D_tilde[n, c, k] * J[n, k, l] * U_tilde[n, c, l] * scale

    return Cout


def accumulate_C_per_branch(D_tilde, U_tilde, J, logNormU, logNormD, logLike,
                            parentIndex, branch_mask=None):
    """Accumulate eigenbasis counts C_{kl} per branch (not summed).

    Same as accumulate_C but returns (R, A, A, C) instead of (A, A, C).
    Branch 0 (root) is all zeros.

    Args:
        D_tilde: (R, C, A) projected outside vectors
        U_tilde: (R, C, A) projected inside vectors
        J: (R, A, A) J matrices per branch
        logNormU: (R, C) per-node inside log-normalizers
        logNormD: (R, C) per-node outside log-normalizers
        logLike: (C,) log-likelihoods
        parentIndex: (R,) parent indices
        branch_mask: (R, C) bool or None — if provided, skip inactive branches

    Returns:
        Cout: (R, A, A, C) eigenbasis counts per branch
    """
    R, C, A = U_tilde.shape
    Cout = np.zeros((R, A, A, C), dtype=np.float64)

    for n in range(1, R):
        for c in range(C):
            if branch_mask is not None and not branch_mask[n, c]:
                continue
            log_s = logNormD[n, c] + logNormU[n, c] - logLike[c]
            scale = np.exp(log_s)
            for k in range(A):
                for l in range(A):
                    Cout[n, k, l, c] = D_tilde[n, c, k] * J[n, k, l] * U_tilde[n, c, l] * scale

    return Cout


# ---------------------------------------------------------------------------
# 13. Back-transform from eigenbasis to natural basis
# ---------------------------------------------------------------------------

def back_transform(C, model):
    """Transform eigenbasis counts to natural basis.

    VCV_{ij,c} = sum_{kl} V_{ik} C_{kl,c} V_{jl}
    Diagonal: dwell w_i = VCV_{ii,c}
    Off-diagonal: subs u_{ij} = S_{ij} * VCV_{ij,c}

    Args:
        C: (A, A, C) eigenbasis counts
        model: dict

    Returns:
        (A, A, C) counts tensor (diag=dwell, off-diag=substitution counts)
    """
    V = model['eigenvectors']  # (A, A)
    mu = model['eigenvalues']  # (A,)
    A = len(mu)
    Cshape = C.shape
    Ccols = Cshape[2]

    # Reconstruct S = V diag(mu) V^T
    S = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            s = 0.0
            for k in range(A):
                s += V[i, k] * mu[k] * V[j, k]
            S[i, j] = s

    # VCV = V C V^T per column
    VCV = np.zeros((A, A, Ccols), dtype=np.float64)
    for c in range(Ccols):
        for i in range(A):
            for j in range(A):
                s = 0.0
                for k in range(A):
                    for l in range(A):
                        s += V[i, k] * C[k, l, c] * V[j, l]
                VCV[i, j, c] = s

    # Build result
    counts = np.zeros((A, A, Ccols), dtype=np.float64)
    for c in range(Ccols):
        for i in range(A):
            for j in range(A):
                if i == j:
                    counts[i, j, c] = VCV[i, j, c]  # dwell
                else:
                    counts[i, j, c] = S[i, j] * VCV[i, j, c]  # subs

    return counts


def back_transform_per_branch(C, model):
    """Transform eigenbasis counts to natural basis, per branch.

    Same as back_transform but works on (R, A, A, C) input.

    Args:
        C: (R, A, A, C) eigenbasis counts per branch
        model: dict

    Returns:
        (R, A, A, C) counts tensor per branch (diag=dwell, off-diag=substitution counts)
    """
    V = model['eigenvectors']  # (A, A)
    mu = model['eigenvalues']  # (A,)
    A = len(mu)
    R = C.shape[0]
    Ccols = C.shape[3]

    # Reconstruct S = V diag(mu) V^T
    S = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            s = 0.0
            for k in range(A):
                s += V[i, k] * mu[k] * V[j, k]
            S[i, j] = s

    counts = np.zeros((R, A, A, Ccols), dtype=np.float64)
    for n in range(R):
        # VCV = V C[n] V^T per column
        for c in range(Ccols):
            for i in range(A):
                for j in range(A):
                    s = 0.0
                    for k in range(A):
                        for l in range(A):
                            s += V[i, k] * C[n, k, l, c] * V[j, l]
                    vcv = s
                    if i == j:
                        counts[n, i, j, c] = vcv  # dwell
                    else:
                        counts[n, i, j, c] = S[i, j] * vcv  # subs

    return counts


# ---------------------------------------------------------------------------
# 14. F81 fast path (direct O(CRA^2) computation)
# ---------------------------------------------------------------------------

def f81_counts(U, D, logNormU, logNormD, logLike, distances, pi, parentIndex,
               branch_mask=None):
    """O(CRA^2) direct computation of expected counts for F81/JC models.

    Args:
        U: (R, C, A) inside vectors (rescaled)
        D: (R, C, A) outside vectors (rescaled)
        logNormU: (R, C) per-node inside log-normalizers
        logNormD: (R, C) per-node outside log-normalizers
        logLike: (C,) log-likelihoods
        distances: (R,) branch lengths
        pi: (A,) equilibrium frequencies
        parentIndex: (R,) parent indices
        branch_mask: (R, C) bool or None — if provided, skip inactive branches

    Returns:
        (A, A, C) counts tensor (diag=dwell, off-diag=substitutions)
    """
    R, C, A = U.shape

    # mu = 1 / (1 - sum(pi^2))
    mu = 1.0 / (1.0 - np.sum(pi ** 2))

    result = np.zeros((A, A, C), dtype=np.float64)

    # Sum over branches n=1..R-1
    for n in range(1, R):
        t = distances[n]
        mu_t = mu * t
        e_t = np.exp(-mu_t)
        p = 1.0 - e_t

        alpha_n = t * e_t
        beta_n = p / mu - t * e_t
        gamma_n = t * (1.0 + e_t) - 2.0 * p / mu

        for c in range(C):
            if branch_mask is not None and not branch_mask[n, c]:
                continue
            log_s = logNormD[n, c] + logNormU[n, c] - logLike[c]
            scale = np.exp(log_s)

            # Compute piU = sum_b pi_b * U[n,c,b]
            piU = 0.0
            for b in range(A):
                piU += pi[b] * U[n, c, b]

            # Compute Dsum = sum_a D[n,c,a] * scale
            Dsum = 0.0
            for a in range(A):
                Dsum += D[n, c, a] * scale

            for i in range(A):
                D_i_scaled = D[n, c, i] * scale
                for j in range(A):
                    # I_sum = alpha*D_i*U_j + beta*(D_i*piU + pi_i*Dsum*U_j) + gamma*pi_i*Dsum*piU
                    I_sum = (
                        alpha_n * D_i_scaled * U[n, c, j]
                        + beta_n * (D_i_scaled * piU + pi[i] * Dsum * U[n, c, j])
                        + gamma_n * pi[i] * Dsum * piU
                    )
                    if i == j:
                        result[i, j, c] += I_sum  # dwell
                    else:
                        result[i, j, c] += mu * pi[j] * I_sum  # substitutions

    return result


def f81_counts_per_branch(U, D, logNormU, logNormD, logLike, distances, pi,
                          parentIndex, branch_mask=None):
    """O(CRA^2) direct computation of expected counts for F81/JC models, per branch.

    Same as f81_counts but returns (R, A, A, C) instead of (A, A, C).
    Branch 0 (root) is all zeros.

    Args:
        U: (R, C, A) inside vectors (rescaled)
        D: (R, C, A) outside vectors (rescaled)
        logNormU: (R, C) per-node inside log-normalizers
        logNormD: (R, C) per-node outside log-normalizers
        logLike: (C,) log-likelihoods
        distances: (R,) branch lengths
        pi: (A,) equilibrium frequencies
        parentIndex: (R,) parent indices
        branch_mask: (R, C) bool or None — if provided, skip inactive branches

    Returns:
        (R, A, A, C) counts tensor per branch (diag=dwell, off-diag=substitutions)
    """
    R, C, A = U.shape

    # mu = 1 / (1 - sum(pi^2))
    mu = 1.0 / (1.0 - np.sum(pi ** 2))

    result = np.zeros((R, A, A, C), dtype=np.float64)

    for n in range(1, R):
        t = distances[n]
        mu_t = mu * t
        e_t = np.exp(-mu_t)
        p = 1.0 - e_t

        alpha_n = t * e_t
        beta_n = p / mu - t * e_t
        gamma_n = t * (1.0 + e_t) - 2.0 * p / mu

        for c in range(C):
            if branch_mask is not None and not branch_mask[n, c]:
                continue
            log_s = logNormD[n, c] + logNormU[n, c] - logLike[c]
            scale = np.exp(log_s)

            # Compute piU = sum_b pi_b * U[n,c,b]
            piU = 0.0
            for b in range(A):
                piU += pi[b] * U[n, c, b]

            # Compute Dsum = sum_a D[n,c,a] * scale
            Dsum = 0.0
            for a in range(A):
                Dsum += D[n, c, a] * scale

            for i in range(A):
                D_i_scaled = D[n, c, i] * scale
                for j in range(A):
                    # I_sum = alpha*D_i*U_j + beta*(D_i*piU + pi_i*Dsum*U_j) + gamma*pi_i*Dsum*piU
                    I_sum = (
                        alpha_n * D_i_scaled * U[n, c, j]
                        + beta_n * (D_i_scaled * piU + pi[i] * Dsum * U[n, c, j])
                        + gamma_n * pi[i] * Dsum * piU
                    )
                    if i == j:
                        result[n, i, j, c] = I_sum  # dwell
                    else:
                        result[n, i, j, c] = mu * pi[j] * I_sum  # substitutions

    return result


# ---------------------------------------------------------------------------
# 15. Mixture posterior
# ---------------------------------------------------------------------------

def mixture_posterior(log_likes, log_weights):
    """Compute posterior probabilities over mixture components.

    P(k | c) = softmax_k(log_likes[k,c] + log_weights[k])

    Args:
        log_likes: (K, C) log-likelihoods per component
        log_weights: (K,) log prior weights

    Returns:
        (K, C) posterior probabilities (sum to 1 over K)
    """
    K, C = log_likes.shape
    posteriors = np.zeros((K, C), dtype=np.float64)

    for c in range(C):
        # Compute log_joint and find max for numerical stability
        max_val = -np.inf
        for k in range(K):
            lj = log_likes[k, c] + log_weights[k]
            if lj > max_val:
                max_val = lj

        denom = 0.0
        for k in range(K):
            posteriors[k, c] = np.exp(log_likes[k, c] + log_weights[k] - max_val)
            denom += posteriors[k, c]

        for k in range(K):
            posteriors[k, c] /= denom

    return posteriors


# ---------------------------------------------------------------------------
# 16. Branch mask (Steiner tree)
# ---------------------------------------------------------------------------

def compute_branch_mask(alignment, parentIndex, A):
    """Identify active branches per column (minimum Steiner tree of ungapped leaves).

    Args:
        alignment: (R, C) int32 tokens
        parentIndex: (R,) int32
        A: alphabet size

    Returns:
        branch_mask: (R, C) bool
    """
    R, C = alignment.shape

    # Determine which nodes are leaves
    child_count = np.zeros(R, dtype=int)
    for n in range(1, R):
        child_count[parentIndex[n]] += 1
    is_leaf = (child_count == 0)

    # Step 1: ungapped leaf classification
    is_ungapped_leaf = np.zeros((R, C), dtype=bool)
    for r in range(R):
        for c in range(C):
            tok = alignment[r, c]
            is_ungapped_leaf[r, c] = is_leaf[r] and (0 <= tok <= A)

    # Step 2: Upward pass — propagate "has ungapped descendant"
    has_ungapped = is_ungapped_leaf.copy()
    for n in range(R - 1, 0, -1):
        p = parentIndex[n]
        for c in range(C):
            if has_ungapped[n, c]:
                has_ungapped[p, c] = True

    # Step 3: Count ungapped children
    ungapped_child_count = np.zeros((R, C), dtype=int)
    for n in range(1, R):
        p = parentIndex[n]
        for c in range(C):
            if has_ungapped[n, c]:
                ungapped_child_count[p, c] += 1

    # Step 4: Steiner nodes — ungapped leaves and branching points
    is_steiner = np.zeros((R, C), dtype=bool)
    for r in range(R):
        for c in range(C):
            if is_ungapped_leaf[r, c] or ungapped_child_count[r, c] >= 2:
                is_steiner[r, c] = True

    # Step 5: Preorder propagation of Steiner membership
    for n in range(1, R):
        p = parentIndex[n]
        for c in range(C):
            if is_steiner[p, c] and has_ungapped[n, c]:
                is_steiner[n, c] = True

    # Step 6: Branch active if both endpoints in Steiner tree
    branch_mask = np.zeros((R, C), dtype=bool)
    for n in range(1, R):
        p = parentIndex[n]
        for c in range(C):
            branch_mask[n, c] = is_steiner[n, c] and is_steiner[p, c]

    return branch_mask


# ---------------------------------------------------------------------------
# 17-20. Public API
# ---------------------------------------------------------------------------

def _is_irrev(model):
    """Check if model is irreversible."""
    return not model.get('reversible', True)


def _get_sub_matrices(model, distances):
    """Get substitution matrices, dispatching on model type."""
    if _is_irrev(model):
        return compute_sub_matrices_irrev(model, distances)
    return compute_sub_matrices(model, distances)


def LogLike(alignment, tree, model):
    """Compute per-column log-likelihoods.

    Args:
        alignment: (R, C) int32 tokens
        tree: dict with 'parentIndex' (R,), 'distanceToParent' (R,)
        model: dict with 'eigenvalues', 'eigenvectors', 'pi'

    Returns:
        (C,) log-likelihoods
    """
    subMats = _get_sub_matrices(model, tree['distanceToParent'])
    _, _, logLike = upward_pass(alignment, tree, subMats, model['pi'])
    return logLike


def Counts(alignment, tree, model, f81_fast=False, branch_mask="auto"):
    """Compute expected substitution counts and dwell times per column.

    Args:
        alignment: (R, C) int32 tokens
        tree: dict with 'parentIndex', 'distanceToParent'
        model: dict with 'eigenvalues', 'eigenvectors', 'pi'
        f81_fast: if True, use O(CRA^2) F81/JC fast path
        branch_mask: "auto" (compute from alignment), None (no masking),
            or (R, C) bool array

    Returns:
        (A, A, C) counts tensor (diag=dwell, off-diag=substitution counts)
    """
    is_irrev = _is_irrev(model)

    if isinstance(branch_mask, str) and branch_mask == "auto":
        A = len(model['pi'])
        branch_mask = compute_branch_mask(alignment, tree['parentIndex'], A)

    if is_irrev:
        assert not f81_fast, "F81 fast path is reversible-only"

    subMats = _get_sub_matrices(model, tree['distanceToParent'])
    U, logNormU, logLike = upward_pass(alignment, tree, subMats, model['pi'])
    D, logNormD = downward_pass(U, logNormU, tree, subMats, model['pi'], alignment)

    if f81_fast:
        return f81_counts(
            U, D, logNormU, logNormD, logLike,
            tree['distanceToParent'], model['pi'], tree['parentIndex'],
            branch_mask=branch_mask,
        )
    elif is_irrev:
        J = compute_J_complex(model['eigenvalues'], tree['distanceToParent'])
        U_tilde, D_tilde = eigenbasis_project_irrev(U, D, model)
        C = accumulate_C_complex(D_tilde, U_tilde, J, logNormU, logNormD, logLike,
                                 tree['parentIndex'], branch_mask=branch_mask)
        return back_transform_irrev(C, model)
    else:
        J = compute_J(model['eigenvalues'], tree['distanceToParent'])
        U_tilde, D_tilde = eigenbasis_project(U, D, model)
        C = accumulate_C(D_tilde, U_tilde, J, logNormU, logNormD, logLike, tree['parentIndex'],
                         branch_mask=branch_mask)
        return back_transform(C, model)


def BranchCounts(alignment, tree, model, f81_fast=False, branch_mask="auto"):
    """Compute per-branch expected substitution counts and dwell times per column.

    Same as Counts but returns (R, A, A, C) with per-branch breakdowns.
    Branch 0 (root) is all zeros.

    Args:
        alignment: (R, C) int32 tokens
        tree: dict with 'parentIndex', 'distanceToParent'
        model: dict with 'eigenvalues', 'eigenvectors', 'pi'
        f81_fast: if True, use O(CRA^2) F81/JC fast path
        branch_mask: "auto" (compute from alignment), None (no masking),
            or (R, C) bool array

    Returns:
        (R, A, A, C) counts tensor per branch (diag=dwell, off-diag=substitution counts)
    """
    is_irrev = _is_irrev(model)

    if isinstance(branch_mask, str) and branch_mask == "auto":
        A = len(model['pi'])
        branch_mask = compute_branch_mask(alignment, tree['parentIndex'], A)

    if is_irrev:
        assert not f81_fast, "F81 fast path is reversible-only"

    subMats = _get_sub_matrices(model, tree['distanceToParent'])
    U, logNormU, logLike = upward_pass(alignment, tree, subMats, model['pi'])
    D, logNormD = downward_pass(U, logNormU, tree, subMats, model['pi'], alignment)

    if f81_fast:
        return f81_counts_per_branch(
            U, D, logNormU, logNormD, logLike,
            tree['distanceToParent'], model['pi'], tree['parentIndex'],
            branch_mask=branch_mask,
        )
    elif is_irrev:
        J = compute_J_complex(model['eigenvalues'], tree['distanceToParent'])
        U_tilde, D_tilde = eigenbasis_project_irrev(U, D, model)
        C = accumulate_C_complex_per_branch(
            D_tilde, U_tilde, J, logNormU, logNormD, logLike,
            tree['parentIndex'], branch_mask=branch_mask,
        )
        return back_transform_irrev_per_branch(C, model)
    else:
        J = compute_J(model['eigenvalues'], tree['distanceToParent'])
        U_tilde, D_tilde = eigenbasis_project(U, D, model)
        C = accumulate_C_per_branch(
            D_tilde, U_tilde, J, logNormU, logNormD, logLike,
            tree['parentIndex'], branch_mask=branch_mask,
        )
        return back_transform_per_branch(C, model)


def RootProb(alignment, tree, model):
    """Compute posterior root state distribution per column.

    Args:
        alignment: (R, C) int32 tokens
        tree: dict with 'parentIndex', 'distanceToParent'
        model: dict with 'eigenvalues', 'eigenvectors', 'pi'

    Returns:
        (A, C) posterior root distribution
    """
    subMats = _get_sub_matrices(model, tree['distanceToParent'])
    U, logNormU, logLike = upward_pass(alignment, tree, subMats, model['pi'])

    A = len(model['pi'])
    R, C, _ = U.shape
    q = np.zeros((A, C), dtype=np.float64)

    for c in range(C):
        log_scale = logNormU[0, c] - logLike[c]
        scale = np.exp(log_scale)
        for a in range(A):
            q[a, c] = model['pi'][a] * U[0, c, a] * scale

    return q


def MixturePosterior(alignment, tree, models, log_weights):
    """Compute posterior over mixture components per column.

    Args:
        alignment: (R, C) int32 tokens
        tree: dict with 'parentIndex', 'distanceToParent'
        models: list of K model dicts
        log_weights: (K,) log prior weights

    Returns:
        (K, C) posterior probabilities
    """
    K = len(models)
    C = alignment.shape[1]
    log_likes = np.zeros((K, C), dtype=np.float64)
    for k in range(K):
        log_likes[k, :] = LogLike(alignment, tree, models[k])
    return mixture_posterior(log_likes, log_weights)


class InsideOutside:
    """Inside-outside DP tables for querying posteriors (pure-Python oracle).

    Runs the upward (inside) and downward (outside) passes once and stores the
    resulting vectors, enabling efficient queries for log-likelihoods, expected
    substitution counts, node state posteriors, and branch endpoint joint
    posteriors without recomputation.

    Args:
        alignment: (R, C) int32 tokens
        tree: dict with 'parentIndex' (R,), 'distanceToParent' (R,)
        model: dict with 'eigenvalues', 'eigenvectors', 'pi'
    """

    def __init__(self, alignment, tree, model):
        self._alignment = alignment
        self._tree = tree
        self._model = model
        self._is_irrev = _is_irrev(model)

        subMats = _get_sub_matrices(model, tree['distanceToParent'])
        self._subMatrices = subMats

        U, logNormU, logLike = upward_pass(alignment, tree, subMats, model['pi'])
        self._U = U
        self._logNormU = logNormU
        self._logLike = logLike

        D, logNormD = downward_pass(U, logNormU, tree, subMats, model['pi'], alignment)

        # Set root's outside vector to the prior pi (downward pass leaves it zero)
        pi = model['pi']
        A = len(pi)
        C = alignment.shape[1]
        for c in range(C):
            max_val = max(pi)
            if max_val < 1e-300:
                max_val = 1e-300
            for a in range(A):
                D[0, c, a] = pi[a] / max_val
            logNormD[0, c] = np.log(max_val)

        self._D = D
        self._logNormD = logNormD

    @property
    def log_likelihood(self):
        """Per-column log-likelihoods. Shape: (C,)."""
        return self._logLike

    def counts(self, f81_fast=False, branch_mask="auto"):
        """Expected substitution counts and dwell times per column.

        Reuses stored inside-outside vectors.

        Args:
            f81_fast: if True, use O(CRA^2) F81/JC fast path
            branch_mask: "auto", None, or (R, C) bool array

        Returns:
            (A, A, C) counts tensor
        """
        model = self._model

        if isinstance(branch_mask, str) and branch_mask == "auto":
            A = len(model['pi'])
            branch_mask = compute_branch_mask(
                self._alignment, self._tree['parentIndex'], A
            )

        if f81_fast:
            return f81_counts(
                self._U, self._D, self._logNormU, self._logNormD,
                self._logLike, self._tree['distanceToParent'], model['pi'],
                self._tree['parentIndex'], branch_mask=branch_mask,
            )
        elif self._is_irrev:
            J = compute_J_complex(model['eigenvalues'], self._tree['distanceToParent'])
            U_tilde, D_tilde = eigenbasis_project_irrev(self._U, self._D, model)
            C = accumulate_C_complex(
                D_tilde, U_tilde, J, self._logNormU, self._logNormD,
                self._logLike, self._tree['parentIndex'], branch_mask=branch_mask,
            )
            return back_transform_irrev(C, model)
        else:
            J = compute_J(model['eigenvalues'], self._tree['distanceToParent'])
            U_tilde, D_tilde = eigenbasis_project(self._U, self._D, model)
            C = accumulate_C(
                D_tilde, U_tilde, J, self._logNormU, self._logNormD,
                self._logLike, self._tree['parentIndex'], branch_mask=branch_mask,
            )
            return back_transform(C, model)

    def branch_counts(self, f81_fast=False, branch_mask="auto"):
        """Per-branch expected substitution counts and dwell times per column.

        Same as counts() but returns (R, A, A, C) with per-branch breakdowns.
        Branch 0 (root) is all zeros.

        Args:
            f81_fast: if True, use O(CRA^2) F81/JC fast path
            branch_mask: "auto", None, or (R, C) bool array

        Returns:
            (R, A, A, C) counts tensor per branch
        """
        model = self._model

        if isinstance(branch_mask, str) and branch_mask == "auto":
            A = len(model['pi'])
            branch_mask = compute_branch_mask(
                self._alignment, self._tree['parentIndex'], A
            )

        if f81_fast:
            return f81_counts_per_branch(
                self._U, self._D, self._logNormU, self._logNormD,
                self._logLike, self._tree['distanceToParent'], model['pi'],
                self._tree['parentIndex'], branch_mask=branch_mask,
            )
        elif self._is_irrev:
            J = compute_J_complex(model['eigenvalues'], self._tree['distanceToParent'])
            U_tilde, D_tilde = eigenbasis_project_irrev(self._U, self._D, model)
            C = accumulate_C_complex_per_branch(
                D_tilde, U_tilde, J, self._logNormU, self._logNormD,
                self._logLike, self._tree['parentIndex'], branch_mask=branch_mask,
            )
            return back_transform_irrev_per_branch(C, model)
        else:
            J = compute_J(model['eigenvalues'], self._tree['distanceToParent'])
            U_tilde, D_tilde = eigenbasis_project(self._U, self._D, model)
            C = accumulate_C_per_branch(
                D_tilde, U_tilde, J, self._logNormU, self._logNormD,
                self._logLike, self._tree['parentIndex'], branch_mask=branch_mask,
            )
            return back_transform_per_branch(C, model)

    def node_posterior(self, node=None):
        """Posterior state distribution at node(s).

        For root: P(X_0 = a | data) ∝ pi_a * U(0,c,a)
        For non-root: P(X_n = j | data) ∝ [sum_a D(n,c,a) * M(n,a,j)] * U(n,c,j)

        D(n,a) is indexed by the parent's state a, so we transform through
        the branch matrix M(n) to get the child state j.

        Args:
            node: int node index, or None for all nodes.

        Returns:
            If node is int: (A, C) posterior distribution
            If node is None: (R, A, C) posteriors for all nodes
        """
        R, C, A = self._U.shape

        if node is not None:
            posterior = np.zeros((A, C), dtype=np.float64)
            for c in range(C):
                log_scale = (
                    self._logNormU[node, c]
                    + self._logNormD[node, c]
                    - self._logLike[c]
                )
                scale = np.exp(log_scale)
                total = 0.0
                if node == 0:
                    # Root: D[0] = pi, direct product
                    for a in range(A):
                        val = self._U[node, c, a] * self._D[node, c, a] * scale
                        posterior[a, c] = val
                        total += val
                else:
                    # Non-root: transform D through M
                    M = self._subMatrices[node]  # (A, A)
                    for j in range(A):
                        d_transformed = 0.0
                        for a in range(A):
                            d_transformed += self._D[node, c, a] * M[a, j]
                        val = d_transformed * self._U[node, c, j] * scale
                        posterior[j, c] = val
                        total += val
                if total > 0:
                    for a in range(A):
                        posterior[a, c] /= total
            return posterior
        else:
            result = np.zeros((R, A, C), dtype=np.float64)
            for n in range(R):
                for c in range(C):
                    log_scale = (
                        self._logNormU[n, c]
                        + self._logNormD[n, c]
                        - self._logLike[c]
                    )
                    scale = np.exp(log_scale)
                    total = 0.0
                    if n == 0:
                        for a in range(A):
                            val = self._U[n, c, a] * self._D[n, c, a] * scale
                            result[n, a, c] = val
                            total += val
                    else:
                        M = self._subMatrices[n]
                        for j in range(A):
                            d_transformed = 0.0
                            for a in range(A):
                                d_transformed += self._D[n, c, a] * M[a, j]
                            val = d_transformed * self._U[n, c, j] * scale
                            result[n, j, c] = val
                            total += val
                    if total > 0:
                        for a in range(A):
                            result[n, a, c] /= total
            return result

    def branch_posterior(self, node=None):
        """Joint posterior of parent-child states on branch(es).

        P(X_{parent(n)}=i, X_n=j | data, c) ∝ D(n,c,i) * M(n,i,j) * U(n,c,j)

        D(n,a) is indexed by the parent's state a (the outside context at
        branch n), so D(n) is used directly (not D at the parent node).

        Args:
            node: int child node index (must be > 0), or None for all branches.

        Returns:
            If node is int: (A, A, C) where [i,j,c] = P(parent=i, child=j)
            If node is None: (R, A, A, C) for all branches (branch 0 is zeros)
        """
        R, C, A = self._U.shape

        if node is not None:
            M = self._subMatrices[node]  # (A, A)
            joint = np.zeros((A, A, C), dtype=np.float64)
            for c in range(C):
                log_scale = (
                    self._logNormD[node, c]
                    + self._logNormU[node, c]
                    - self._logLike[c]
                )
                scale = np.exp(log_scale)
                total = 0.0
                for i in range(A):
                    for j in range(A):
                        val = (
                            self._D[node, c, i]
                            * M[i, j]
                            * self._U[node, c, j]
                            * scale
                        )
                        joint[i, j, c] = val
                        total += val
                if total > 0:
                    for i in range(A):
                        for j in range(A):
                            joint[i, j, c] /= total
            return joint
        else:
            result = np.zeros((R, A, A, C), dtype=np.float64)
            for n in range(1, R):
                M = self._subMatrices[n]
                for c in range(C):
                    log_scale = (
                        self._logNormD[n, c]
                        + self._logNormU[n, c]
                        - self._logLike[c]
                    )
                    scale = np.exp(log_scale)
                    total = 0.0
                    for i in range(A):
                        for j in range(A):
                            val = (
                                self._D[n, c, i]
                                * M[i, j]
                                * self._U[n, c, j]
                                * scale
                            )
                            result[n, i, j, c] = val
                            total += val
                    if total > 0:
                        for i in range(A):
                            for j in range(A):
                                result[n, i, j, c] /= total
            return result


# ---------------------------------------------------------------------------
# Standalone CTMC branch integral functions
# ---------------------------------------------------------------------------

def expected_counts_eigen(eigenvalues, eigenvectors, pi, t):
    """Expected counts from pre-computed eigendecomposition (reversible).

    For a reversible CTMC with symmetrized rate matrix S = V diag(mu) V^T:

    W[a,b,i,j] = sum_{kl} V[a,k]*V[i,k] * J[k,l](t) * V[b,l]*V[j,l]
    S_t[a,b] = sum_k V[a,k] exp(mu_k*t) V[b,k]  (symmetrized transition matrix)
    S[i,j] = sum_k V[i,k] mu_k V[j,k]

    result[a,b,i,i] = W[a,b,i,i] / S_t[a,b]                    (dwell)
    result[a,b,i,j] = S[i,j] * W[a,b,i,j] / S_t[a,b]          (subs, i!=j)

    Args:
        eigenvalues: (A,) real eigenvalues
        eigenvectors: (A, A) real orthogonal eigenvectors
        pi: (A,) equilibrium distribution
        t: scalar branch length

    Returns:
        (A, A, A, A) tensor: result[a, b, i, j]
    """
    A = len(eigenvalues)
    mu = eigenvalues
    V = eigenvectors

    # J matrix for a single branch
    J = np.zeros((A, A), dtype=np.float64)
    for k in range(A):
        exp_k = np.exp(mu[k] * t)
        for l in range(A):
            diff = mu[k] - mu[l]
            if abs(diff) < 1e-8:
                J[k, l] = t * exp_k
            else:
                J[k, l] = (exp_k - np.exp(mu[l] * t)) / diff

    # Symmetrized transition matrix: S_t[a,b] = sum_k V[a,k] exp(mu_k*t) V[b,k]
    S_t = np.zeros((A, A), dtype=np.float64)
    for a in range(A):
        for b in range(A):
            s = 0.0
            for k in range(A):
                s += V[a, k] * np.exp(mu[k] * t) * V[b, k]
            S_t[a, b] = s

    # Symmetrized rate matrix: S[i,j] = sum_k V[i,k] mu_k V[j,k]
    S_mat = np.zeros((A, A), dtype=np.float64)
    for i in range(A):
        for j in range(A):
            s = 0.0
            for k in range(A):
                s += V[i, k] * mu[k] * V[j, k]
            S_mat[i, j] = s

    # W[a,b,i,j] = sum_{kl} V[a,k]*V[i,k] * J[k,l] * V[b,l]*V[j,l]
    W = np.zeros((A, A, A, A), dtype=np.float64)
    for a in range(A):
        for b in range(A):
            for i in range(A):
                for j in range(A):
                    s = 0.0
                    for k in range(A):
                        for l in range(A):
                            s += V[a, k] * V[i, k] * J[k, l] * V[b, l] * V[j, l]
                    W[a, b, i, j] = s

    # Build result (divide by S_t, not M, in the symmetrized basis)
    result = np.zeros((A, A, A, A), dtype=np.float64)
    for a in range(A):
        for b in range(A):
            if abs(S_t[a, b]) < 1e-300:
                continue
            for i in range(A):
                for j in range(A):
                    if i == j:
                        result[a, b, i, j] = W[a, b, i, j] / S_t[a, b]
                    else:
                        result[a, b, i, j] = S_mat[i, j] * W[a, b, i, j] / S_t[a, b]

    return result


def expected_counts_eigen_irrev(eigenvalues, eigenvectors, eigenvectors_inv, pi, t):
    """Expected counts from pre-computed eigendecomposition (irreversible).

    For an irreversible CTMC with Q = V diag(mu) V^{-1}:

    L[a,i,k] = V[a,k] * V_inv[k,i]
    R[b,j,l] = V[j,l] * V_inv[l,b]
    W[a,b,i,j] = sum_{kl} L[a,i,k] * J[k,l] * R[b,j,l]
    M[a,b] = Re(sum_k V[a,k] exp(mu_k*t) V_inv[k,b])
    Q[i,j] = Re(sum_k V[i,k] mu_k V_inv[k,j])

    result[a,b,i,i] = Re(W[a,b,i,i]) / M[a,b]
    result[a,b,i,j] = Re(Q[i,j] * W[a,b,i,j]) / M[a,b]      (i!=j)

    Args:
        eigenvalues: (A,) complex eigenvalues
        eigenvectors: (A, A) complex right eigenvectors V
        eigenvectors_inv: (A, A) complex V^{-1}
        pi: (A,) real equilibrium distribution
        t: scalar branch length

    Returns:
        (A, A, A, A) real tensor: result[a, b, i, j]
    """
    A = len(eigenvalues)
    mu = eigenvalues
    V = eigenvectors
    V_inv = eigenvectors_inv

    # J matrix (complex) for a single branch
    J = np.zeros((A, A), dtype=np.complex128)
    for k in range(A):
        exp_k = np.exp(mu[k] * t)
        for l in range(A):
            diff = mu[k] - mu[l]
            if abs(diff) < 1e-8:
                J[k, l] = t * exp_k
            else:
                J[k, l] = (exp_k - np.exp(mu[l] * t)) / diff

    # Transition matrix: M[a,b] = Re(sum_k V[a,k] exp(mu_k*t) V_inv[k,b])
    M = np.zeros((A, A), dtype=np.float64)
    for a in range(A):
        for b in range(A):
            s = 0.0 + 0.0j
            for k in range(A):
                s += V[a, k] * np.exp(mu[k] * t) * V_inv[k, b]
            M[a, b] = s.real

    # Rate matrix: Q[i,j] = sum_k V[i,k] mu_k V_inv[k,j]
    Q_mat = np.zeros((A, A), dtype=np.complex128)
    for i in range(A):
        for j in range(A):
            s = 0.0 + 0.0j
            for k in range(A):
                s += V[i, k] * mu[k] * V_inv[k, j]
            Q_mat[i, j] = s

    # W[a,b,i,j] = sum_{kl} V[a,k]*V_inv[k,i] * J[k,l] * V[j,l]*V_inv[l,b]
    W = np.zeros((A, A, A, A), dtype=np.complex128)
    for a in range(A):
        for b in range(A):
            for i in range(A):
                for j in range(A):
                    s = 0.0 + 0.0j
                    for k in range(A):
                        for l in range(A):
                            s += V[a, k] * V_inv[k, i] * J[k, l] * V[j, l] * V_inv[l, b]
                    W[a, b, i, j] = s

    # Build result
    result = np.zeros((A, A, A, A), dtype=np.float64)
    for a in range(A):
        for b in range(A):
            if abs(M[a, b]) < 1e-300:
                continue
            for i in range(A):
                for j in range(A):
                    if i == j:
                        result[a, b, i, j] = W[a, b, i, j].real / M[a, b]
                    else:
                        result[a, b, i, j] = (Q_mat[i, j] * W[a, b, i, j]).real / M[a, b]

    return result


def ExpectedCounts(model, t):
    """Expected substitution counts and dwell times for a single CTMC branch.

    Outer function: accepts reversible or irreversible model dict.
    Auto-detects model type from 'reversible' flag or presence of 'eigenvectors_inv'.

    Args:
        model: dict with eigenvalues, eigenvectors, pi (and eigenvectors_inv if irreversible)
        t: scalar branch length

    Returns:
        (A, A, A, A) tensor: result[a, b, i, j]
        Diagonal [a,b,i,i] = dwell time; off-diagonal [a,b,i,j] = sub count.
    """
    is_irrev = model.get('reversible', True) is False or 'eigenvectors_inv' in model
    if is_irrev:
        return expected_counts_eigen_irrev(
            model['eigenvalues'], model['eigenvectors'],
            model['eigenvectors_inv'], model['pi'], t,
        )
    else:
        return expected_counts_eigen(
            model['eigenvalues'], model['eigenvectors'], model['pi'], t,
        )
