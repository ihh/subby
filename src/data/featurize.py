import jax.numpy as jnp

from ..phylo.jax import (
    Tree, DiagModel, LogLike, Counts, RootProb, MixturePosterior,
)
from ..phylo.jax.models import hky85_diag, jukes_cantor_model, f81_model, gamma_rate_categories, scale_model
from ..phylo.jax.mixture import mixture_posterior
from .tokenizer import tokenize_msa


def extract_features(msa_int, tree, annotations=None, root_row=0, M=15,
                     kappa=2.0, pi_nuc=None, alpha=1.0, K=4):
    """Extract all featurization streams and concatenate.

    Output feature count: 12 + 4 + K + 64 + 64 + 12 + M + M + M = 92 + K + 3M

    Args:
        msa_int: (R, C) int32 — {A=0,C=1,G=2,T=3,N=4,gap=5}
        tree: Tree
        annotations: (R, C) int32 or None
        root_row: int
        M: number of annotation labels
        kappa: HKY85 transition/transversion ratio
        pi_nuc: (4,) nucleotide frequencies, or None for uniform
        alpha: gamma shape parameter for rate categories
        K: number of rate categories

    Returns:
        (F, C) float feature matrix
    """
    if pi_nuc is None:
        pi_nuc = jnp.ones(4) / 4.0
    pi_nuc = jnp.asarray(pi_nuc, dtype=jnp.float64)

    R, C = msa_int.shape

    # Tokenize
    tokens = tokenize_msa(msa_int, annotations, root_row, M)

    # === [SUBS] features: 12 off-diag subs + 4 dwell + K rate posteriors ===
    subs_features = _subs_features(tokens.subs, tree, kappa, pi_nuc, alpha, K)

    # === [TRIPLETS] features: 64 subs-to-triplet + 64 dwell ===
    triplet_features = _triplet_features(tokens.triplets, tree)

    # === [PHASE] features: 12 off-diag subs ===
    phase_features = _phase_features(tokens.phase, tree)

    # === [ANNOT] features: M subs + M dwell + M root posterior ===
    annot_features = _annot_features(tokens.annot, tree, M)

    # Concatenate all features: (F, C)
    all_features = jnp.concatenate([
        subs_features,     # 16 + K
        triplet_features,  # 128
        phase_features,    # 12
        annot_features,    # 3*M
    ], axis=0)

    return all_features


def _subs_features(subs_tokens, tree, kappa, pi_nuc, alpha, K):
    """[SUBS] features: mixture of K rate-scaled HKY85 models.

    Returns: (16+K, C) features
    """
    # Build HKY85 base model
    base_model = hky85_diag(kappa, pi_nuc)

    # Gamma rate categories
    rates, weights = gamma_rate_categories(alpha, K)
    log_weights = jnp.log(weights)

    # K scaled models
    models = [scale_model(base_model, r) for r in rates]

    # Log-likelihoods per component
    log_likes = jnp.stack([
        LogLike(subs_tokens, tree, m) for m in models
    ])  # (K, C)
    # Squeeze if there are extra dims
    while log_likes.ndim > 2:
        log_likes = log_likes[..., 0, :]  # remove H dim if singleton

    # Mixture posteriors: (K, C)
    posteriors = mixture_posterior(log_likes, log_weights)

    # Weighted counts: sum_k posterior_k * Counts_k
    counts_list = [Counts(subs_tokens, tree, m) for m in models]
    # Each counts is (*H, 4, 4, C)
    # Squeeze H dims
    counts_list = [c.squeeze() if c.ndim > 3 else c for c in counts_list]

    # Weighted average
    weighted_counts = sum(
        posteriors[k, None, None, :] * counts_list[k] for k in range(K)
    )  # (4, 4, C)

    # Extract features
    A = 4
    # Off-diagonal substitution counts: 12 features
    mask_offdiag = ~jnp.eye(A, dtype=bool)
    subs_counts = weighted_counts[mask_offdiag]  # (12, C)

    # Diagonal dwell times: 4 features
    dwell = jnp.array([weighted_counts[i, i] for i in range(A)])  # (4, C)

    # Stack: subs_counts (12,C) + dwell (4,C) + posteriors (K,C)
    return jnp.concatenate([subs_counts, dwell, posteriors], axis=0)  # (16+K, C)


def _triplet_features(triplet_tokens, tree):
    """[TRIPLETS] features: JC(64) model.

    Returns: (128, C) features — 64 subs-to-triplet + 64 dwell
    """
    A = 64
    model = jukes_cantor_model(A)
    counts = Counts(triplet_tokens, tree, model, f81_fast_flag=True)  # (64, 64, C)

    # Substitution counts to each triplet: sum over source states for each dest
    # subs_to_j = sum_{i!=j} counts[i, j]
    mask_offdiag = ~jnp.eye(A, dtype=bool)
    subs_to = jnp.sum(counts * mask_offdiag[:, :, None], axis=0)  # (64, C)

    # Dwell times
    dwell = jnp.array([counts[i, i] for i in range(A)])  # (64, C)

    return jnp.concatenate([subs_to, dwell], axis=0)  # (128, C)


def _phase_features(phase_tokens, tree):
    """[PHASE] features: JC(4) model.

    Returns: (12, C) features — 12 off-diagonal subs counts
    """
    A = 4
    model = jukes_cantor_model(A)
    counts = Counts(phase_tokens, tree, model, f81_fast_flag=True)  # (4, 4, C)

    mask_offdiag = ~jnp.eye(A, dtype=bool)
    subs_counts = counts[mask_offdiag]  # (12, C)

    return subs_counts


def _annot_features(annot_tokens, tree, M):
    """[ANNOT] features: JC(M) model.

    Returns: (3*M, C) features — M subs + M dwell + M root posterior
    """
    model = jukes_cantor_model(M)
    counts = Counts(annot_tokens, tree, model, f81_fast_flag=True)  # (M, M, C)

    # Subs to each label: sum_{i!=j} counts[i, j] for each j
    mask_offdiag = ~jnp.eye(M, dtype=bool)
    subs_to = jnp.sum(counts * mask_offdiag[:, :, None], axis=0)  # (M, C)

    # Dwell times
    dwell = jnp.array([counts[i, i] for i in range(M)])  # (M, C)

    # Root posterior
    root_prob = RootProb(annot_tokens, tree, model)  # (M, C)

    return jnp.concatenate([subs_to, dwell, root_prob], axis=0)  # (3*M, C)
