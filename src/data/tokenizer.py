from typing import NamedTuple
import jax
import jax.numpy as jnp


class TokenizedMSA(NamedTuple):
    subs: jnp.ndarray      # (R, C) int32 — SUBS tokens: 0..3=ACGT, 4=ungapped-unobs, 5=gapped
    triplets: jnp.ndarray  # (R, C) int32 — TRIPLETS tokens: 0..63=triplet, 64=gapped
    phase: jnp.ndarray     # (R, C) int32 — PHASE tokens: 0=NUC, 1/2/3=gap mod 3
    annot: jnp.ndarray     # (R, C) int32 — ANNOT tokens: 0..M-1=label, M=ungapped-unobs, M+1=gapped


def tokenize_msa(msa_int, annotations=None, root_row=0, M=15):
    """Pure-JAX FSM tokenizer producing 4 parallel token streams.

    Pass 1 (left-to-right): emit SUBS, PHASE tokens, track prev_nongap_nuc.
    Pass 2 (right-to-left): find next_nongap_nuc.
    Combine passes for TRIPLETS.
    ANNOT from annotations array.

    Args:
        msa_int: (R, C) int32 — {A=0, C=1, G=2, T=3, N=4, gap=5}
        annotations: (R, C) int32 or None — annotation labels {0..M-1} or -1 for absent
        root_row: int — index of the root (target) row
        M: int — number of annotation labels

    Returns:
        TokenizedMSA with .subs, .triplets, .phase, .annot — each (R, C) int32
    """
    R, C = msa_int.shape

    # === SUBS tokens ===
    # A=0,C=1,G=2,T=3 -> 0..3, N=4 -> 4 (ungapped-unobs), gap=5 -> 5 (gapped)
    subs = msa_int  # already in the right encoding

    # === PHASE tokens ===
    # NUC (ungapped) -> 0, gap -> 1 + gap_len_mod3
    # Need gap_len_mod3 per row via scan
    phase = _compute_phase(msa_int)  # (R, C)

    # === TRIPLETS tokens ===
    # Need prev and next non-gap nucleotide per position per row
    prev_nuc = _compute_prev_nongap(msa_int)    # (R, C) int32, -1 if BOS
    next_nuc = _compute_next_nongap(msa_int)    # (R, C) int32, -1 if EOS

    triplets = _compute_triplets(msa_int, prev_nuc, next_nuc)  # (R, C)

    # === ANNOT tokens ===
    annot = _compute_annot(msa_int, annotations, root_row, M)  # (R, C)

    return TokenizedMSA(subs=subs, triplets=triplets, phase=phase, annot=annot)


def _compute_phase(msa_int):
    """Compute PHASE tokens: 0=NUC, 1+gap_len_mod3 for gaps."""
    R, C = msa_int.shape
    is_gap = (msa_int == 5)  # gap token

    def _phase_row(row):
        """Scan a single row to compute phase tokens."""
        def _step(gap_len_mod3, token):
            is_g = (token == 5)
            new_gap_len = jnp.where(is_g, (gap_len_mod3 + 1) % 3, 0)
            phase_token = jnp.where(is_g, 1 + gap_len_mod3, 0)
            return new_gap_len, phase_token

        _, phase_tokens = jax.lax.scan(_step, jnp.int32(0), row)
        return phase_tokens

    return jax.vmap(_phase_row)(msa_int)  # (R, C)


def _compute_prev_nongap(msa_int):
    """Left-to-right scan to find previous non-gap nucleotide (0..3). -1 if BOS or N."""
    def _row_scan(row):
        def _step(prev_nuc, token):
            is_nuc = (token >= 0) & (token <= 3)  # A,C,G,T
            new_prev = jnp.where(is_nuc, token, prev_nuc)
            return new_prev, prev_nuc  # emit the prev BEFORE update

        _, prev_nucs = jax.lax.scan(_step, jnp.int32(-1), row)
        return prev_nucs

    return jax.vmap(_row_scan)(msa_int)  # (R, C)


def _compute_next_nongap(msa_int):
    """Right-to-left scan to find next non-gap nucleotide (0..3). -1 if EOS or N."""
    def _row_scan(row):
        def _step(next_nuc, token):
            is_nuc = (token >= 0) & (token <= 3)  # A,C,G,T
            new_next = jnp.where(is_nuc, token, next_nuc)
            return new_next, next_nuc  # emit the next BEFORE update

        # Scan right-to-left by reversing
        _, next_nucs = jax.lax.scan(_step, jnp.int32(-1), row[::-1])
        return next_nucs[::-1]

    return jax.vmap(_row_scan)(msa_int)  # (R, C)


def _compute_triplets(msa_int, prev_nuc, next_nuc):
    """Compute TRIPLET tokens: prev*16 + curr*4 + next, or 64 if gapped/BOS/EOS/N."""
    curr = msa_int  # (R, C)
    is_valid_curr = (curr >= 0) & (curr <= 3)
    is_valid_prev = (prev_nuc >= 0) & (prev_nuc <= 3)
    is_valid_next = (next_nuc >= 0) & (next_nuc <= 3)
    is_valid = is_valid_curr & is_valid_prev & is_valid_next

    triplet_token = prev_nuc * 16 + curr * 4 + next_nuc
    return jnp.where(is_valid, triplet_token, 64).astype(jnp.int32)


def _compute_annot(msa_int, annotations, root_row, M):
    """Compute ANNOT tokens.

    Non-root rows: annotation label if present (0..M-1), else M+1 (gapped).
    Root row: M (ungapped-unobserved) for non-gap positions, M+1 (gapped) for gaps.
    """
    R, C = msa_int.shape

    if annotations is None:
        # No annotations: non-root is all gapped (M+1), root is ungapped-unobs (M) or gapped (M+1)
        annot = jnp.full((R, C), M + 1, dtype=jnp.int32)
        is_root = jnp.arange(R)[:, None] == root_row
        is_ungapped = (msa_int >= 0) & (msa_int <= 4)  # A,C,G,T,N are ungapped
        annot = jnp.where(is_root & is_ungapped, M, annot)
        return annot

    # annotations is (R, C): 0..M-1 = label, -1 = absent
    has_annot = (annotations >= 0) & (annotations < M)
    annot = jnp.where(has_annot, annotations, M + 1).astype(jnp.int32)

    # Root row: ungapped-unobserved (M) for non-gap, gapped (M+1) for gap
    is_root = jnp.arange(R)[:, None] == root_row
    is_ungapped = (msa_int >= 0) & (msa_int <= 4)
    annot = jnp.where(is_root & is_ungapped, M, annot)
    annot = jnp.where(is_root & ~is_ungapped, M + 1, annot)

    return annot
