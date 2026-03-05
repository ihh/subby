from __future__ import annotations

import jax
import jax.numpy as jnp


def validate_binary_tree(parentIndex: jnp.ndarray) -> None:
    """Assert every non-leaf node has exactly 2 children. Raise on violation."""
    R = parentIndex.shape[0]
    child_count = jnp.zeros(R, dtype=jnp.int32)
    child_count = child_count.at[parentIndex[1:]].add(1)
    # Root (node 0) can have 2 children; leaves have 0
    # Internal nodes must have exactly 2
    has_children = child_count > 0
    valid = jnp.all(child_count[has_children] == 2)
    if not valid:
        raise ValueError(
            "Tree is not binary: every internal node must have exactly 2 children. "
            f"Child counts: {child_count}"
        )


def children_of(
    parentIndex: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute left_child, right_child, sibling arrays for a binary tree.

    Returns:
        left_child: (R,) int32, left child of each node (-1 if leaf)
        right_child: (R,) int32, right child of each node (-1 if leaf)
        sibling: (R,) int32, sibling of each node (-1 for root)
    """
    R = parentIndex.shape[0]
    nodes = jnp.arange(1, R, dtype=jnp.int32)
    parent = parentIndex[1:]

    # left_child[p] = min child index, right_child[p] = max child index
    left_child = jnp.full(R, R, dtype=jnp.int32)
    left_child = left_child.at[parent].min(nodes)
    left_child = jnp.where(left_child >= R, -1, left_child)

    right_child = jnp.full(R, -1, dtype=jnp.int32)
    right_child = right_child.at[parent].max(nodes)

    # Sibling: for node n with parent p, sibling = the other child of p
    all_nodes = jnp.arange(R, dtype=jnp.int32)
    p = parentIndex
    is_left = all_nodes == left_child[p]
    sibling = jnp.where(is_left, right_child[p], left_child[p])
    sibling = sibling.at[0].set(-1)  # root has no sibling

    return left_child, right_child, sibling


def token_to_likelihood(alignment: jnp.ndarray, A: int) -> jnp.ndarray:
    """Convert integer token alignment to likelihood vectors.

    Token encoding:
        0..A-1 = observed (one-hot)
        A      = ungapped-unobserved (all ones)
        A+1    = gapped (all ones)
        -1     = gap (legacy, all ones)

    Args:
        alignment: (R, C) int32 tokens
        A: alphabet size

    Returns:
        (R, C, A) float32 likelihood vectors
    """
    # Build lookup: row 0 = all-ones (for -1 token), rows 1..A = one-hot, row A+1 = all-ones, row A+2 = all-ones
    lookup = jnp.concatenate([
        jnp.ones((1, A)),           # index 0: token -1 (gap, legacy)
        jnp.eye(A),                 # indices 1..A: tokens 0..A-1 (observed)
        jnp.ones((1, A)),           # index A+1: token A (ungapped-unobserved)
        jnp.ones((1, A)),           # index A+2: token A+1 (gapped)
    ])
    return lookup[alignment + 1]


def pad_alignment(
    alignment: jnp.ndarray, bin_size: int = 128,
) -> tuple[jnp.ndarray, int]:
    """Pad alignment columns to the next multiple of bin_size with gap tokens.

    Gap-padded columns (-1) produce all-ones likelihood vectors, so they are
    mathematically neutral: logL=0, zero counts, root prior unchanged.
    Padding avoids JAX recompilation when C varies across inputs.

    Args:
        alignment: (R, C) int32 tokens
        bin_size: round C up to the next multiple of this

    Returns:
        (padded_alignment, C_original) where padded_alignment has
        C_padded = ceil(C / bin_size) * bin_size columns.
    """
    C_original = alignment.shape[1]
    remainder = C_original % bin_size
    if remainder == 0:
        return alignment, C_original
    pad_width = bin_size - remainder
    padding = jnp.full((alignment.shape[0], pad_width), -1, dtype=alignment.dtype)
    return jnp.concatenate([alignment, padding], axis=1), C_original


def unpad_columns(result: jnp.ndarray, C_original: int) -> jnp.ndarray:
    """Strip padding columns from a result array.

    Works for any array whose last axis is the column dimension C.

    Args:
        result: (..., C_padded) array
        C_original: original number of columns before padding

    Returns:
        (..., C_original) array
    """
    return result[..., :C_original]


def rescale(
    vec: jnp.ndarray, log_norm: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rescale vector to prevent underflow, accumulate log-normalizer.

    Args:
        vec: (..., A) float array
        log_norm: (...) float array, running log-normalizer

    Returns:
        (vec_rescaled, log_norm_updated)
    """
    max_val = jnp.max(vec, axis=-1)
    max_val = jnp.maximum(max_val, 1e-300)  # avoid log(0)
    vec_rescaled = vec / max_val[..., None]
    log_norm_updated = log_norm + jnp.log(max_val)
    return vec_rescaled, log_norm_updated
