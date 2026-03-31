from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..formats import Tree


# --- Geometric bin sizes for JIT cache reuse ---
# Approximately 1.19^k, giving O(log n) distinct sizes up to any n.
GEOM_BINS: list[int] = sorted(set(
    max(1, round(1.19 ** k)) for k in range(120)
))


def pad_to_geom_bin(n: int) -> int:
    """Return the smallest geometric bin size >= n.

    Uses a precomputed sequence of ~1.19^k values so the number of distinct
    padded sizes grows as O(log n), dramatically reducing JIT recompilation.
    """
    for b in GEOM_BINS:
        if b >= n:
            return b
    # Beyond precomputed bins: round up to next power of 2
    return int(2 ** np.ceil(np.log2(max(n, 1))))


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
    alignment: jnp.ndarray,
    bin_size: int = 128,
    bin_fn=None,
) -> tuple[jnp.ndarray, int]:
    """Pad alignment columns to a bin size with gap tokens.

    Gap-padded columns (-1) produce all-ones likelihood vectors, so they are
    mathematically neutral: logL=0, zero counts, root prior unchanged.
    Padding avoids JAX recompilation when C varies across inputs.

    Args:
        alignment: (R, C) int32 tokens
        bin_size: round C up to the next multiple of this (used only when
            bin_fn is None)
        bin_fn: optional callable(int) -> int that returns padded size.
            Pass ``pad_to_geom_bin`` for geometric bins.  When provided,
            ``bin_size`` is ignored.

    Returns:
        (padded_alignment, C_original) where padded_alignment has
        C_padded columns.
    """
    C_original = alignment.shape[1]
    if bin_fn is not None:
        C_padded = bin_fn(C_original)
    else:
        remainder = C_original % bin_size
        C_padded = C_original if remainder == 0 else C_original + bin_size - remainder
    if C_padded == C_original:
        return alignment, C_original
    pad_width = C_padded - C_original
    padding = jnp.full((alignment.shape[0], pad_width), -1, dtype=alignment.dtype)
    return jnp.concatenate([alignment, padding], axis=1), C_original


def pad_tree(tree: Tree, n_nodes_padded: int) -> Tree:
    """Pad a Tree to n_nodes_padded nodes.

    Dummy nodes point to the root (parentIndex=0) with zero branch length.
    They act as leaves with unobserved data (all-ones likelihood), so they
    do not affect the pruning computation.

    Args:
        tree: Tree namedtuple with R nodes
        n_nodes_padded: target number of nodes (>= R)

    Returns:
        Tree with n_nodes_padded nodes
    """
    R = tree.parentIndex.shape[0]
    if n_nodes_padded <= R:
        return tree
    n_pad = n_nodes_padded - R
    pad_parent = jnp.zeros(n_pad, dtype=tree.parentIndex.dtype)
    pad_dist = jnp.zeros(n_pad, dtype=tree.distanceToParent.dtype)
    return Tree(
        parentIndex=jnp.concatenate([tree.parentIndex, pad_parent]),
        distanceToParent=jnp.concatenate([tree.distanceToParent, pad_dist]),
    )


def pad_tree_and_alignment(
    tree: Tree,
    alignment: jnp.ndarray,
    node_bin_fn=None,
    col_bin_fn=None,
) -> tuple[Tree, jnp.ndarray, int, int]:
    """Pad both tree (nodes) and alignment (nodes + columns) to bin sizes.

    Padded tree nodes are dummy leaves pointing to root with zero branch
    length and unobserved (gap = -1) tokens, so they are mathematically
    neutral in pruning.

    Args:
        tree: Tree namedtuple with R nodes
        alignment: (R, C) int32 tokens
        node_bin_fn: callable(int) -> int for node padding.
            Default: ``pad_to_geom_bin``.
        col_bin_fn: callable(int) -> int for column padding.
            Default: ``pad_to_geom_bin``.

    Returns:
        (padded_tree, padded_alignment, n_real_nodes, n_real_cols)
    """
    if node_bin_fn is None:
        node_bin_fn = pad_to_geom_bin
    if col_bin_fn is None:
        col_bin_fn = pad_to_geom_bin

    R, C = alignment.shape[0], alignment.shape[1]
    R_pad = node_bin_fn(R)
    C_pad = col_bin_fn(C)

    # Pad tree nodes
    padded_tree = pad_tree(tree, R_pad)

    # Pad alignment: first add rows (dummy nodes get gap tokens)
    if R_pad > R:
        row_padding = jnp.full((R_pad - R, C), -1, dtype=alignment.dtype)
        alignment = jnp.concatenate([alignment, row_padding], axis=0)

    # Then add columns
    if C_pad > C:
        col_padding = jnp.full((R_pad, C_pad - C), -1, dtype=alignment.dtype)
        alignment = jnp.concatenate([alignment, col_padding], axis=1)

    return padded_tree, alignment, R, C


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
