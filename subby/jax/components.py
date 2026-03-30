from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_branch_mask(
    alignment: jnp.ndarray,
    parentIndex: jnp.ndarray,
    A: int,
) -> jnp.ndarray:
    """Identify active branches per column (minimum Steiner tree of ungapped leaves).

    A branch parent->child is active if both endpoints are in the Steiner tree
    connecting ungapped leaves.

    This implementation replaces sequential jax.lax.scan passes (which carry
    full (R,C) arrays through R steps) with iterative vectorized scatter
    operations that converge in O(depth) iterations — typically 6-7 for a
    balanced binary tree of ~63 nodes.

    Args:
        alignment: (R, C) int32 tokens
        parentIndex: (R,) int32 parent indices
        A: alphabet size

    Returns:
        branch_mask: (R, C) bool — True if branch from parent(n)->n is active.
                     branch_mask[0] is always False (root has no parent branch).
    """
    R, C = alignment.shape

    # Fix parentIndex[0] = -1 to avoid negative indexing issues
    # (root's parent is never used in the algorithm, but -1 wraps to last element)
    parentIndex = parentIndex.at[0].set(0)

    # Step 1: Mark leaves as ungapped
    is_ungapped_leaf = (alignment >= 0) & (alignment <= A)  # (R, C)

    # Determine which nodes are leaves
    child_count = jnp.zeros(R, dtype=jnp.int32)
    child_count = child_count.at[parentIndex[1:]].add(1)
    is_leaf = (child_count == 0)  # (R,)

    has_ungapped = jnp.where(
        is_leaf[:, None],
        is_ungapped_leaf,
        jnp.zeros((R, C), dtype=bool),
    )

    # Step 2: Upward pass — propagate "has ungapped descendant"
    # Iterative scatter-OR: each iteration propagates one tree level upward.
    # Python for-loop with static trip count for JIT compatibility.
    n_iters = _max_depth_bound(R)
    for _ in range(n_iters):
        parent_update = jnp.zeros((R, C), dtype=bool)
        parent_update = parent_update.at[parentIndex[1:]].max(has_ungapped[1:])
        has_ungapped = has_ungapped | parent_update

    # Step 3: Count children with ungapped descendants
    ungapped_child_count = jnp.zeros((R, C), dtype=jnp.int32)
    ungapped_child_count = ungapped_child_count.at[parentIndex[1:]].add(
        has_ungapped[1:].astype(jnp.int32)
    )

    is_ungapped_leaf_node = is_leaf[:, None] & is_ungapped_leaf
    is_steiner = is_ungapped_leaf_node | (ungapped_child_count >= 2)

    # Step 4: Top-down pass — propagate Steiner membership
    for _ in range(n_iters):
        parent_steiner = is_steiner[parentIndex]
        is_steiner = is_steiner | (parent_steiner & has_ungapped)

    # Branch parent(n)->n is active if both endpoints are Steiner
    parent_is_steiner = is_steiner[parentIndex]
    branch_mask = is_steiner & parent_is_steiner
    branch_mask = branch_mask.at[0].set(False)

    return branch_mask


def _max_depth_bound(R: int) -> int:
    """Upper bound on tree depth: ceil(log2(R)) + 1, always >= 1."""
    if R <= 1:
        return 1
    d = 0
    v = R - 1
    while v > 0:
        v >>= 1
        d += 1
    return d + 1
