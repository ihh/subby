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

    Args:
        alignment: (R, C) int32 tokens
        parentIndex: (R,) int32 parent indices
        A: alphabet size

    Returns:
        branch_mask: (R, C) bool — True if branch from parent(n)->n is active.
                     branch_mask[0] is always False (root has no parent branch).
    """
    R, C = alignment.shape

    # Step 1: Mark leaves as ungapped if token in {0..A-1, A} (observed or ungapped-unobserved)
    # Gapped tokens: A+1 or -1
    is_ungapped_leaf = (alignment >= 0) & (alignment <= A)  # (R, C)

    # Determine which nodes are leaves: nodes with no children
    child_count = jnp.zeros(R, dtype=jnp.int32)
    child_count = child_count.at[parentIndex[1:]].add(1)
    is_leaf = (child_count == 0)  # (R,)

    # Step 2: Upward pass — mark internal nodes as "has ungapped descendant"
    # ungapped[n] = True if n is ungapped leaf, or any child has ungapped descendant
    has_ungapped = jnp.where(
        is_leaf[:, None],
        is_ungapped_leaf,
        jnp.zeros((R, C), dtype=bool),
    )

    # Scan in postorder (R-1 down to 1) to propagate ungapped status up
    def _propagate_step(has_ungapped, n):
        parent = parentIndex[n]
        # If child n has ungapped descendants, mark parent
        has_ungapped = has_ungapped.at[parent].set(
            has_ungapped[parent] | has_ungapped[n]
        )
        return has_ungapped, None

    postorder = jnp.arange(R - 1, 0, -1, dtype=jnp.int32)
    has_ungapped, _ = jax.lax.scan(_propagate_step, has_ungapped, postorder)

    # Step 3: Identify Steiner tree nodes
    # Node is in Steiner tree if:
    # (a) it's an ungapped leaf, OR
    # (b) it has >= 2 children with ungapped descendants (branching point), OR
    # (c) it has ungapped descendants AND its parent is in the Steiner tree (pass-through)

    # Count how many children have ungapped descendants
    ungapped_child_count = jnp.zeros((R, C), dtype=jnp.int32)

    def _count_step(ungapped_child_count, n):
        parent = parentIndex[n]
        ungapped_child_count = ungapped_child_count.at[parent].add(
            has_ungapped[n].astype(jnp.int32)
        )
        return ungapped_child_count, None

    ungapped_child_count, _ = jax.lax.scan(_count_step, ungapped_child_count, postorder)

    is_ungapped_leaf_node = is_leaf[:, None] & is_ungapped_leaf
    # Core Steiner nodes: ungapped leaves and branching points
    is_steiner = is_ungapped_leaf_node | (ungapped_child_count >= 2)

    # Top-down pass: propagate Steiner membership to pass-through nodes
    # A node with ungapped descendants whose parent is already in the Steiner tree
    # is also in the Steiner tree (it's on the path)
    preorder = jnp.arange(1, R, dtype=jnp.int32)

    def _propagate_steiner(is_steiner, n):
        parent = parentIndex[n]
        # If parent is Steiner and this node has ungapped descendants, it's on the path
        is_steiner = is_steiner.at[n].set(
            is_steiner[n] | (is_steiner[parent] & has_ungapped[n])
        )
        return is_steiner, None

    is_steiner, _ = jax.lax.scan(_propagate_steiner, is_steiner, preorder)

    # Branch parent(n)->n is active if both n and parent(n) are in the Steiner tree
    parent_is_steiner = is_steiner[parentIndex]  # (R, C)
    branch_mask = is_steiner & parent_is_steiner  # (R, C)
    branch_mask = branch_mask.at[0].set(False)  # root has no parent branch

    return branch_mask
