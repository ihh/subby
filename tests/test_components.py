"""Tests for ungapped component detection and branch masking."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax.numpy as jnp
import numpy as np
import pytest


from subby.jax.components import compute_branch_mask


class TestBranchMask:

    def test_all_observed_all_active(self):
        """If all leaves are observed, all branches should be active."""
        #     0
        #    / \
        #   1   2
        parentIndex = jnp.array([-1, 0, 0], dtype=jnp.int32)
        A = 4
        alignment = jnp.array([[0, 1], [2, 3], [1, 0]], dtype=jnp.int32)
        mask = compute_branch_mask(alignment, parentIndex, A)
        # Branches 1 and 2 (root's children) should be active
        assert mask[0, 0] == False  # root has no parent branch
        assert mask[1, 0] == True
        assert mask[2, 0] == True

    def test_all_gapped_no_active(self):
        """If all leaves are gapped, no branches should be active."""
        parentIndex = jnp.array([-1, 0, 0], dtype=jnp.int32)
        A = 4
        alignment = jnp.array([[-1], [-1], [-1]], dtype=jnp.int32)
        mask = compute_branch_mask(alignment, parentIndex, A)
        assert jnp.all(~mask)

    def test_single_observed_no_branches(self):
        """If only one leaf is observed, the Steiner tree has no branches."""
        #     0
        #    / \
        #   1   2
        #  / \
        # 3   4
        parentIndex = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
        A = 4
        # Only leaf 3 observed
        alignment = jnp.array([[-1], [-1], [-1], [0], [-1]], dtype=jnp.int32)
        mask = compute_branch_mask(alignment, parentIndex, A)
        # No branches active (need >=2 ungapped leaves for a tree)
        # Actually, a single leaf still has its branch to parent...
        # But the Steiner tree of a single node has no edges.
        # The parent (node 1) has only 1 ungapped child, so it's not a Steiner node.
        # Therefore no branch has both endpoints in the Steiner tree.
        assert jnp.sum(mask) == 0

    def test_two_leaves_two_branches(self):
        """Two observed leaves on opposite sides of root."""
        #     0
        #    / \
        #   1   2
        #  / \
        # 3   4
        parentIndex = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
        A = 4
        # Leaves 3 and 2 observed
        alignment = jnp.array([[-1], [-1], [1], [0], [-1]], dtype=jnp.int32)
        mask = compute_branch_mask(alignment, parentIndex, A)
        # Steiner tree connects leaves 3 and 2 through 1 and 0.
        # Active branches: 3->1, 1->0, 2->0
        assert mask[3, 0] == True
        assert mask[1, 0] == True
        assert mask[2, 0] == True
        assert mask[4, 0] == False  # leaf 4 is gapped

    def test_root_never_active(self):
        """Root branch (node 0) should never be active."""
        parentIndex = jnp.array([-1, 0, 0], dtype=jnp.int32)
        alignment = jnp.array([[0], [1], [2]], dtype=jnp.int32)
        mask = compute_branch_mask(alignment, parentIndex, 4)
        assert mask[0, 0] == False

    def test_larger_tree(self):
        """Test on a 31-node balanced binary tree with gaps."""
        R = 31
        parentIndex = np.zeros(R, dtype=np.int32)
        parentIndex[0] = -1
        for i in range(1, R):
            parentIndex[i] = (i - 1) // 2
        parentIndex = jnp.array(parentIndex)

        A = 4
        C = 16
        # All observed: all branches should be active except root
        alignment = jnp.zeros((R, C), dtype=jnp.int32)
        mask = compute_branch_mask(alignment, parentIndex, A)
        assert mask[0, 0] == False
        assert jnp.all(mask[1:, :])

    def test_multi_column_gapping(self):
        """Test that different columns can have different active branches."""
        #     0
        #    / \
        #   1   2
        #  / \
        # 3   4
        parentIndex = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
        A = 4

        # Column 0: only leaves 3,4 observed (subtree of node 1 only)
        # Column 1: all leaves observed
        alignment = jnp.array([
            [-1, 0],   # root
            [-1, 1],   # node 1 (internal)
            [-1, 2],   # node 2 (leaf)
            [0, 3],    # node 3 (leaf)
            [1, 0],    # node 4 (leaf)
        ], dtype=jnp.int32)

        mask = compute_branch_mask(alignment, parentIndex, A)

        # Column 0: only 3,4 observed -> Steiner tree has nodes {1,3,4}
        # Branch 3->1 active, 4->1 active, 1->0 NOT active (0 not Steiner)
        assert mask[3, 0] == True
        assert mask[4, 0] == True
        assert mask[1, 0] == False  # parent (0) not in Steiner tree
        assert mask[2, 0] == False  # gapped

        # Column 1: all observed -> all branches active
        assert jnp.all(mask[1:, 1])

    def test_deep_caterpillar_tree(self):
        """Test on a caterpillar tree (depth = R-1)."""
        # 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6
        # Each internal node has one child that is the next internal node
        # and one leaf child
        #     0
        #    / \
        #   1   7
        #  / \
        # 2   8
        # / \
        # 3  9
        # etc.
        R = 13  # 6 internal + 7 leaves (last internal has 2 leaf children)
        parentIndex = jnp.array([-1, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=jnp.int32)
        A = 4

        # All leaves observed
        alignment = jnp.zeros((R, 1), dtype=jnp.int32)
        # Internal nodes are not leaves, they get -1
        alignment = alignment.at[:6, :].set(-1)

        mask = compute_branch_mask(alignment, parentIndex, A)
        # All branches should be active since all leaves are observed
        assert jnp.all(mask[1:, :])
