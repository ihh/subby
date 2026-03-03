"""Tests for ungapped component detection and branch masking."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax.numpy as jnp
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.phylo.jax.components import compute_branch_mask


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
