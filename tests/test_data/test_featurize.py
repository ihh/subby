"""End-to-end tests for feature extraction."""
import os
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.phylo.jax.types import Tree
from src.data.featurize import extract_features


def _make_small_tree():
    """5-node tree with 3 leaves."""
    parentIndex = jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32)
    distanceToParent = jnp.array([0.0, 0.1, 0.2, 0.15, 0.25])
    return Tree(parentIndex=parentIndex, distanceToParent=distanceToParent)


class TestFeaturize:

    def test_output_shape(self):
        """Feature matrix should be (F, C) with expected F."""
        tree = _make_small_tree()
        R = 5
        C = 8
        M = 15
        K = 4
        msa = jax.random.randint(jax.random.PRNGKey(0), (R, C), 0, 4).astype(jnp.int32)

        features = extract_features(msa, tree, M=M, K=K)
        expected_F = 16 + K + 128 + 12 + 3 * M  # = 16+4+128+12+45 = 205
        assert features.shape == (expected_F, C)

    def test_output_finite(self):
        """All features should be finite."""
        tree = _make_small_tree()
        R = 5
        C = 6
        msa = jax.random.randint(jax.random.PRNGKey(1), (R, C), 0, 4).astype(jnp.int32)

        features = extract_features(msa, tree)
        assert jnp.all(jnp.isfinite(features))

    def test_with_annotations(self):
        """Feature extraction should work with annotations."""
        tree = _make_small_tree()
        R = 5
        C = 4
        M = 3
        msa = jax.random.randint(jax.random.PRNGKey(2), (R, C), 0, 4).astype(jnp.int32)
        annot = jax.random.randint(jax.random.PRNGKey(3), (R, C), -1, M).astype(jnp.int32)

        features = extract_features(msa, tree, annotations=annot, M=M, K=2)
        expected_F = 16 + 2 + 128 + 12 + 3 * M  # = 16+2+128+12+9 = 167
        assert features.shape == (expected_F, C)
        assert jnp.all(jnp.isfinite(features))

    def test_with_gaps(self):
        """Should handle MSA with gaps."""
        tree = _make_small_tree()
        R = 5
        C = 4
        # Mix of nucs and gaps
        msa = jnp.array([
            [0, 5, 1, 2],
            [1, 2, 5, 3],
            [5, 0, 2, 5],
            [3, 1, 0, 2],
            [2, 5, 3, 1],
        ], dtype=jnp.int32)

        features = extract_features(msa, tree, K=2)
        assert jnp.all(jnp.isfinite(features))
