"""Tests for the Mamba tower, track encoder, cross-attention, and full model."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.model.jax.tower import (
    TrackEncoder,
    TrackCrossAttention,
    MambaTower,
    SubbyModel,
)
from src.model.jax.selectssm import BidirectionalMamba


# Use small dimensions for fast tests
B, C, D = 1, 32, 16
F = 20  # phylo feature dim
T = 2   # number of RNA-Seq tracks
NUM_LABELS = 15
HIDDEN = 4
EXPANSION = 2.0
D_TRACK = 8


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


class TestBidirectionalMamba:
    """Test the existing BidirectionalMamba layer."""

    def test_forward_shape(self, rng):
        model = BidirectionalMamba(
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
            norm_type='rms',
        )
        x = jnp.ones((B, C, D))
        params = model.init(rng, x)
        y = model.apply(params, x)
        assert y.shape == (B, C, D)

    def test_output_differs_from_input(self, rng):
        model = BidirectionalMamba(
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
            norm_type='rms',
        )
        x = jax.random.normal(rng, (B, C, D))
        params = model.init(rng, x)
        y = model.apply(params, x)
        # Due to residual connection, output should differ but not be identical
        assert not jnp.allclose(y, x)


class TestTrackEncoder:
    """Test the RNA-Seq track encoder."""

    def test_output_shape(self, rng):
        model = TrackEncoder(
            d_track=D_TRACK,
            n_layers=1,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
        )
        tracks = jnp.ones((B, T, 6, C))
        params = model.init(rng, tracks)
        out = model.apply(params, tracks)
        assert out.shape == (B, T, C, D_TRACK)

    def test_shared_weights(self, rng):
        """Weights should be shared across tracks."""
        model = TrackEncoder(
            d_track=D_TRACK,
            n_layers=1,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
        )
        # Two identical tracks should produce identical embeddings
        track = jax.random.normal(rng, (B, 1, 6, C))
        tracks = jnp.concatenate([track, track], axis=1)  # (B, 2, 6, C)
        params = model.init(rng, tracks)
        out = model.apply(params, tracks)
        np.testing.assert_allclose(
            np.array(out[:, 0, :, :]),
            np.array(out[:, 1, :, :]),
            atol=1e-5,
        )

    def test_different_tracks_different_output(self, rng):
        model = TrackEncoder(
            d_track=D_TRACK,
            n_layers=1,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
        )
        rng1, rng2 = jax.random.split(rng)
        track1 = jax.random.normal(rng1, (B, 1, 6, C))
        track2 = jax.random.normal(rng2, (B, 1, 6, C)) * 10
        tracks = jnp.concatenate([track1, track2], axis=1)
        params = model.init(rng, tracks)
        out = model.apply(params, tracks)
        assert not jnp.allclose(out[:, 0], out[:, 1])


class TestTrackCrossAttention:
    """Test the cross-attention mechanism."""

    def test_output_shape(self, rng):
        model = TrackCrossAttention(
            d_model=D,
            num_heads=2,
            d_track=D_TRACK,
        )
        x = jnp.ones((B, C, D))
        track_emb = jnp.ones((B, T, C, D_TRACK))
        params = model.init(rng, x, track_emb)
        out = model.apply(params, x, track_emb)
        assert out.shape == (B, C, D)

    def test_residual_connection(self, rng):
        """Output should differ from input due to cross-attention."""
        model = TrackCrossAttention(
            d_model=D,
            num_heads=2,
            d_track=D_TRACK,
        )
        x = jax.random.normal(rng, (B, C, D))
        track_emb = jax.random.normal(rng, (B, T, C, D_TRACK))
        params = model.init(rng, x, track_emb)
        out = model.apply(params, x, track_emb)
        # Should have residual: out ≈ x + attention_output
        assert not jnp.allclose(out, x)
        assert jnp.isfinite(out).all()


class TestMambaTower:
    """Test the Mamba tower with and without cross-attention."""

    def test_basic_tower(self, rng):
        model = MambaTower(
            d_model=D,
            n_layers=2,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
        )
        x = jnp.ones((B, C, D))
        params = model.init(rng, x)
        out = model.apply(params, x)
        assert out.shape == (B, C, D)

    def test_tower_with_cross_attention(self, rng):
        model = MambaTower(
            d_model=D,
            n_layers=3,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
            num_heads=2,
            d_track=D_TRACK,
            cross_attn_layers=(1,),
        )
        x = jnp.ones((B, C, D))
        track_emb = jnp.ones((B, T, C, D_TRACK))
        params = model.init(rng, x, track_emb)
        out = model.apply(params, x, track_emb)
        assert out.shape == (B, C, D)

    def test_no_tracks_skips_cross_attention(self, rng):
        """Cross-attention layers should be skipped when no tracks are provided."""
        model = MambaTower(
            d_model=D,
            n_layers=2,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
            cross_attn_layers=(0,),
        )
        x = jnp.ones((B, C, D))
        params = model.init(rng, x)
        out = model.apply(params, x, track_embeddings=None)
        assert out.shape == (B, C, D)


class TestSubbyModel:
    """Test the full SubbyModel."""

    def test_phylo_only(self, rng):
        """Model should work with phylo features only (no RNA-Seq)."""
        model = SubbyModel(
            d_model=D,
            num_labels=NUM_LABELS,
            n_layers=2,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
        )
        phylo = jnp.ones((B, F, C))
        params = model.init(rng, phylo)
        logits = model.apply(params, phylo)
        assert logits.shape == (B, C, NUM_LABELS)
        assert jnp.isfinite(logits).all()

    def test_with_rnaseq(self, rng):
        """Model should work with both phylo features and RNA-Seq tracks."""
        model = SubbyModel(
            d_model=D,
            num_labels=NUM_LABELS,
            n_layers=3,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
            num_heads=2,
            d_track=D_TRACK,
            track_n_layers=1,
            track_hidden_features=HIDDEN,
            cross_attn_layers=(1,),
        )
        phylo = jnp.ones((B, F, C))
        tracks = jnp.ones((B, T, 6, C))
        params = model.init(rng, phylo, tracks)
        logits = model.apply(params, phylo, tracks)
        assert logits.shape == (B, C, NUM_LABELS)
        assert jnp.isfinite(logits).all()

    def test_different_inputs_different_outputs(self, rng):
        model = SubbyModel(
            d_model=D,
            num_labels=NUM_LABELS,
            n_layers=2,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
        )
        rng1, rng2 = jax.random.split(rng)
        phylo1 = jax.random.normal(rng1, (B, F, C))
        phylo2 = jax.random.normal(rng2, (B, F, C))
        params = model.init(rng, phylo1)
        out1 = model.apply(params, phylo1)
        out2 = model.apply(params, phylo2)
        assert not jnp.allclose(out1, out2)

    def test_gradient_flows(self, rng):
        """Gradients should flow through the entire model."""
        model = SubbyModel(
            d_model=D,
            num_labels=NUM_LABELS,
            n_layers=2,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
        )
        phylo = jax.random.normal(rng, (B, F, C))
        params = model.init(rng, phylo)

        def loss_fn(params):
            logits = model.apply(params, phylo)
            return jnp.sum(logits ** 2)

        grads = jax.grad(loss_fn)(params)
        # Check that gradients exist for key parameters
        assert grads['params']['input_proj']['kernel'] is not None
        assert jnp.any(grads['params']['input_proj']['kernel'] != 0)
        assert grads['params']['output_head']['kernel'] is not None
        assert jnp.any(grads['params']['output_head']['kernel'] != 0)

    def test_gradient_with_tracks(self, rng):
        """Gradients should flow through track encoder and cross-attention."""
        model = SubbyModel(
            d_model=D,
            num_labels=NUM_LABELS,
            n_layers=2,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
            num_heads=2,
            d_track=D_TRACK,
            track_n_layers=1,
            track_hidden_features=HIDDEN,
            cross_attn_layers=(0,),
        )
        phylo = jax.random.normal(rng, (B, F, C))
        tracks = jax.random.normal(rng, (B, T, 6, C))
        params = model.init(rng, phylo, tracks)

        def loss_fn(params):
            logits = model.apply(params, phylo, tracks)
            return jnp.sum(logits ** 2)

        grads = jax.grad(loss_fn)(params)
        # Track encoder gradients should be nonzero
        track_grads = grads['params']['track_encoder']
        assert any(
            jnp.any(jax.tree.leaves(track_grads)[i] != 0)
            for i in range(len(jax.tree.leaves(track_grads)))
        )

    def test_param_count(self, rng):
        """Check that parameter count is reasonable."""
        model = SubbyModel(
            d_model=D,
            num_labels=NUM_LABELS,
            n_layers=2,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
        )
        phylo = jnp.ones((B, F, C))
        params = model.init(rng, phylo)
        n_params = sum(x.size for x in jax.tree.leaves(params))
        # Should be in the thousands for this small config
        assert n_params > 100
        assert n_params < 1_000_000

    def test_softmax_outputs_valid_distribution(self, rng):
        """Softmax of logits should sum to 1 per position."""
        model = SubbyModel(
            d_model=D,
            num_labels=NUM_LABELS,
            n_layers=2,
            hidden_features=HIDDEN,
            expansion_factor=EXPANSION,
        )
        phylo = jax.random.normal(rng, (B, F, C))
        params = model.init(rng, phylo)
        logits = model.apply(params, phylo)
        probs = jax.nn.softmax(logits, axis=-1)
        sums = jnp.sum(probs, axis=-1)
        np.testing.assert_allclose(np.array(sums), 1.0, atol=1e-5)
