"""Mamba tower with RNA-Seq track encoder and cross-attention.

Architecture (per Model.md):
  - Input: phylo features (F, C) → input projection → (D, C)
  - Tower: N bidirectional Mamba layers with RmsNorm pre-normalization
  - Track encoder: bidirectional Mamba on 6-channel RNA-Seq tensors, shared weights
  - Cross-attention: at selected layers, each genomic position queries over
    per-track embeddings (one key-value pair per track), weighted pool → residual
  - Output: linear head → (num_labels, C) logits
"""

import math
from typing import Sequence, Optional
from dataclasses import field

import flax.linen as nn
import jax
import jax.numpy as jnp

from .selectssm import BidirectionalMamba


class TrackEncoder(nn.Module):
    """Encode RNA-Seq tracks with a shared bidirectional Mamba.

    Input: (B, T, 6, C) — T tracks × 6 channels × C positions
    Output: (B, T, C, D_track) — per-position per-track embeddings

    The same Mamba weights are shared across all T tracks.
    """
    d_track: int = 64
    n_layers: int = 2
    hidden_features: int = 16
    expansion_factor: float = 2.0
    ssm_args: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, tracks, train: bool = False):
        """
        Args:
            tracks: (B, T, 6, C) float — RNA-Seq tensors
            train: training mode flag

        Returns:
            (B, T, C, d_track) float — per-position per-track embeddings
        """
        B, T, channels, C = tracks.shape

        # Reshape to (B*T, C, 6) for the shared encoder
        x = jnp.transpose(tracks, (0, 1, 3, 2))  # (B, T, C, 6)
        x = x.reshape(B * T, C, channels)  # (B*T, C, 6)

        # Project to d_track
        x = nn.Dense(self.d_track, name='track_in_proj',
                      kernel_init=nn.initializers.lecun_normal())(x)  # (B*T, C, d_track)

        # Stack of bidirectional Mamba layers
        for i in range(self.n_layers):
            x = BidirectionalMamba(
                hidden_features=self.hidden_features,
                expansion_factor=self.expansion_factor,
                norm_type='rms',
                ssm_args=self.ssm_args,
                name=f'track_mamba_{i}',
            )(x, train=train)

        # Reshape back to (B, T, C, d_track)
        x = x.reshape(B, T, C, self.d_track)
        return x


class TrackCrossAttention(nn.Module):
    """Cross-attention: genomic positions attend over RNA-Seq track embeddings.

    For each position c, the query is from the main tower embedding,
    keys and values are the T track embeddings at that position.
    Output is a weighted pool added to the residual stream.
    """
    d_model: int = 256
    num_heads: int = 4
    d_track: int = 64

    @nn.compact
    def __call__(self, x, track_embeddings, train: bool = False):
        """
        Args:
            x: (B, C, D) main tower hidden states
            track_embeddings: (B, T, C, d_track) from TrackEncoder
            train: training mode flag

        Returns:
            (B, C, D) updated hidden states (residual connection applied)
        """
        B, C, D = x.shape
        T = track_embeddings.shape[1]
        d_head = D // self.num_heads

        residual = x

        # Pre-norm
        x = nn.RMSNorm(name='cross_attn_norm')(x)

        # Queries from main tower: (B, C, num_heads, d_head)
        Q = nn.Dense(D, name='query',
                     kernel_init=nn.initializers.lecun_normal())(x)
        Q = Q.reshape(B, C, self.num_heads, d_head)

        # Keys and values from track embeddings: (B, T, C, d_track) → (B, C, T, d_head)
        # Project track embeddings to match head dimension
        track_flat = track_embeddings.transpose(0, 2, 1, 3)  # (B, C, T, d_track)
        K = nn.Dense(D, name='key',
                     kernel_init=nn.initializers.lecun_normal())(track_flat)
        K = K.reshape(B, C, T, self.num_heads, d_head)

        V = nn.Dense(D, name='value',
                     kernel_init=nn.initializers.lecun_normal())(track_flat)
        V = V.reshape(B, C, T, self.num_heads, d_head)

        # Attention: Q (B, C, H, d) @ K^T (B, C, T, H, d) → (B, C, H, T)
        # For each position c, attend over T tracks
        scale = 1.0 / math.sqrt(d_head)
        attn_logits = jnp.einsum('bchd,bcthd->bcht', Q, K) * scale  # (B, C, H, T)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # (B, C, H, T)

        # Weighted sum of values: (B, C, H, T) @ (B, C, T, H, d) → (B, C, H, d)
        attn_out = jnp.einsum('bcht,bcthd->bchd', attn_weights, V)  # (B, C, H, d)
        attn_out = attn_out.reshape(B, C, D)

        # Output projection
        attn_out = nn.Dense(D, name='cross_attn_out',
                           kernel_init=nn.initializers.lecun_normal())(attn_out)

        return residual + attn_out


class MambaTower(nn.Module):
    """Tower of bidirectional Mamba layers with optional cross-attention.

    Cross-attention layers are interleaved at positions specified by
    cross_attn_layers (e.g. every 3rd layer: [2, 5, 8, ...]).
    """
    d_model: int = 256
    n_layers: int = 8
    hidden_features: int = 16
    expansion_factor: float = 2.0
    num_heads: int = 4
    d_track: int = 64
    cross_attn_layers: Sequence[int] = ()
    ssm_args: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, track_embeddings=None, train: bool = False):
        """
        Args:
            x: (B, C, D) input embeddings
            track_embeddings: (B, T, C, d_track) or None
            train: training mode flag

        Returns:
            (B, C, D) output embeddings
        """
        cross_attn_set = set(self.cross_attn_layers)

        for i in range(self.n_layers):
            x = BidirectionalMamba(
                hidden_features=self.hidden_features,
                expansion_factor=self.expansion_factor,
                norm_type='rms',
                ssm_args=self.ssm_args,
                name=f'mamba_{i}',
            )(x, train=train)

            if i in cross_attn_set and track_embeddings is not None:
                x = TrackCrossAttention(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_track=self.d_track,
                    name=f'cross_attn_{i}',
                )(x, track_embeddings, train=train)

        return x


class SubbyModel(nn.Module):
    """Full subby model: phylo features → Mamba tower → annotation logits.

    Input: phylo features (B, F, C) and optional RNA-Seq tracks (B, T, 6, C).
    Output: (B, C, num_labels) logits over annotation labels.
    """
    # Model dimensions
    d_model: int = 256
    num_labels: int = 15

    # Tower config
    n_layers: int = 8
    hidden_features: int = 16
    expansion_factor: float = 2.0
    num_heads: int = 4
    cross_attn_layers: Sequence[int] = ()

    # Track encoder config
    d_track: int = 64
    track_n_layers: int = 2
    track_hidden_features: int = 16
    track_expansion_factor: float = 2.0

    ssm_args: dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, phylo_features, rnaseq_tracks=None, train: bool = False):
        """
        Args:
            phylo_features: (B, F, C) float — phylogenetic feature vectors
            rnaseq_tracks: (B, T, 6, C) float or None — RNA-Seq tensors
            train: training mode flag

        Returns:
            (B, C, num_labels) float — logits over annotation labels
        """
        B, F, C = phylo_features.shape

        # Transpose to (B, C, F) for dense layers
        x = jnp.transpose(phylo_features, (0, 2, 1))  # (B, C, F)

        # Input projection
        x = nn.Dense(self.d_model, name='input_proj',
                     kernel_init=nn.initializers.lecun_normal())(x)  # (B, C, D)

        # Track encoder (if RNA-Seq is provided)
        track_embeddings = None
        if rnaseq_tracks is not None:
            track_embeddings = TrackEncoder(
                d_track=self.d_track,
                n_layers=self.track_n_layers,
                hidden_features=self.track_hidden_features,
                expansion_factor=self.track_expansion_factor,
                ssm_args=self.ssm_args,
                name='track_encoder',
            )(rnaseq_tracks, train=train)

        # Mamba tower
        x = MambaTower(
            d_model=self.d_model,
            n_layers=self.n_layers,
            hidden_features=self.hidden_features,
            expansion_factor=self.expansion_factor,
            num_heads=self.num_heads,
            d_track=self.d_track,
            cross_attn_layers=self.cross_attn_layers,
            ssm_args=self.ssm_args,
            name='tower',
        )(x, track_embeddings=track_embeddings, train=train)

        # Final norm
        x = nn.RMSNorm(name='final_norm')(x)

        # Output head
        logits = nn.Dense(self.num_labels, name='output_head',
                         kernel_init=nn.initializers.lecun_normal())(x)  # (B, C, num_labels)

        return logits
