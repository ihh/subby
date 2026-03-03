# Model API Reference

The model module implements the subby gene annotation architecture in JAX (flax.linen):

1. **TrackEncoder** — bidirectional Mamba on RNA-Seq tensors (shared weights across tracks)
2. **TrackCrossAttention** — per-position attention over track embeddings
3. **MambaTower** — stack of bidirectional Mamba layers with optional cross-attention
4. **SubbyModel** — full end-to-end model from phylo features to annotation logits

## SubbyModel

The top-level model class.

```python
from src.model.jax import SubbyModel

model = SubbyModel(
    d_model=256,
    num_labels=15,
    n_layers=8,
    hidden_features=16,
    expansion_factor=2.0,
    num_heads=4,
    d_track=64,
    cross_attn_layers=(2, 5),
    track_n_layers=2,
)

# Initialize
params = model.init(rng, phylo_features)
# or with RNA-Seq:
params = model.init(rng, phylo_features, rnaseq_tracks)

# Forward pass
logits = model.apply(params, phylo_features)
# or with RNA-Seq:
logits = model.apply(params, phylo_features, rnaseq_tracks)
```

### Input/Output

| Tensor | Shape | Description |
|--------|-------|-------------|
| `phylo_features` | `(B, F, C)` | Phylogenetic feature vectors |
| `rnaseq_tracks` | `(B, T, 6, C)` or `None` | RNA-Seq tensors (optional) |
| **output** (logits) | `(B, C, num_labels)` | Per-position annotation logits |

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | 256 | Hidden dimension of the tower |
| `num_labels` | int | 15 | Number of annotation labels (Tiberius: 15) |
| `n_layers` | int | 8 | Number of bidirectional Mamba layers |
| `hidden_features` | int | 16 | SSM hidden state dimension ($N$) |
| `expansion_factor` | float | 2.0 | Mamba internal expansion factor ($E$) |
| `num_heads` | int | 4 | Number of cross-attention heads |
| `d_track` | int | 64 | Track encoder output dimension |
| `cross_attn_layers` | tuple[int] | `()` | Tower layers with cross-attention |
| `track_n_layers` | int | 2 | Number of BiMamba layers in track encoder |
| `track_hidden_features` | int | 16 | Track encoder SSM hidden dimension |
| `track_expansion_factor` | float | 2.0 | Track encoder expansion factor |

### Pipeline

```
phylo_features (B, F, C)
    → transpose to (B, C, F)
    → Dense(d_model)             → (B, C, D)
    → MambaTower                 → (B, C, D)
    → RMSNorm
    → Dense(num_labels)          → (B, C, num_labels)

rnaseq_tracks (B, T, 6, C)      [optional]
    → TrackEncoder               → (B, T, C, d_track)
    → fed into cross-attention at selected tower layers
```

---

## MambaTower

Stack of `BidirectionalMamba` layers with optional `TrackCrossAttention` at specified layer indices.

```python
tower = MambaTower(
    d_model=256, n_layers=8,
    cross_attn_layers=(2, 5),  # cross-attention after layers 2 and 5
)
```

When `track_embeddings=None`, cross-attention layers are skipped.

---

## TrackEncoder

Bidirectional Mamba encoder for RNA-Seq tracks. The same weights are shared across all $T$ tracks (implemented by reshaping to `(B*T, C, 6)` and running a single encoder).

```python
encoder = TrackEncoder(d_track=64, n_layers=2)
track_emb = encoder(tracks)  # (B, T, 6, C) → (B, T, C, 64)
```

---

## TrackCrossAttention

Multi-head cross-attention where each genomic position queries over track embeddings at that position.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{\text{head}}}}\right)V$$

where $Q \in \mathbb{R}^{H \times d}$ comes from the main tower and $K, V \in \mathbb{R}^{T \times H \times d}$ come from track embeddings. The result is added to the residual stream.

---

## BidirectionalMamba

Pre-existing bidirectional selective state space model layer (Mamba). See `src/model/jax/selectssm.py`.

Each layer:
1. **RMSNorm** pre-normalization
2. **Input projection** to expanded dimension ($E \cdot D$)
3. **Forward SSM** (causal convolution → selective scan)
4. **Reverse SSM** (time-reversed, shared or separate weights)
5. **Gate**: concatenate `[fwd * σ(gate_fwd), rev * σ(gate_rev)]`
6. **Output projection** back to $D$
7. **Residual connection**: `output = skip + projected`

Supports optional MLP sub-layer after the SSM.

### SelectiveSSM Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_features` | int | 16 | SSM state dimension $N$ |
| `dt_rank` | int or `'auto'` | `'auto'` | $\Delta t$ rank ($\lceil D/16 \rceil$ if auto) |
| `dt_min` | float | 0.001 | Minimum $\Delta t$ (long-range context) |
| `dt_max` | float | 0.1 | Maximum $\Delta t$ (short-range context) |
| `shift_conv_size` | int | 3 | Causal convolution kernel size |
| `activation` | str | `'silu'` | Activation function |
| `recursive_scan` | bool | `False` | Memory-efficient recursive scan |
