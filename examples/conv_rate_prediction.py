"""Per-column rate prediction with a 1D convolutional network.

Proof-of-concept: a small 1D CNN reads the one-hot-encoded leaf sequences
of an MSA and predicts per-column rate multipliers for a Jukes-Cantor
model.  The network is trained to maximize the phylogenetic likelihood
of the full alignment under those per-column rates.

This demonstrates that gradients flow from LogLike through the per-column
model API back into a neural network's parameters.

Usage (CPU is fine for this toy example):
    source ~/jax-env/bin/activate
    python examples/conv_rate_prediction.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

from subby.jax import LogLike
from subby.jax.types import Tree
from subby.jax.models import jukes_cantor_model, scale_model


# ---------------------------------------------------------------------------
# Tiny conv network: one-hot leaf sequences -> per-column rates
# ---------------------------------------------------------------------------

def init_params(key, n_input_channels, hidden=32, kernel_size=5):
    """Initialise CNN parameters."""
    k1, k2, k3 = jr.split(key, 3)
    return {
        "conv1_w": jr.normal(k1, (hidden, n_input_channels, kernel_size)) * 0.05,
        "conv1_b": jnp.zeros(hidden),
        "conv2_w": jr.normal(k2, (hidden, hidden, kernel_size)) * 0.05,
        "conv2_b": jnp.zeros(hidden),
        "conv3_w": jr.normal(k3, (1, hidden, 1)) * 0.05,
        "conv3_b": jnp.zeros(1),
    }


def conv1d(x, w, b):
    """1D convolution: x (channels, length), w (out, in, kernel), b (out,)."""
    out = jax.lax.conv_general_dilated(
        x[None, ...], w, window_strides=(1,), padding="SAME",
    )[0]
    return out + b[:, None]


def predict_rates(params, leaf_one_hot):
    """Forward pass: one-hot leaves (n_leaves * A, C) -> positive rates (C,)."""
    h = jax.nn.relu(conv1d(leaf_one_hot, params["conv1_w"], params["conv1_b"]))
    h = jax.nn.relu(conv1d(h, params["conv2_w"], params["conv2_b"]))
    h = conv1d(h, params["conv3_w"], params["conv3_b"])  # (1, C)
    return jax.nn.softplus(h[0])  # (C,) positive


# ---------------------------------------------------------------------------
# Loss: negative log-likelihood under per-column JC rates
# ---------------------------------------------------------------------------

def make_per_column_models(base_model, rates):
    """Create a list of C scaled JC models from per-column rate predictions."""
    return [scale_model(base_model, rates[c]) for c in range(rates.shape[0])]


def loss_fn(params, alignment, tree, base_model, leaf_indices):
    """Negative total log-likelihood with CNN-predicted per-column rates."""
    A = base_model.pi.shape[0]
    # One-hot encode leaf sequences and stack: (n_leaves, C, A) -> (n_leaves * A, C)
    leaves = alignment[leaf_indices]  # (n_leaves, C)
    one_hot = jax.nn.one_hot(leaves, A)  # (n_leaves, C, A)
    one_hot_flat = one_hot.reshape(-1, one_hot.shape[-1]).T  # (n_leaves * A, C)
    # Transpose so channels come first
    one_hot_input = one_hot.transpose(0, 2, 1).reshape(-1, alignment.shape[1])  # (n_leaves*A, C)
    rates = predict_rates(params, one_hot_input)
    models = make_per_column_models(base_model, rates)
    ll = LogLike(alignment, tree, models)
    return -jnp.sum(ll)


# ---------------------------------------------------------------------------
# Simulate evolution along a tree
# ---------------------------------------------------------------------------

def simulate_alignment(rng, parent_idx, distances, rates, A=4):
    """Simulate sequences along a tree under JC with per-column rates."""
    R = len(parent_idx)
    C = len(rates)
    mu = A / (A - 1.0)

    seqs = np.zeros((R, C), dtype=np.int32)
    seqs[0] = rng.integers(0, A, size=C)

    for n in range(1, R):
        p = parent_idx[n]
        t = float(distances[n])
        p_change = (1.0 - np.exp(-mu * np.array(rates) * t)) * (A - 1) / A
        mutate = rng.random(C) < p_change
        offsets = rng.integers(1, A, size=C)
        seqs[n] = np.where(mutate, (seqs[p] + offsets) % A, seqs[p])

    # Identify leaves
    has_child = np.zeros(R, dtype=bool)
    for n in range(1, R):
        has_child[parent_idx[n]] = True
    is_leaf = ~has_child

    alignment = np.where(is_leaf[:, None], seqs, A)
    return alignment.astype(np.int32), np.where(is_leaf)[0]


def make_toy_data(seed=0, C=40, A=4):
    """Generate a 15-node tree + simulated alignment with varying true rates."""
    rng = np.random.default_rng(seed)
    # 15-node balanced binary tree (8 leaves)
    parent_idx = np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.int32)
    distances = np.array([0.0, 0.15, 0.20, 0.10, 0.15, 0.12, 0.18,
                          0.25, 0.30, 0.20, 0.35, 0.22, 0.40, 0.28, 0.45])
    tree = Tree(
        parentIndex=jnp.array(parent_idx, dtype=jnp.int32),
        distanceToParent=jnp.array(distances),
    )

    # Ground-truth per-column rates: first half slow, second half fast
    true_rates = np.concatenate([
        np.full(C // 2, 0.2),
        np.full(C - C // 2, 4.0),
    ])

    alignment, leaf_indices = simulate_alignment(rng, parent_idx, distances, true_rates, A)
    return tree, jnp.array(alignment), jnp.array(true_rates), jnp.array(leaf_indices)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(n_steps=500, lr=1e-3, seed=42):
    """Train the conv net to predict per-column evolutionary rates."""
    key = jr.PRNGKey(seed)
    A = 4
    tree, alignment, true_rates, leaf_indices = make_toy_data(seed=0, C=40, A=A)
    n_leaves = len(leaf_indices)
    base_model = jukes_cantor_model(A)
    params = init_params(key, n_input_channels=n_leaves * A, hidden=32, kernel_size=5)

    # Baselines
    ll_uniform = jnp.sum(LogLike(alignment, tree, base_model))
    oracle_models = make_per_column_models(base_model, true_rates)
    ll_oracle = jnp.sum(LogLike(alignment, tree, oracle_models))
    print(f"Baseline logL (uniform rate=1):  {ll_uniform:.4f}")
    print(f"Oracle logL (true rates):        {ll_oracle:.4f}")

    grad_fn = jax.jit(jax.grad(loss_fn))

    print(f"\nTraining for {n_steps} steps (lr={lr})...")
    for step in range(n_steps):
        grads = grad_fn(params, alignment, tree, base_model, leaf_indices)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

        if step % 100 == 0 or step == n_steps - 1:
            nll = loss_fn(params, alignment, tree, base_model, leaf_indices)
            leaves = alignment[leaf_indices]
            one_hot = jax.nn.one_hot(leaves, A)
            one_hot_input = one_hot.transpose(0, 2, 1).reshape(-1, alignment.shape[1])
            rates = predict_rates(params, one_hot_input)
            slow = rates[:20].mean()
            fast = rates[20:].mean()
            print(f"  step {step:4d}  logL={-nll:.4f}  "
                  f"slow_mean={slow:.3f}  fast_mean={fast:.3f}")

    # Final summary
    leaves = alignment[leaf_indices]
    one_hot = jax.nn.one_hot(leaves, A)
    one_hot_input = one_hot.transpose(0, 2, 1).reshape(-1, alignment.shape[1])
    final_rates = predict_rates(params, one_hot_input)
    final_ll = -loss_fn(params, alignment, tree, base_model, leaf_indices)
    print(f"\nTrue rates:      slow={true_rates[0]:.1f}  fast={true_rates[-1]:.1f}")
    print(f"Predicted rates: slow={final_rates[:20].mean():.3f}  fast={final_rates[20:].mean():.3f}")
    print(f"Final logL:  {final_ll:.4f}  (oracle: {ll_oracle:.4f})")

    return params, final_rates


if __name__ == "__main__":
    train()
