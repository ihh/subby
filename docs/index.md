# subby Phylogenetic Sufficient Statistics Library

Phylogenetic sufficient statistics via the Holmes & Rubin (2002) eigensubstitution-accumulation algorithm, with implementations in JAX, Python, WebGPU, and Rust/WASM.

## What is this?

Given a multiple sequence alignment (MSA) on a phylogenetic tree, this library computes **per-column expected substitution counts** and **dwell times** — the sufficient statistics for continuous-time Markov models of sequence evolution. These statistics are used as input features for gene annotation models.

The core algorithm is the **eigensubstitution accumulation** method of [Holmes & Rubin (2002)](https://doi.org/10.1089/10665270252935467), which computes posterior expectations of substitution events along every branch of the tree using:

1. **Felsenstein pruning** (upward/inside pass) — leaf-to-root likelihood propagation
2. **Outside algorithm** (downward pass) — root-to-leaf posterior propagation
3. **Eigensubstitution accumulation** — posterior expected counts in the eigenbasis of the rate matrix
4. **Back-transformation** — conversion to natural-basis substitution counts and dwell times

## Implementations

| Backend | Language | Precision | Use case |
|---------|----------|-----------|----------|
| [JAX](api/jax.md) | Python (JAX) | f64 | Training, GPU-accelerated batch computation |
| [Oracle](api/oracle.md) | Python (NumPy) | f64 | Test oracle, reference implementation |
| [WebGPU](api/webgpu.md) | WGSL + JavaScript | f32 | In-browser inference (primary) |
| [WASM](api/wasm.md) | Rust → WebAssembly | f64 | In-browser fallback, native CLI |

## Quick start

```python
import numpy as np
from subby.oracle import LogLike, Counts, RootProb, jukes_cantor_model

# Define a 5-node binary tree (preorder parent indices)
tree = {
    'parentIndex': np.array([-1, 0, 0, 1, 1], dtype=np.int32),
    'distanceToParent': np.array([0.0, 0.1, 0.2, 0.15, 0.25]),
}

# Alignment: 5 sequences × 4 columns, nucleotides {A=0, C=1, G=2, T=3}
alignment = np.array([
    [0, 1, 2, 3],
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [0, 1, 2, 3],
    [0, 0, 2, 3],
], dtype=np.int32)

# Jukes-Cantor model (4-state, uniform equilibrium)
model = jukes_cantor_model(4)

# Compute
log_likelihoods = LogLike(alignment, tree, model)        # (4,)
counts = Counts(alignment, tree, model)                   # (4, 4, 4)
root_posterior = RootProb(alignment, tree, model)          # (4, 4)
```

## Documentation

- **[Mathematical Background](background.md)** — the eigensubstitution accumulation algorithm
- **[Architecture Guide](architecture.md)** — how the code is organized
- **[Tutorial](tutorial.md)** — step-by-step worked example
- **[Testing Strategy](testing.md)** — cross-backend validation approach
- **API Reference**
  - [JAX API](api/jax.md)
  - [Oracle API](api/oracle.md)
  - [WebGPU API](api/webgpu.md)
  - [Rust/WASM API](api/wasm.md)

## Source

- [GitHub repository](https://github.com/ihh/subby)
