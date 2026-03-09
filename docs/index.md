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
from subby.oracle import (
    LogLike, Counts, RootProb, jukes_cantor_model,
    parse_newick, parse_dict, combine_tree_alignment,
)

# Parse tree from Newick and sequences from a dictionary
tree = parse_newick("((human:0.1,chimp:0.15):0.05,(mouse:0.2,rat:0.25):0.1);")
aln = parse_dict({
    "human": "ACGT",
    "chimp": "ACGT",
    "mouse": "CAGT",
    "rat":   "AAGT",
})
combined = combine_tree_alignment(tree, aln)

# Jukes-Cantor model (4-state, uniform equilibrium)
model = jukes_cantor_model(4)

# Compute per-column statistics
log_likelihoods = LogLike(combined.alignment, combined.tree, model)   # (4,)
counts = Counts(combined.alignment, combined.tree, model)              # (4, 4, 4)
root_posterior = RootProb(combined.alignment, combined.tree, model)     # (4, 4)
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
