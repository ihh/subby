# subby

Phylogenetic sufficient statistics via eigensubstitution accumulation.

## Project overview

subby computes per-column evolutionary features from multiple sequence alignments using the eigensubstitution accumulation algorithm (Holmes & Rubin, 2002). Given an MSA and a phylogenetic tree, it computes log-likelihoods, substitution counts, dwell times, and root state distributions via the Felsenstein inside-outside algorithm combined with eigendecomposition-based branch integrals.

## Repository structure

```
subby/              # Phylogenetic sufficient statistics library
  jax/              # JAX implementation (training)
  oracle/           # Pure-Python reference implementation (testing)
  webgpu/           # WebGPU compute shader implementation
  wasm/             # WebAssembly fallback
  rust/             # Rust implementation
references/         # Holmes & Rubin (2002) paper
tests/              # Unit and integration tests
benchmarks/         # Performance benchmarks
scripts/            # Build, generation, and utility scripts
docs/               # Documentation source
```

## Key files

- `Model.md` — detailed specification of the eigensubstitution accumulation algorithm
- `references/Holmes_Rubin_2002.tex` — reference for eigensubstitution accumulation algorithm
- `scripts/toy_felsenstein_pruning.py` — prototype JAX implementation of Felsenstein pruning

## Conventions

- JAX for training; WebGPU + WASM for browser inference
- Rust for the native backend
- All parallelizable operations should use `jax.lax.scan`, `vmap`, or equivalent
- Phylogenetic computations operate on all MSA columns simultaneously
- Use F81/Jukes-Cantor for large state spaces (O(CRA^4) instead of O(CRA^6))

## Public API

The main entry points are:
- `subby.jax.LogLike(alignment, tree, model)` — per-column log-likelihoods
- `subby.jax.LogLikeCustomGrad(alignment, tree, model)` — same as LogLike but with custom VJP for faster distance gradients
- `subby.jax.Counts(alignment, tree, model)` — expected substitution counts and dwell times
- `subby.jax.RootProb(alignment, tree, model)` — posterior root state distribution
- `subby.jax.MixturePosterior(alignment, tree, models, log_weights)` — mixture component posteriors
- `subby.oracle.*` — identical API using pure NumPy (test reference)

All high-level functions (`LogLike`, `Counts`, `RootProb`) accept a list of models for per-column substitution rates (one model per alignment column).
