# src/phylo/

Phylogenetic sufficient statistics library implementing the eigensubstitution accumulation algorithm (Holmes & Rubin, 2002).

Computes per-column substitution counts, dwell times, and rate-category posteriors from a multiple sequence alignment and phylogenetic tree. Public API: `LogLike`, `Counts`, `RootProb`, `MixturePosterior`.

## Backends

- `jax/` — JAX implementation for training. Vectorized over columns (C), batched over rate categories (K). See `docs/api/jax.md`.
- `oracle/` — Pure-Python (numpy) reference with explicit for-loops. Test oracle for cross-language validation. See `docs/api/oracle.md`.
- `webgpu/` — WGSL compute shaders with JS wrapper for browser inference. See `docs/api/webgpu.md`.
- `wasm/` — Rust crate compiled to WASM via wasm-bindgen. Browser fallback. See `docs/api/wasm.md`.
- `rust/` — Native Rust target (same crate as `wasm/`, different build target).
- `references/` — Holmes & Rubin (2002) paper (PDF and LaTeX source).

## Tests

`tests/test_phylo/` — unit tests for all JAX components, oracle-vs-JAX cross-validation, and golden file tests. Golden test data is in `tests/test_phylo/golden/`.
