# src/phylo/oracle/

Pure-Python (numpy-only) reference implementation of phylogenetic sufficient statistics.

Every algorithm is written as explicit nested for-loops over indices. No vectorization, no einsum, no JAX. Designed to be obviously correct at the expense of speed. Serves as the cross-language test oracle for WebGPU and WASM backends.

Same public API as `src/phylo/jax/`: `LogLike`, `Counts`, `RootProb`, `MixturePosterior`.

See `docs/api/oracle.md` for API reference. Cross-validation tests in `tests/test_phylo/test_oracle.py`.
