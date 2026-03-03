# src/phylo/jax/

JAX implementation of phylogenetic sufficient statistics. Used for training.

Vectorized over alignment columns (C) and batched over rate categories (K) via `jax.vmap`. All tree traversals use `jax.lax.scan`.

## Modules

- `types.py` — `Tree`, `DiagModel`, `RateModel` named tuples
- `models.py` — `hky85_diag`, `jukes_cantor_model`, `f81_model`, `gamma_rate_categories`, `scale_model`
- `diagonalize.py` — Rate matrix eigendecomposition, substitution matrix computation
- `pruning.py` — Felsenstein upward pass (inside algorithm) with log-rescaling
- `outside.py` — Downward pass (outside algorithm)
- `eigensub.py` — J matrices, eigenbasis projection, C accumulation, back-transform
- `f81_fast.py` — O(CRA^2) closed-form for F81/JC models
- `mixture.py` — Rate-category mixture posteriors
- `components.py` — Tree construction, token-to-likelihood, branch masking
- `_utils.py` — Internal helpers

See `docs/api/jax.md` for full API reference.
