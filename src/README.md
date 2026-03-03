# src/

Source code for the subby gene annotation system.

- `phylo/` — Phylogenetic sufficient statistics library (multi-backend). See `docs/api/jax.md`, `docs/api/oracle.md`, `docs/api/webgpu.md`, `docs/api/wasm.md`.
- `model/` — Mamba tower, track encoder, cross-attention, HMM decoder. See `docs/api/model.md`.
- `data/` — Data pipeline: MSA tokenization, phylo feature extraction, RNA-Seq BAM preprocessing. See `docs/api/rnaseq.md`.
- `inference/` — In-browser inference runtime (planned).

Tests are in `tests/`, mirroring this directory structure.
