# subby

Phylogenetically-informed gene annotation using bidirectional Mamba with in-browser inference.

## Project overview

subby is a gene annotation model that uses phylogenetic sufficient statistics (substitution counts, dwell times, rate-category posteriors) derived from multiple sequence alignments as input features, processed by a tower of bidirectional Mamba layers, with an HMM decoder (via Machine Boss) producing gene structure annotations. It optionally incorporates RNA-Seq evidence via cross-attention. The model targets in-browser deployment via WebGPU with a WASM fallback.

## Repository structure

```
src/
  phylo/          # Phylogenetic sufficient statistics (Felsenstein pruning + eigensubstitution accumulation)
    jax/          # JAX implementation
    webgpu/       # WebGPU compute shader implementation
    wasm/         # WebAssembly fallback
    rust/         # Rust implementation
  model/          # Mamba tower, HMM decoder, RNA-Seq encoder
    jax/          # JAX training implementation
    webgpu/       # WebGPU inference implementation
  data/           # Data pipeline: tiling, splitting, featurization
  inference/      # In-browser inference runtime
data/             # Data configs, download scripts, split definitions
configs/          # Training and experiment configs
scripts/          # Training, evaluation, ablation scripts
tests/            # Unit and integration tests
notebooks/        # Exploration and analysis notebooks
paper/            # Manuscript LaTeX source
```

## Key files

- `Model.md` — detailed model specification (featurizations, architecture, loss)
- `Holmes_Rubin_2002.tex` — reference for eigensubstitution accumulation algorithm
- `toy_felsenstein_pruning.py` — prototype JAX implementation of Felsenstein pruning
- `PLAN.md` — implementation, evaluation, and publication plan

## Architecture

- **Featurization**: HKY85 mixture (16+K features), triplet Jukes-Cantor (128 features), phase (12 features), annotation transfer (3M features)
- **Core**: Tower of bidirectional Mamba layers with RmsNorm pre-normalization
- **RNA-Seq encoder**: Bidirectional Mamba on 6-channel coverage/junction tensors, integrated via interleaved cross-attention
- **Decoder**: Forward-Backward HMM (15 Tiberius-compatible annotation states) via Machine Boss
- **Loss**: Categorical cross-entropy over annotation labels
- **Equivariance**: Reverse complement via data augmentation (not baked in)

## Conventions

- JAX for training; WebGPU + WASM for browser inference
- Rust for the phylo library's native backend
- All parallelizable operations should use `jax.lax.scan`, `vmap`, or equivalent
- Phylogenetic computations operate on all MSA columns simultaneously
- Use F81/Jukes-Cantor for large state spaces (O(CRA^4) instead of O(CRA^6))
- Data splits must be disjoint by both species AND genomic region to avoid homology leakage
