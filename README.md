# subby

Phylogenetically-informed gene annotation using bidirectional Mamba with in-browser inference.

## Overview

subby annotates gene structures on genomic multiple sequence alignments. It takes phylogenetic sufficient statistics as input features, processes them through a tower of bidirectional Mamba (selective state space) layers, and decodes gene structure annotations via a dual-strand HMM. RNA-Seq evidence can optionally be incorporated through cross-attention.

The model targets in-browser deployment via WebGPU compute shaders, with a Rust/WASM fallback for devices without WebGPU support.

## Components

### Phylogenetic sufficient statistics (`src/phylo/`)

Computes per-column evolutionary features from a multiple sequence alignment and phylogenetic tree using the eigensubstitution accumulation algorithm (Holmes & Rubin, 2002). For each column, this produces:

- **Substitution counts**: expected number of substitutions of each type (A matrix), computed by combining Felsenstein inside-outside partial likelihoods with eigendecomposition-based integrals over branch lengths
- **Dwell times**: expected time spent in each state (diagonal of the A matrix)
- **Rate-category posteriors**: posterior weights over discrete-gamma rate categories (for modeling among-site rate variation)

Three backends: JAX (training), WebGPU (browser inference), Rust/WASM (browser fallback). A pure-Python oracle implementation serves as the cross-language test reference.

Substitution models supported: HKY85, F81, Jukes-Cantor. F81/JC have an O(CRA^2) fast path (vs O(CRA^4) for the general eigensub algorithm), enabling large state spaces (A=64 for triplet features).

### Gene annotation model (`src/model/`)

- **Input projection**: phylo feature vectors (B, F, C) projected to hidden dimension D
- **Mamba tower**: stack of bidirectional selective state space layers with RMSNorm pre-normalization. Each layer runs a forward and reverse selective scan, gates the outputs, and adds a residual connection
- **Track encoder**: shared-weight bidirectional Mamba applied independently to each RNA-Seq track. Each track has 6 channels (2 strands x coverage, donor junctions, acceptor junctions), Borzoi-transformed
- **Cross-attention**: at configurable tower layers, per-position multi-head attention where the main tower queries over RNA-Seq track embeddings
- **HMM decoder**: 27-state dual-strand gene structure transducer (Machine Boss format) consuming 15 Tiberius-compatible emission labels. 1 intergenic state + 13 forward-strand + 13 reverse-strand states enforcing reading frame and splice site grammar

### Data pipeline (`src/data/`)

- MSA tokenization into four parallel streams (substitution, triplet, phase, annotation transfer)
- Phylogenetic feature extraction from tokenized MSAs
- RNA-Seq BAM preprocessing with Borzoi nonlinear compression

## Repository layout

```
src/
  phylo/            Phylogenetic sufficient statistics library
    jax/            JAX implementation (training)
    oracle/         Pure-Python reference implementation (testing)
    webgpu/         WebGPU compute shaders + JS wrapper (browser)
    wasm/           Rust crate compiled to WASM (browser fallback)
    rust/           Native Rust target (preprocessing)
    references/     Holmes & Rubin (2002) paper
  model/            Gene annotation model
    jax/            JAX/Flax training implementation
    webgpu/         WebGPU inference (planned)
  data/             Data pipeline (tokenization, featurization, RNA-Seq)
  inference/        In-browser inference runtime (planned)
tests/              Unit and integration tests (mirrors src/ structure)
scripts/            Build, generation, and utility scripts
docs/               Documentation source (Markdown, built to docs/_site/)
configs/            Training and experiment configurations
data/               Data download scripts and split definitions
notebooks/          Exploration and analysis notebooks
paper/              Manuscript LaTeX source
```

## Documentation

Built documentation is at [ihh.github.io/subby](https://ihh.github.io/subby/). Source is in `docs/`; build with `python scripts/build_docs.py`.

## Key references

- `Model.md` — detailed model specification (featurizations, architecture, loss)
- `PLAN.md` — implementation and evaluation plan
- `src/phylo/references/Holmes_Rubin_2002.tex` — eigensubstitution accumulation algorithm
