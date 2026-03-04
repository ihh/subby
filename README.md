# subby
[![Tests](https://github.com/ihh/subby/actions/workflows/test.yml/badge.svg)](https://github.com/ihh/subby/actions/workflows/test.yml)

Phylogenetic sufficient statistics via eigensubstitution accumulation.

## Overview

subby computes per-column evolutionary features from a multiple sequence alignment and phylogenetic tree using the eigensubstitution accumulation algorithm (Holmes & Rubin, 2002). For each column, this produces:

- **Substitution counts**: expected number of substitutions of each type, computed by combining Felsenstein inside-outside partial likelihoods with eigendecomposition-based integrals over branch lengths
- **Dwell times**: expected time spent in each state (diagonal of the counts matrix)
- **Log-likelihoods**: marginal likelihood of the observed data in each column
- **Root posteriors**: posterior state distribution at the root node

Three backends: JAX (training), WebGPU (browser inference), Rust/WASM (browser fallback). A pure-Python oracle implementation serves as the cross-language test reference.

Substitution models supported: HKY85, F81, Jukes-Cantor, and arbitrary reversible or irreversible rate matrices. F81/JC have an O(CRA^2) fast path (vs O(CRA^4) for the general eigensub algorithm), enabling large state spaces (A=64 for triplet features).

## Installation

```bash
pip install -e .          # minimal (numpy only)
pip install -e ".[jax]"   # with JAX backend
```

## Quick start

### Python (JAX)

```python
import jax.numpy as jnp
from subby.jax import LogLike, Counts
from subby.jax.types import Tree
from subby.jax.models import jukes_cantor_model

tree = Tree(
    parentIndex=jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32),
    distanceToParent=jnp.array([0.0, 0.1, 0.2, 0.15, 0.25]),
)
alignment = jnp.array([[0,1],[2,3],[1,0],[3,2],[0,1]], dtype=jnp.int32)
model = jukes_cantor_model(4)

log_likelihoods = LogLike(alignment, tree, model)
counts = Counts(alignment, tree, model)
```

### JavaScript (browser)

```javascript
import { createPhyloEngine, jukesCantor } from './subby/webgpu/index.js';

// Auto-detects WebGPU; falls back to WASM
const { engine } = await createPhyloEngine({
  shaderBasePath: './subby/webgpu/shaders/',
  wasmUrl: './phylo_wasm_bg.wasm',
});

const model = jukesCantor(4);  // 4 DNA states
const parentIndex = new Int32Array([-1, 0, 0, 1, 1]);
const distances = new Float32Array([0.0, 0.1, 0.2, 0.15, 0.25]);
const alignment = new Int32Array([0,1, 2,3, 1,0, 3,2, 0,1]);  // 5 nodes × 2 columns

const logLike = await engine.LogLike(
  alignment, parentIndex, distances,
  model.eigenvalues, model.eigenvectors, model.pi,
);
const counts = await engine.Counts(
  alignment, parentIndex, distances,
  model.eigenvalues, model.eigenvectors, model.pi,
);

engine.destroy();
```

## Repository layout

```
subby/              Phylogenetic sufficient statistics library
  jax/              JAX implementation
  oracle/           Pure-Python reference implementation
  webgpu/           WebGPU compute shaders + JS wrapper (browser)
  wasm/             Rust crate compiled to WASM (browser fallback)
  rust/             Native Rust target (preprocessing)
references/         Holmes & Rubin (2002) paper
tests/              Unit and integration tests
benchmarks/         Performance benchmarks
scripts/            Build, generation, and utility scripts
docs/               Documentation source
```

## Documentation

Built documentation is at [ihh.github.io/subby](https://ihh.github.io/subby/). Source is in `docs/`; build with `python scripts/build_docs.py`.

## Key references

- `Model.md` — detailed specification of the eigensubstitution accumulation algorithm
- `references/Holmes_Rubin_2002.tex` — eigensubstitution accumulation paper
