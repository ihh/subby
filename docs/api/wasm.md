# Rust/WASM API Reference

The Rust implementation provides f64 precision phylogenetic computations, compilable to both native Rust and WebAssembly (via wasm-bindgen). It serves as the browser fallback when WebGPU is unavailable.

## Crate: `phylo_wasm`

**Location:** `src/phylo/wasm/`

**Targets:**
- `cdylib` — WebAssembly module via wasm-bindgen
- `rlib` — Native Rust library for CLI and testing

**Feature flags:**
- `wasm` — gates wasm-bindgen dependencies

```toml
[dependencies]
wasm-bindgen = { version = "0.2", optional = true }
js-sys = { version = "0.3", optional = true }

[features]
default = []
wasm = ["dep:wasm-bindgen", "dep:js-sys"]
```

---

## Native Rust API

### `DiagModel`

```rust
pub struct DiagModel {
    pub eigenvalues: Vec<f64>,   // (A,)
    pub eigenvectors: Vec<f64>,  // (A*A,) row-major
    pub pi: Vec<f64>,            // (A,)
}
```

### `log_like(alignment, parent_index, distances, model) -> Vec<f64>`

Compute per-column log-likelihoods.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `&[i32]` | Flat `(R*C)` row-major tokens |
| `parent_index` | `&[i32]` | `(R,)` preorder parent indices |
| `distances` | `&[f64]` | `(R,)` branch lengths |
| `model` | `&DiagModel` | Substitution model |

**Returns:** `Vec<f64>` of length `C`.

### `counts(alignment, parent_index, distances, model, f81_fast_flag) -> Vec<f64>`

Compute expected substitution counts and dwell times.

**Returns:** `Vec<f64>` of length `A*A*C`, row-major `(A, A, C)`.

### `root_prob(alignment, parent_index, distances, model) -> Vec<f64>`

Compute posterior root state distribution.

**Returns:** `Vec<f64>` of length `A*C`, row-major `(A, C)`.

### `mixture_posterior_full(alignment, parent_index, distances, models, log_weights) -> Vec<f64>`

Compute mixture component posteriors.

**Returns:** `Vec<f64>` of length `K*C`, row-major `(K, C)`.

---

## Model constructors

### `hky85_diag(kappa: f64, pi: &[f64]) -> DiagModel`

HKY85 model with closed-form eigendecomposition.

### `jukes_cantor_model(a: usize) -> DiagModel`

Jukes-Cantor model for `a` states.

### `f81_model(pi: &[f64]) -> DiagModel`

F81 model with given equilibrium frequencies.

### `gamma_rate_categories(alpha: f64, k: usize) -> (Vec<f64>, Vec<f64>)`

Discretized gamma rate categories.

### `scale_model(model: &DiagModel, rate_multiplier: f64) -> DiagModel`

Scale eigenvalues by a rate multiplier.

---

## Internal modules

| Module | Functions |
|--------|-----------|
| `tree` | `children_of`, `validate_binary_tree` |
| `model` | Model constructors, `DiagModel` |
| `token` | `token_to_likelihood` |
| `sub_matrices` | `compute_sub_matrices` |
| `pruning` | `upward_pass` |
| `outside` | `downward_pass` |
| `eigensub` | `compute_j`, `eigenbasis_project`, `accumulate_c`, `back_transform` |
| `f81_fast` | `f81_counts` |
| `mixture` | `mixture_posterior` |
| `branch_mask` | `compute_branch_mask` |

---

## WASM bindings

When compiled with `--features wasm`, the crate exports flat-array wasm-bindgen functions:

### `wasm_log_like(alignment, parent_index, distances, eigenvalues, eigenvectors, pi) -> Vec<f64>`

### `wasm_counts(alignment, parent_index, distances, eigenvalues, eigenvectors, pi, f81_fast) -> Vec<f64>`

### `wasm_root_prob(alignment, parent_index, distances, eigenvalues, eigenvectors, pi) -> Vec<f64>`

### `wasm_branch_mask(alignment, parent_index, a) -> Vec<u8>`

All parameters are flat typed arrays. Model parameters are passed individually rather than as a struct.

---

## Building

### Native

```bash
cd src/phylo/wasm
cargo build --release
cargo test
```

### WASM

```bash
cd src/phylo/wasm
wasm-pack build --target web --features wasm
```

Produces `pkg/phylo_wasm_bg.wasm` and `pkg/phylo_wasm.js`.

---

## JS wrapper: PhyloWASM

`src/phylo/webgpu/phylo_wasm.js` wraps the WASM module with the same async API as `PhyloGPU`:

```javascript
import { PhyloWASM } from './phylo_wasm.js';

const engine = await PhyloWASM.create('./phylo_wasm_bg.wasm');
const logLike = await engine.LogLike(alignment, parentIndex, distances,
                                      eigenvalues, eigenvectors, pi);
engine.destroy();
```

### `PhyloWASM.create(wasmUrl)`

Fetch and instantiate the WASM module.

### `PhyloWASM.fromModule(wasmModule)`

Wrap a pre-initialized wasm-bindgen module.

The API methods (`LogLike`, `Counts`, `RootProb`, `MixturePosterior`, `computeBranchMask`, `destroy`) are identical to `PhyloGPU` — see the [WebGPU API reference](webgpu.md).

---

## Precision

All internal computation uses f64. Test tolerance: **atol=1e-8** against golden files.
