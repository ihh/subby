# Rust/WASM API Reference

The Rust implementation provides f64 precision phylogenetic computations, compilable to both native Rust and WebAssembly (via wasm-bindgen). It serves as the browser fallback when WebGPU is unavailable.

## Crate: `phylo_wasm`

**Location:** `subby/wasm/`

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

### `eig_symmetric(s: &mut [f64], a: usize) -> (Vec<f64>, Vec<f64>)`

Cyclic Jacobi eigendecomposition for a symmetric matrix. Computes eigenvalues and eigenvectors in-place.

| Parameter | Type | Description |
|-----------|------|-------------|
| `s` | `&mut [f64]` | Flat `(a*a)` row-major symmetric matrix (modified in-place) |
| `a` | `usize` | Matrix dimension |

**Returns:** `(eigenvalues, eigenvectors)` — eigenvalues as `Vec<f64>` of length `a`, eigenvectors as `Vec<f64>` of length `a*a` (row-major, `v[i*a+k]` = component $i$ of eigenvector $k$).

### `diagonalize_rate_matrix(sub_rate: &[f64], pi: &[f64]) -> DiagModel`

Diagonalize a reversible rate matrix via symmetrization and Jacobi eigendecomposition. Symmetrizes $Q$ as $S_{ij} = Q_{ij} \sqrt{\pi_i} / \sqrt{\pi_j}$, then computes eigenvalues/eigenvectors of $S$.

| Parameter | Type | Description |
|-----------|------|-------------|
| `sub_rate` | `&[f64]` | Flat `(A*A)` row-major rate matrix $Q$ |
| `pi` | `&[f64]` | `(A,)` equilibrium distribution |

**Returns:** `DiagModel`.

### `gy94_model(omega: f64, kappa: f64, pi: Option<&[f64]>) -> DiagModel`

Goldman-Yang (1994) codon substitution model. Operates on 61 sense codons.

| Parameter | Type | Description |
|-----------|------|-------------|
| `omega` | `f64` | dN/dS ratio (Ka/Ks) |
| `kappa` | `f64` | Transition/transversion ratio |
| `pi` | `Option<&[f64]>` | `(61,)` codon equilibrium frequencies (`None` = uniform $1/61$) |

**Returns:** `DiagModel` with $A = 61$ states.

---

## Format utilities

### `GeneticCode`

```rust
pub struct GeneticCode {
    pub codons: Vec<String>,           // 64 codon strings
    pub amino_acids: Vec<char>,        // 64 amino acid letters (stop = '*')
    pub sense_mask: Vec<bool>,         // (64,) true for sense codons
    pub sense_indices: Vec<usize>,     // (61,) indices of sense codons in 0..63
    pub codon_to_sense: Vec<i32>,      // (64,) maps codon idx to sense idx (stop -> -1)
    pub sense_codons: Vec<String>,     // 61 sense codon strings
    pub sense_amino_acids: Vec<char>,  // 61 amino acid letters
}
```

### `genetic_code() -> GeneticCode`

Return the standard genetic code. Codons in ACGT lexicographic order; stop codons (TAA, TAG, TGA) marked with `'*'`.

### `codon_to_sense(alignment: &[i32], n: usize, c: usize, a: usize) -> Vec<i32>`

Remap a 64-codon tokenized flat alignment to 61-sense-codon tokens. Stop codons become the gap token.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `&[i32]` | Flat `(n*c)` row-major tokens (64-codon encoding) |
| `n` | `usize` | Number of sequences |
| `c` | `usize` | Number of columns |
| `a` | `usize` | Input codon alphabet size (64) |

**Returns:** `Vec<i32>` of length `n*c` with $A_\text{sense} = 61$ tokens.

### `KmerResult`

```rust
pub struct KmerResult {
    pub alignment: Vec<i32>,  // flat (n * c_k) tokens
    pub a_kmer: usize,        // A^k
    pub n: usize,             // number of sequences
    pub c_k: usize,           // C / k columns
}
```

### `kmer_tokenize(alignment: &[i32], n: usize, c: usize, a: usize, k: usize, gap_mode: &str) -> KmerResult`

Convert single-character tokens to k-mer tokens. Groups `k` consecutive columns into one k-mer column (non-overlapping). `c` must be divisible by `k`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `&[i32]` | Flat `(n*c)` row-major tokens |
| `n` | `usize` | Number of sequences |
| `c` | `usize` | Number of columns |
| `a` | `usize` | Single-character alphabet size |
| `k` | `usize` | k-mer size (e.g., 3 for codons) |
| `gap_mode` | `&str` | `"any"`: gap in any position gaps the k-mer; `"all"`: only all-gap k-mers become gaps |

**Returns:** `KmerResult`. Token encoding: 0..$A^k - 1$ for observed k-mers, $A^k$ for ungapped-unobserved, $A^k + 1$ for gap.

---

## Internal modules

| Module | Functions |
|--------|-----------|
| `tree` | `children_of`, `validate_binary_tree` |
| `model` | Model constructors, `DiagModel`, `eig_symmetric`, `diagonalize_rate_matrix`, `gy94_model` |
| `token` | `token_to_likelihood` |
| `sub_matrices` | `compute_sub_matrices` |
| `pruning` | `upward_pass` |
| `outside` | `downward_pass` |
| `eigensub` | `compute_j`, `eigenbasis_project`, `accumulate_c`, `back_transform` |
| `f81_fast` | `f81_counts` |
| `mixture` | `mixture_posterior` |
| `branch_mask` | `compute_branch_mask` |
| `formats` | `parse_newick`, `parse_fasta`, `parse_strings`, `combine_tree_alignment`, `detect_alphabet`, `genetic_code`, `codon_to_sense`, `kmer_tokenize` |

---

## WASM bindings

When compiled with `--features wasm`, the crate exports flat-array wasm-bindgen functions:

### `wasm_log_like(alignment, parent_index, distances, eigenvalues, eigenvectors, pi) -> Vec<f64>`

### `wasm_counts(alignment, parent_index, distances, eigenvalues, eigenvectors, pi, f81_fast) -> Vec<f64>`

### `wasm_root_prob(alignment, parent_index, distances, eigenvalues, eigenvectors, pi) -> Vec<f64>`

### `wasm_branch_mask(alignment, parent_index, a) -> Vec<u8>`

All parameters are flat typed arrays. Model parameters are passed individually rather than as a struct.

### Format parser bindings

### `wasm_parse_newick(newick_str) -> JsValue`

Parse a Newick string. Returns object with `parentIndex` (Int32Array), `distanceToParent` (Float64Array), `leafNames`, `nodeNames`, `R`.

### `wasm_parse_fasta(text) -> JsValue`

Parse FASTA text. Returns object with `alignment` (Int32Array, flat N*C), `leafNames`, `alphabet`, `N`, `C`.

### `wasm_parse_strings(sequences) -> JsValue`

Parse an array of equal-length strings. Returns object with `alignment` (Int32Array, flat N*C), `alphabet`, `N`, `C`.

---

## Building

### Native

```bash
cd subby/wasm
cargo build --release
cargo test
```

### WASM

```bash
cd subby/wasm
wasm-pack build --target web --features wasm
```

Produces `pkg/phylo_wasm_bg.wasm` and `pkg/phylo_wasm.js`.

---

## JS wrapper: PhyloWASM

`subby/webgpu/phylo_wasm.js` wraps the WASM module with the same async API as `PhyloGPU`:

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
