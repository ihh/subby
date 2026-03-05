# Architecture Guide

## Repository layout

```
subby/
├── jax/                    # JAX training implementation (f64, GPU-accelerated)
│   ├── __init__.py         # Public API: LogLike, LogLikeCustomGrad, Counts, RootProb, MixturePosterior
│   ├── types.py            # Tree, DiagModel, IrrevDiagModel, RateModel named tuples
│   ├── models.py           # hky85_diag, jukes_cantor_model, f81_model, gamma, scale
│   ├── diagonalize.py      # diagonalize_rate_matrix, compute_sub_matrices
│   ├── pruning.py          # upward_pass (Felsenstein pruning via jax.lax.scan)
│   ├── outside.py          # downward_pass (outside algorithm via jax.lax.scan)
│   ├── eigensub.py         # compute_J, eigenbasis_project, accumulate_C, back_transform
│   ├── vjp.py              # Custom VJP for LogLike (fast distance gradients via Fisher identity)
│   ├── f81_fast.py         # f81_counts (O(CRA²) direct computation)
│   ├── mixture.py          # mixture_posterior (softmax over components)
│   ├── components.py       # compute_branch_mask (Steiner tree identification)
│   └── _utils.py           # token_to_likelihood, children_of, rescale, validate
│
├── oracle/                 # NumPy reference implementation (f64, explicit loops)
│   ├── __init__.py         # Re-exports all public functions
│   └── oracle.py           # All algorithms as nested for-loops
│
├── webgpu/                 # WebGPU browser implementation (f32)
│   ├── index.js            # createPhyloEngine() — unified entry point
│   ├── phylo_gpu.js        # PhyloGPU class — WebGPU backend
│   ├── phylo_wasm.js       # PhyloWASM class — WASM fallback backend
│   └── shaders/            # WGSL compute shaders (one per kernel)
│
├── wasm/                   # Rust → WASM crate (f64)
│
└── rust/                   # Symlink/pointer to wasm/ for native compilation
    └── README
```

## Data representations

### Tree

A binary tree of $R$ nodes in **preorder** layout (parents before children):

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `parentIndex` | int32 | `(R,)` | Parent of node $n$; `parentIndex[0] = -1` (root) |
| `distanceToParent` | float | `(R,)` | Branch length from node $n$ to its parent |

Preorder invariant: `parentIndex[n] < n` for all `n > 0`.

Nodes $0 \ldots R-1$ are ordered such that node 0 is the root. Internal nodes have exactly 2 children; leaves have 0.

### Alignment

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `alignment` | int32 | `(R, C)` | Token per sequence per column |

Token encoding:

| Token | Meaning | Likelihood vector |
|-------|---------|-------------------|
| `0` to `A-1` | Observed nucleotide | One-hot |
| `A` | Ungapped, unobserved | All ones |
| `A+1` or `-1` | Gapped | All ones |

### Model (diagonalized)

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `eigenvalues` | float | `(*H, A)` | Eigenvalues of symmetrized rate matrix |
| `eigenvectors` | float | `(*H, A, A)` | Eigenvectors; `v[a,k]` = component $a$ of eigenvector $k$ |
| `pi` | float | `(*H, A)` | Equilibrium distribution |

`*H` denotes optional leading batch dimensions (e.g., `(K,)` for rate categories).

**JAX**: `DiagModel` named tuple. **Oracle**: plain dict with the same keys. **WebGPU/WASM**: flat typed arrays.

## Computation pipeline

```
alignment ──→ token_to_likelihood ──→ U (inside vectors)
                                          │
model ──→ compute_sub_matrices ──────────→├──→ upward_pass ──→ logLike
                                          │         │
                                          │         ▼
                                          └──→ downward_pass ──→ D (outside vectors)
                                                    │
model ──→ compute_J ────────────────────────────────├──→ eigenbasis_project
                                                    │         │
                                                    │         ▼
                                                    └──→ accumulate_C ──→ back_transform ──→ counts
```

Alternative for F81/JC models:
```
U, D, logNormU, logNormD, logLike ──→ f81_counts ──→ counts
```

## Backend-specific design

### JAX

- Tree traversals implemented as `jax.lax.scan` over branches in post/preorder
- Column chunking (`maxChunkSize`) for memory control
- All operations support batched models via `*H` leading dimensions
- Differentiable — `jax.grad` works through `LogLike`; `LogLikeCustomGrad` provides a faster custom VJP for distance gradients via the Fisher identity
- Per-column models — `LogLike`, `Counts`, `RootProb` accept a list of models (one per column) for position-specific substitution rates

### Oracle

- Identical algorithm, explicit `for r in range(R): for c in range(C): for a in range(A):` loops
- No vectorization — deliberately slow but obviously correct
- Serves as the numerically trustworthy test oracle (atol=1e-8 vs JAX)

### WebGPU

- One WGSL shader per algorithmic kernel
- Tree traversals dispatch one compute pass per branch step (R-1 sequential passes), all C columns parallel within each pass
- f32 precision with rescaling; test tolerance atol=1e-3

### Rust/WASM

- Single crate, dual target: `cdylib` (WASM via wasm-bindgen) + `rlib` (native)
- All internal computation in f64
- Same algorithm as oracle, but in idiomatic Rust

## Testing architecture

```
                    ┌─────────────┐
                    │   JAX impl  │ ◄── unit tests
                    └──────┬──────┘
                           │ atol=1e-8
                    ┌──────▼──────┐
                    │   Oracle    │ ◄── comparison tests
                    └──────┬──────┘
                           │ generates
                    ┌──────▼──────┐
                    │ Golden JSON │ ── 6 test cases
                    └──┬──────┬──┘
              atol=1e-3│      │atol=1e-8
                 ┌─────▼──┐ ┌─▼──────┐
                 │ WebGPU │ │  WASM  │
                 └────────┘ └────────┘
```
