# WebGPU API Reference

The WebGPU implementation runs phylogenetic computations in the browser using WGSL compute shaders at f32 precision. Tree traversals dispatch one compute pass per branch step with all alignment columns processed in parallel.

## Entry point

```javascript
import { createPhyloEngine, parseNewick, parseFasta, combineTreeAlignment } from './subby/webgpu/index.js';

const { engine, backend } = await createPhyloEngine({
  shaderBasePath: './subby/webgpu/shaders/',
  wasmUrl: './phylo_wasm_bg.wasm',
});
console.log(`Using ${backend} backend`);  // 'webgpu' or 'wasm'
```

### `createPhyloEngine(options)`

Feature-detects WebGPU and falls back to WASM. Both backends expose the same async API.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `shaderBasePath` | string | `'./shaders/'` | URL prefix for `.wgsl` shader files |
| `wasmUrl` | string | — | URL to compiled WASM module (required for fallback) |
| `backend` | string | — | Force `'webgpu'` or `'wasm'` |
| `shaderSources` | object | — | Pre-loaded shader source strings (bypasses fetch) |
| `wasmModule` | object | — | Pre-initialized WASM module (bypasses fetch) |

**Returns:** `Promise<{ engine, backend }>` where `engine` is `PhyloGPU` or `PhyloWASM`.

---

## PhyloGPU class

### `PhyloGPU.create(shaderBasePath)`

Static async factory. Requests a WebGPU adapter and device, fetches all 10 WGSL shaders.

### `PhyloGPU.createFromSources(shaderSources)`

Static async factory using pre-loaded shader source strings.

---

## Engine API

All methods are async and accept flat typed arrays in row-major layout. The API is identical for both `PhyloGPU` and `PhyloWASM`.

### `engine.LogLike(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi)`

Compute per-column log-likelihoods.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `alignment` | `Int32Array` | `R*C` | Flat row-major alignment tokens |
| `parentIndex` | `Int32Array` | `R` | Preorder parent indices |
| `distances` | `Float32Array` | `R` | Branch lengths |
| `eigenvalues` | `Float32Array` | `A` | Model eigenvalues |
| `eigenvectors` | `Float32Array` | `A*A` | Row-major eigenvector matrix |
| `pi` | `Float32Array` | `A` | Equilibrium distribution |

**Returns:** `Promise<Float32Array>` of length `C`.

### `engine.Counts(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi, f81Fast?)`

Compute expected substitution counts and dwell times.

| Parameter | Type | Description |
|-----------|------|-------------|
| ... | ... | Same as `LogLike` |
| `f81Fast` | `boolean` | Use F81/JC fast path (default: `false`) |

**Returns:** `Promise<Float32Array>` of length `A*A*C` (row-major `(A, A, C)`).

### `engine.RootProb(alignment, parentIndex, distances, eigenvalues, eigenvectors, pi)`

Compute posterior root state distribution.

**Returns:** `Promise<Float32Array>` of length `A*C` (row-major `(A, C)`).

### `engine.MixturePosterior(alignment, parentIndex, distances, models, logWeights)`

Compute posterior over mixture components.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `Int32Array` | Flat alignment |
| `parentIndex` | `Int32Array` | Parent indices |
| `distances` | `Float32Array` | Branch lengths |
| `models` | `Array<{eigenvalues, eigenvectors, pi}>` | $K$ model specs |
| `logWeights` | `Float32Array` | Log prior weights |

**Returns:** `Promise<Float32Array>` of length `K*C` (row-major `(K, C)`).

### `engine.computeBranchMask(alignment, parentIndex, A)`

Compute Steiner tree branch mask (runs on CPU).

**Returns:** `Uint8Array` of length `R*C`.

### `engine.destroy()`

Release GPU resources.

---

## Model constructors

Model constructors are exported from `./subby/webgpu/models.js`:

```javascript
import { jukesCantor, f81, hky85, gy94, diagonalize } from './subby/webgpu/models.js';
```

### `gy94(omega, kappa, pi?)`

Goldman-Yang (1994) codon substitution model. Operates on 61 sense codons.

| Parameter | Type | Description |
|-----------|------|-------------|
| `omega` | `number` | dN/dS ratio (Ka/Ks) |
| `kappa` | `number` | Transition/transversion ratio |
| `pi` | `Float64Array` or `null` | `(61,)` codon equilibrium frequencies (default: uniform $1/61$) |

**Returns:** `{ eigenvalues: Float64Array, eigenvectors: Float64Array, pi: Float64Array }` with $A = 61$.

```javascript
const model = gy94(0.5, 2.0);
// model.eigenvalues has length 61
```

Also available: `jukesCantor(A)`, `f81(pi)`, `hky85(kappa, pi)`, `diagonalize(Q, pi)`.

---

## Format parsers

Standard file format parsers are exported from the same module:

```javascript
import {
  detectAlphabet, parseNewick, parseFasta, parseStockholm,
  parseMaf, parseStrings, combineTreeAlignment,
} from './subby/webgpu/index.js';
```

### `parseNewick(newickStr) -> object`

Parse a Newick tree string. Returns `{ parentIndex: Int32Array, distanceToParent: Float64Array, leafNames, nodeNames, R }`.

### `parseFasta(text, alphabet?) -> object`

Parse FASTA alignment. Returns `{ alignment: Int32Array, leafNames, alphabet, N, C }`.

### `combineTreeAlignment(treeResult, alignmentResult) -> object`

Match leaf names between tree and alignment. Returns `{ alignment: Int32Array, parentIndex, distanceToParent, alphabet, leafNames, R, C }`.

```javascript
const tree = parseNewick('((A:0.1,B:0.2):0.05,C:0.3);');
const aln = parseFasta('>A\nACGT\n>B\nTGCA\n>C\nGGGG\n');
const combined = combineTreeAlignment(tree, aln);
// combined.alignment is a flat (R*C) Int32Array ready for engine.LogLike()
```

Also available: `parseStockholm(text)`, `parseMaf(text)`, `parseStrings(sequences)`, `parseDict(sequences)`, `detectAlphabet(chars)`.

### `parseDict(sequences) -> object`

Parse a `{name: sequence}` object. Returns same shape as `parseFasta`.

```javascript
const aln = parseDict({ human: 'ACGT', mouse: 'TGCA' });
```

### `geneticCode() -> object`

Return the standard genetic code. Codons in ACGT lexicographic order; stop codons marked with `'*'`.

**Returns:** `{ codons: string[], aminoAcids: string[], senseMask: boolean[], senseIndices: Int32Array, codonToSense: Int32Array, senseCodons: string[], senseAminoAcids: string[] }`.

```javascript
import { geneticCode } from './subby/webgpu/index.js';
const gc = geneticCode();
console.log(gc.senseCodons.length);  // 61
```

### `codonToSense(alignment, A?) -> object`

Remap a 64-codon tokenized flat alignment to 61-sense-codon tokens. Stop codons become the gap token.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `Int32Array` | Flat token array (64-codon encoding) |
| `A` | `number` | Input alphabet size (default 64) |

**Returns:** `{ alignment: Int32Array, A_sense: 61, alphabet: string[] }`.

### `KmerIndex`

Maps between column tuples and output alignment indices. Provides O(1) lookup in both directions.

```javascript
import { KmerIndex } from './subby/webgpu/index.js';

const index = new KmerIndex([[0, 1], [2, 3], [4, 5]]);
index.tupleToIdx([2, 3]);  // → 1
index.idxToTuple(0);       // → [0, 1]
index.length;              // → 3
```

### `slidingWindows(C, k, stride?, offset?, edge?) -> number[][]`

Generate column index tuples for sliding-window k-mer tokenization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `C` | `number` | Number of columns |
| `k` | `number` | Window size |
| `stride` | `number` or `null` | Step between window starts (default: `k`) |
| `offset` | `number` | Starting column (default: `0`) |
| `edge` | `string` | `'truncate'` (default) or `'pad'` |

**Returns:** `(M, k)` array of column indices. `-1` for out-of-bounds (with `edge='pad'`).

### `allColumnKtuples(C, k, ordered?) -> number[][]`

Generate all k-tuples of column indices. **WARNING:** $O(C^k)$.

| Parameter | Type | Description |
|-----------|------|-------------|
| `C` | `number` | Number of columns |
| `k` | `number` | Tuple size |
| `ordered` | `boolean` | `true` (default): permutations; `false`: combinations |

**Returns:** `(T, k)` array of column index tuples.

### `kmerTokenize(alignment, N, C, A, kOrTuples, gapMode?, alphabet?) -> object`

Convert a single-character token alignment to k-mer tokens. Accepts either an integer `k` (backward compatible, non-overlapping windows where `C` must be divisible by `k`) or a `(T, k)` array of column index tuples.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `Int32Array` | Flat `(N*C)` row-major tokens |
| `N` | `number` | Number of sequences |
| `C` | `number` | Number of columns |
| `A` | `number` | Single-character alphabet size |
| `kOrTuples` | `number` or `number[][]` | Integer `k` or column tuples |
| `gapMode` | `string` | `'any'` (default) or `'all'` |
| `alphabet` | `string[]` or `null` | Single-character labels for building k-mer labels |

**Returns:** `{ alignment: Int32Array, Ak: number, N: number, Ck: number, index: KmerIndex, alphabet?: string[] }`. Token encoding: 0..$A^k - 1$ for observed k-mers, $A^k$ for ungapped-unobserved, $A^k + 1$ for gap.

```javascript
import { kmerTokenize, slidingWindows } from './subby/webgpu/index.js';

// Non-overlapping codons (backward compatible)
const kmer = kmerTokenize(dnaAlignment, 3, 9, 4, 3, 'any', ['A','C','G','T']);

// Overlapping stride-1 windows
const windows = slidingWindows(100, 3, 1);
const result = kmerTokenize(dnaAlignment, 3, 100, 4, windows);
result.index.tupleToIdx([5, 6, 7]);  // → 5
```

---

## WGSL Shaders

Each shader is a WGSL compute shader with `@workgroup_size(64)`:

| Shader | Dispatch | Description |
|--------|----------|-------------|
| `token_to_likelihood.wgsl` | `ceil(R*C/64)` | Token → likelihood vectors |
| `compute_sub_matrices.wgsl` | `ceil(R/64)` | Eigendecomposition → $M(t)$ |
| `upward_step.wgsl` | `ceil(C/64)` × (R-1) | One step of Felsenstein pruning |
| `downward_step.wgsl` | `ceil(C/64)` × (R-1) | One step of outside algorithm |
| `compute_J.wgsl` | `ceil(R/64)` | $J$ interaction matrix |
| `eigenbasis_project.wgsl` | `ceil(R*C/64)` | Project to eigenbasis |
| `accumulate_C.wgsl` | `ceil(A*A*C/64)` | Sum eigenbasis contributions |
| `back_transform.wgsl` | `ceil(A*A*C/64)` | Eigenbasis → natural basis |
| `f81_fast.wgsl` | `ceil(A*A*C/64)` | F81/JC direct counts |
| `mixture_posterior.wgsl` | `ceil(C/64)` | Softmax over components |

### Buffer layout

All arrays are flattened to 1D `storage` buffers in row-major (C-order) layout:

- Shape `(R, C, A)`: offset = `r*C*A + c*A + a`
- Shape `(A, A, C)`: offset = `i*A*C + j*C + c`

Dimensions are passed as uniforms.

### Precision

f32 with log-space rescaling to prevent underflow. Test tolerance: **atol=1e-3** against golden files.
