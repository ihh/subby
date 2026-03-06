# Oracle API Reference

The oracle is a pure NumPy reference implementation using explicit for-loops. It serves as the numerically trustworthy test oracle for all other backends (atol=1e-8 vs JAX).

**Dependencies:** `numpy`, `scipy` (for `gammainc` only).

## Data representations

### Tree

A plain dict:

```python
tree = {
    'parentIndex': np.array([-1, 0, 0, 1, 1], dtype=np.int32),
    'distanceToParent': np.array([0.0, 0.1, 0.2, 0.15, 0.25]),
}
```

### Model

A plain dict:

```python
model = {
    'eigenvalues': np.array([0.0, -1.333, -1.333, -1.333]),
    'eigenvectors': V,  # (A, A) orthogonal matrix
    'pi': np.array([0.25, 0.25, 0.25, 0.25]),
}
```

### Alignment

`(R, C)` int32 array with token encoding:

| Token | Meaning | Likelihood vector |
|-------|---------|-------------------|
| `0` to `A-1` | Observed state | One-hot |
| `A` | Ungapped, unobserved | All ones |
| `A+1` or `-1` | Gapped | All ones |

---

## High-level API

```python
from subby.oracle import LogLike, Counts, RootProb, MixturePosterior
```

### `LogLike(alignment, tree, model)`

Compute per-column log-likelihoods.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `dict` | `{'parentIndex': ..., 'distanceToParent': ...}` |
| `model` | `dict` | `{'eigenvalues': ..., 'eigenvectors': ..., 'pi': ...}` |

**Returns:** `(C,)` float64 array.

### `Counts(alignment, tree, model, f81_fast=False)`

Compute expected substitution counts and dwell times.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `dict` | Tree dict |
| `model` | `dict` | Model dict |
| `f81_fast` | `bool` | Use $O(CRA^2)$ fast path (F81/JC only) |

**Returns:** `(A, A, C)` float64 tensor. Diagonal = dwell times, off-diagonal = substitution counts.

### `RootProb(alignment, tree, model)`

Compute posterior root state distribution per column.

**Returns:** `(A, C)` float64 array. Sums to 1 over $A$ for each column.

### `MixturePosterior(alignment, tree, models, log_weights)`

Compute posterior over mixture components per column.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `dict` | Tree dict |
| `models` | `list[dict]` | $K$ model dicts |
| `log_weights` | `(K,)` | Log prior weights |

**Returns:** `(K, C)` float64 posterior probabilities.

---

## Model constructors

### `hky85_diag(kappa, pi)`

HKY85 substitution model.

```python
model = hky85_diag(kappa=2.0, pi=np.array([0.3, 0.2, 0.25, 0.25]))
```

### `jukes_cantor_model(A)`

Jukes-Cantor model for $A$ states.

```python
model = jukes_cantor_model(4)    # nucleotide
model = jukes_cantor_model(64)   # triplet (codon-like)
```

### `f81_model(pi)`

F81 model with non-uniform equilibrium frequencies.

### `gamma_rate_categories(alpha, K)`

Discretized gamma rate categories (Yang 1994).

```python
rates, weights = gamma_rate_categories(alpha=0.5, K=4)
# rates: [0.03, 0.26, 0.82, 2.89] (approximately)
# weights: [0.25, 0.25, 0.25, 0.25]
```

### `scale_model(model, rate_multiplier)`

Scale eigenvalues by a rate multiplier (scalar).

---

## Tree utilities

### `children_of(parentIndex)`

Compute child and sibling arrays.

```python
left_child, right_child, sibling = children_of(parentIndex)
```

**Returns:** Three `(R,)` int arrays. `-1` for absent children (leaves) or absent sibling (root).

### `validate_binary_tree(parentIndex)`

Assert every internal node has exactly 2 children. Raises `ValueError` if not.

### `compute_branch_mask(alignment, parentIndex, A)`

Compute Steiner tree branch mask per column.

```python
mask = compute_branch_mask(alignment, parentIndex, A=4)
# Shape: (R, C) — boolean, True where branch is informative
```

---

## Format parsers

Standard phylogenetic file format parsers are re-exported from `subby.oracle` for convenience:

```python
from subby.oracle import parse_newick, parse_fasta, parse_stockholm, parse_maf, parse_strings, combine_tree_alignment, detect_alphabet
```

### `parse_newick(newick_str) -> dict`

Parse a Newick tree string into subby's internal tree representation.

**Returns:** dict with `parentIndex` (int32), `distanceToParent` (float64), `leaf_names`, `node_names`.

```python
tree = parse_newick("((A:0.1,B:0.2):0.05,C:0.3);")
```

### `parse_fasta(text, alphabet=None) -> dict`

Parse FASTA-formatted alignment. Auto-detects DNA/RNA/protein alphabet.

**Returns:** dict with `alignment` (int32 `(N, C)`), `leaf_names`, `alphabet`.

### `parse_stockholm(text, alphabet=None) -> dict`

Parse Stockholm format. Extracts `#=GF NH` tree if present. When a tree is found, automatically calls `combine_tree_alignment`.

### `parse_maf(text, alphabet=None) -> dict`

Parse MAF (Multiple Alignment Format). Concatenates alignment blocks; fills gaps for missing species.

### `parse_strings(sequences, alphabet=None) -> dict`

Parse a list of equal-length strings into an alignment tensor.

### `combine_tree_alignment(tree_result, alignment_result) -> dict`

Map leaf sequences to tree positions by name. Creates full `(R, C)` alignment with internal nodes filled with the ungapped-unobserved token.

**Returns:** dict with `alignment` (int32 `(R, C)`), `parentIndex`, `distanceToParent`, `alphabet`, `leaf_names`.

```python
tree = parse_newick("((A:0.1,B:0.2):0.05,C:0.3);")
aln = parse_fasta(">A\nACGT\n>B\nTGCA\n>C\nGGGG\n")
combined = combine_tree_alignment(tree, aln)
ll = LogLike(combined['alignment'],
             {'parentIndex': combined['parentIndex'],
              'distanceToParent': combined['distanceToParent']},
             jukes_cantor_model(4))
```

---

## Low-level pipeline functions

### `token_to_likelihood(alignment, A)`

Convert integer tokens to likelihood vectors.

**Returns:** `(R, C, A)` float64.

### `compute_sub_matrices(model, distances)`

Compute transition probability matrices.

**Returns:** `(R, A, A)` float64. Rows sum to 1.

### `upward_pass(alignment, tree, subMatrices, rootProb)`

Felsenstein pruning (postorder).

**Returns:** `(U, logNormU, logLike)`:
- `U`: `(R, C, A)` rescaled inside vectors
- `logNormU`: `(R, C)` log-normalizers
- `logLike`: `(C,)` log-likelihoods

### `downward_pass(U, logNormU, tree, subMatrices, rootProb, alignment)`

Outside algorithm (preorder).

**Returns:** `(D, logNormD)`:
- `D`: `(R, C, A)` rescaled outside vectors
- `logNormD`: `(R, C)` log-normalizers

### `compute_J(eigenvalues, distances)`

$J$ interaction matrix.

**Returns:** `(R, A, A)` float64.

### `eigenbasis_project(U, D, model)`

Project inside/outside vectors into eigenbasis.

**Returns:** `(U_tilde, D_tilde)` — each `(R, C, A)`.

### `accumulate_C(D_tilde, U_tilde, J, logNormU, logNormD, logLike, parentIndex)`

Sum eigenbasis contributions over non-root branches.

**Returns:** `(A, A, C)`.

### `back_transform(C, model)`

Transform to natural-basis counts.

**Returns:** `(A, A, C)`.

### `f81_counts(U, D, logNormU, logNormD, logLike, distances, pi, parentIndex)`

F81/JC fast path for counts.

**Returns:** `(A, A, C)`.

### `mixture_posterior(log_likes, log_weights)`

Numerically stable softmax.

**Returns:** `(K, C)`.
