# Oracle API Reference

The oracle is a pure NumPy reference implementation using explicit for-loops. It serves as the numerically trustworthy test oracle for all other backends (atol=1e-8 vs JAX).

**Dependencies:** `numpy`, `scipy` (for `gammainc` only).

## Data representations

### Tree

From a Newick string:

```python
from subby.formats import Tree

tree_result = parse_newick("((A:0.1,B:0.2):0.05,C:0.3);")
tree = Tree(parentIndex=tree_result['parentIndex'],
            distanceToParent=tree_result['distanceToParent'])
```

Or construct directly with numpy arrays:

```python
tree = Tree(
    parentIndex=np.array([-1, 0, 0, 1, 1], dtype=np.int32),
    distanceToParent=np.array([0.0, 0.1, 0.2, 0.15, 0.25]),
)
```

### Model

A plain dict (constructed via model functions below):

```python
model = jukes_cantor_model(4)
# {'eigenvalues': array([0., -1.333, -1.333, -1.333]),
#  'eigenvectors': V,   # (4, 4) orthogonal matrix
#  'pi': array([0.25, 0.25, 0.25, 0.25])}
```

### Alignment

Use parsers to create alignments from standard formats (see [Format parsers](#format-parsers) below):

```python
aln = parse_fasta(">A\nACGT\n>B\nTGCA\n")
aln = parse_dict({"A": "ACGT", "B": "TGCA"})
combined = combine_tree_alignment(tree_result, aln)
```

The resulting `(R, C)` int32 array uses this token encoding:

| Token | Meaning | Likelihood vector |
|-------|---------|-------------------|
| `0` to `A-1` | Observed state (alphabet position) | One-hot |
| `A` | Ungapped, unobserved (internal nodes) | All ones |
| `A+1` or `-1` | Gapped (`-` or `.`) | All ones |

Alphabet detection is automatic: DNA (`ACGT`), RNA (`ACGU`), protein (20 AAs), or sorted unique characters. Override with the `alphabet` parameter on any parser.

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
| `tree` | `Tree` | `Tree(parentIndex=..., distanceToParent=...)` |
| `model` | `dict` | `{'eigenvalues': ..., 'eigenvectors': ..., 'pi': ...}` |

**Returns:** `(C,)` float64 array.

### `Counts(alignment, tree, model, f81_fast=False)`

Compute expected substitution counts and dwell times.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `Tree` | Tree NamedTuple |
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
| `tree` | `Tree` | Tree NamedTuple |
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

### `gy94_model(omega, kappa, pi=None)`

Goldman-Yang (1994) codon substitution model. Operates on 61 sense codons.

Rate matrix:
- $Q_{ij} = 0$ if codons differ at more than 1 nucleotide position
- $Q_{ij} = \pi_j \cdot \kappa^{\mathbb{1}[\text{transition}]} \cdot \omega^{\mathbb{1}[\text{nonsynonymous}]}$
- Diagonal: $Q_{ii} = -\sum_{j \neq i} Q_{ij}$
- Normalized so $-\sum_i \pi_i Q_{ii} = 1$

This is reversible ($\pi_i Q_{ij} = \pi_j Q_{ji}$), so uses symmetric eigendecomposition.

```python
model = gy94_model(omega=0.5, kappa=2.0)
# {'eigenvalues': ..., 'eigenvectors': ..., 'pi': ..., 'reversible': True}
# A=61 sense codons
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `omega` | scalar | dN/dS ratio (Ka/Ks) |
| `kappa` | scalar | Transition/transversion ratio |
| `pi` | `(61,)` or `None` | Codon equilibrium frequencies (default: uniform $1/61$) |

**Returns:** dict with `eigenvalues`, `eigenvectors`, `pi`, `reversible: True`. $A = 61$ states.

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
from subby.oracle import (
    parse_newick, parse_fasta, parse_stockholm, parse_maf,
    parse_strings, parse_dict, combine_tree_alignment, detect_alphabet,
)
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

### `parse_dict(sequences, alphabet=None) -> dict`

Parse a `{name: sequence}` dictionary into an alignment tensor.

**Returns:** dict with `alignment` (int32 `(N, C)`), `leaf_names`, `alphabet`.

```python
aln = parse_dict({"human": "ACGT", "mouse": "TGCA", "dog": "GGGG"})
```

### `combine_tree_alignment(tree_result, alignment_result) -> CombinedResult`

Map leaf sequences to tree positions by name. Creates full `(R, C)` alignment with internal nodes filled with the ungapped-unobserved token.

**Returns:** `CombinedResult` NamedTuple with `alignment` (int32 `(R, C)`), `tree` (`Tree`), `alphabet`, `leaf_names`.

```python
tree = parse_newick("((A:0.1,B:0.2):0.05,C:0.3);")
aln = parse_fasta(">A\nACGT\n>B\nTGCA\n>C\nGGGG\n")
combined = combine_tree_alignment(tree, aln)
ll = LogLike(combined.alignment, combined.tree, jukes_cantor_model(4))
```

### `genetic_code() -> dict`

Return the standard genetic code. Codons in ACGT lexicographic order; stop codons (TAA=48, TAG=50, TGA=56) marked with `'*'`.

**Returns:** dict with `codons` (64 strings), `amino_acids` (64 letters), `sense_mask` (64 bools), `sense_indices` (61 ints), `codon_to_sense` (64 ints, stop -> -1), `sense_codons` (61 strings), `sense_amino_acids` (61 letters).

```python
from subby.formats import genetic_code
gc = genetic_code()
print(len(gc['sense_codons']))  # 61
```

### `codon_to_sense(alignment, A=64) -> dict`

Remap a 64-codon tokenized alignment to 61-sense-codon tokens. Stop codons become the gap token.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `int32 (N, C)` | Tokens 0..63 for codons, 64 unobserved, 65 gap |
| `A` | `int` | Input codon alphabet size (default 64) |

**Returns:** dict with `alignment` (int32 `(N, C)` with $A_\text{sense} = 61$), `A_sense`, `alphabet`.

---

## K-mer tokenization

K-mer tokenization converts single-character token alignments into multi-column tokens (e.g., codons = 3-mers). The API cleanly separates column indexing from tokenization:

1. **Column indexing**: `sliding_windows` and `all_column_ktuples` produce `(T, k)` arrays of column indices.
2. **Tokenization**: `kmer_tokenize` converts alignment columns into k-mer tokens given column tuples.
3. **Index mapping**: `KmerIndex` provides O(1) lookup between column tuples and output positions.

### `KmerIndex`

Maps between column tuples and output alignment indices. Provides O(1) lookup in both directions.

```python
from subby.oracle import KmerIndex

index = KmerIndex([(0, 1), (2, 3), (4, 5)])
index.tuple_to_idx((2, 3))  # → 1
index.idx_to_tuple(0)       # → (0, 1)
len(index)                  # → 3
```

| Method | Description |
|--------|-------------|
| `tuple_to_idx(t)` | Column tuple → output index. Returns `-1` if absent. |
| `idx_to_tuple(idx)` | Output index → column tuple. |
| `__len__()` | Number of tuples. |

### `sliding_windows(C, k, stride=None, offset=0, edge='truncate')`

Generate column index tuples for sliding-window k-mer tokenization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `C` | `int` | Number of columns in the alignment |
| `k` | `int` | Window size (k-mer length) |
| `stride` | `int` or `None` | Step between window starts. Default `None` → `k` (non-overlapping). |
| `offset` | `int` | Starting column index. Default `0`. |
| `edge` | `str` | `'truncate'` (default): drop incomplete trailing window; `'pad'`: include partial window, using `-1` for out-of-bounds columns |

**Returns:** `(M, k)` int64 array. Entries of `-1` indicate out-of-bounds positions (only with `edge='pad'`).

```python
from subby.oracle import sliding_windows

# Non-overlapping codons
sliding_windows(9, 3)                # → [[0,1,2],[3,4,5],[6,7,8]]

# Overlapping stride-1 windows
sliding_windows(5, 3, stride=1)      # → [[0,1,2],[1,2,3],[2,3,4]]

# Three reading frames
for frame in range(3):
    windows = sliding_windows(9, 3, stride=3, offset=frame)
```

### `all_column_ktuples(C, k, ordered=True)`

Generate all k-tuples of column indices. **WARNING:** produces $O(C^k)$ tuples.

| Parameter | Type | Description |
|-----------|------|-------------|
| `C` | `int` | Number of columns |
| `k` | `int` | Tuple size |
| `ordered` | `bool` | `True`: permutations ($C \cdot (C-1)$ for $k=2$); `False`: combinations ($\binom{C}{k}$) |

**Returns:** `(T, k)` int64 array.

### `kmer_tokenize(alignment, A, k_or_tuples, gap_mode='any', alphabet=None)`

Core tokenizer. Accepts either an integer `k` (backward compatible, non-overlapping windows) or a `(T, k)` array of column index tuples.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `A` | `int` | Single-character alphabet size |
| `k_or_tuples` | `int` or `(T, k)` array | Integer `k` for non-overlapping windows (`C` must be divisible by `k`), or column tuples from `sliding_windows()` / `all_column_ktuples()` / custom |
| `gap_mode` | `str` | `'any'`: gap in any position gaps the k-mer; `'all'`: only all-gap k-mers become gaps |
| `alphabet` | `list[str]` or `None` | Single-character labels for building k-mer labels |

**Returns:** dict with:

| Key | Type | Description |
|-----|------|-------------|
| `alignment` | `int32 (R, T)` | K-mer tokens |
| `A_kmer` | `int` | $A^k$ |
| `index` | `KmerIndex` | Bidirectional tuple ↔ index mapping |
| `alphabet` | `list[str]` | K-mer labels (only if `alphabet` given) |

Token encoding: `0..A^k-1` observed, `A^k` ungapped-unobserved, `A^k+1` gap, `A^k+2` illegal (partial gap with `gap_mode='all'`). Entries of `-1` in column tuples are treated as unobserved positions.

```python
from subby.oracle import kmer_tokenize, sliding_windows

# Non-overlapping codons (backward compatible)
result = kmer_tokenize(dna_alignment, A=4, k_or_tuples=3)

# Overlapping codon windows
windows = sliding_windows(C=100, k=3, stride=1)
result = kmer_tokenize(dna_alignment, A=4, k_or_tuples=windows)
result['index'].tuple_to_idx((5, 6, 7))  # → 5
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
