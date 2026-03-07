# Tutorial: Computing Phylogenetic Sufficient Statistics

This tutorial walks through a complete worked example, from raw inputs to feature vectors suitable for a gene annotation model.

## Setup

The oracle implementation has no dependencies beyond NumPy and SciPy:

```python
import numpy as np
from subby.oracle import (
    LogLike, Counts, RootProb, MixturePosterior,
    hky85_diag, jukes_cantor_model, f81_model,
    gamma_rate_categories, scale_model,
    compute_sub_matrices, upward_pass, downward_pass,
    compute_J, eigenbasis_project, accumulate_C, back_transform,
    compute_branch_mask, children_of,
    parse_newick, parse_fasta, parse_dict, combine_tree_alignment,
)
```

## Step 1: Define the tree

The easiest way to define a tree is from a Newick string:

```python
tree_result = parse_newick("((leaf2:0.05,leaf3:0.15):0.1,(leaf4:0.3):0.0):0.0;")
# tree_result['parentIndex']:      [-1, 0, 1, 1, 0, 4]
# tree_result['distanceToParent']: [0.0, 0.1, 0.05, 0.15, 0.0, 0.3]
# tree_result['leaf_names']:       ['leaf2', 'leaf3', 'leaf4']
```

This produces a DFS preorder indexed tree:

```
Tree:         0 (root)
             / \
            1   4
           / \   \
          2   3   5
```

Under the hood, subby represents trees as two arrays. `parentIndex` encodes the topology (each node's parent has a smaller index), and `distanceToParent` gives branch lengths:

```python
tree = {
    'parentIndex': tree_result['parentIndex'],
    'distanceToParent': tree_result['distanceToParent'],
}
```

You can also construct these arrays directly:

```python
parentIndex = np.array([-1, 0, 0, 1, 1], dtype=np.int32)
distanceToParent = np.array([0.0, 0.1, 0.3, 0.05, 0.15])
tree = {'parentIndex': parentIndex, 'distanceToParent': distanceToParent}
```

We can inspect the tree structure:

```python
left_child, right_child, sibling = children_of(tree['parentIndex'])
# left_child:  [1, 3, -1, -1, -1]  (node 0's left child is 1, etc.)
# right_child: [2, 4, -1, -1, -1]
# sibling:     [-1, 2, 1, 4, 3]    (node 1's sibling is 2, etc.)
```

## Step 2: Define the alignment

The easiest way to provide an alignment is from FASTA format:

```python
fasta_text = """\
>leaf2
CACGAA
>leaf3
ACTGCA
>leaf4
ACGAAA
"""
aln_result = parse_fasta(fasta_text)
# aln_result['alignment']: (3, 6) int32 array of leaf sequences
# aln_result['alphabet']:  ['A', 'C', 'G', 'T']
```

To combine a tree and alignment (mapping leaves by name), use `combine_tree_alignment`:

```python
tree_result = parse_newick("((leaf2:0.05,leaf3:0.15):0.1,leaf4:0.3);")
combined = combine_tree_alignment(tree_result, aln_result)
alignment = combined['alignment']  # (R, C) int32 — full tree, internal nodes filled
tree = {'parentIndex': combined['parentIndex'],
        'distanceToParent': combined['distanceToParent']}
```

Internal nodes are automatically filled with the ungapped-unobserved token (`A = len(alphabet)`).

Alternatively, you can supply sequences as a dictionary:

```python
aln_result = parse_dict({
    "leaf2": "CACGAA",
    "leaf3": "ACTGCA",
    "leaf4": "ACGAAA",
})
```

<details>
<summary>Advanced: constructing the alignment tensor manually</summary>

Under the hood, the alignment is an `(R, C)` int32 array with integer tokens. For a DNA alphabet (A=0, C=1, G=2, T=3):

```python
alignment = np.array([
    [0, 1, 2, 3, 0, -1],   # root (internal — typically unobserved)
    [0, 1, 2, 3, 0,  4],   # internal node 1 (token 4 = ungapped-unobserved)
    [1, 0, 2, 3, 0,  0],   # leaf 2
    [0, 1, 3, 3, 1,  0],   # leaf 3
    [0, 1, 2, 2, 0,  0],   # leaf 4
], dtype=np.int32)
```

Token encoding:

- `0–3`: observed nucleotide (maps to alphabet position)
- `4` (= `A`, the alphabet size): ungapped but unobserved (uniform likelihood)
- `-1` or `A+1`: gapped (uniform likelihood)

The parsers handle this conversion automatically — most users do not need to work with integer tokens directly.
</details>

## Step 3: Choose a substitution model

### Jukes-Cantor (simplest)

Equal substitution rates between all states:

```python
model_jc = jukes_cantor_model(4)
# eigenvalues: [0, -4/3, -4/3, -4/3]
# pi: [0.25, 0.25, 0.25, 0.25]
```

### HKY85 (nucleotide-specific)

Transition/transversion bias with non-uniform base frequencies:

```python
kappa = 2.0  # transition/transversion ratio
pi = np.array([0.3, 0.2, 0.25, 0.25])  # equilibrium frequencies
model_hky = hky85_diag(kappa, pi)
# 4 distinct eigenvalues, closed-form eigenvectors
```

### F81 (intermediate)

Non-uniform frequencies but equal substitution rates:

```python
model_f81 = f81_model(np.array([0.3, 0.2, 0.25, 0.25]))
```

## Step 4: Compute log-likelihoods

```python
ll = LogLike(alignment, tree, model_hky)
print(ll)
# Array of 6 log-likelihoods, one per column
# All values ≤ 0 (log-probabilities)
```

Columns where all leaves agree will have higher (less negative) log-likelihoods. Column 5 (with gaps) may have a log-likelihood of 0 if all leaves are gapped.

## Step 5: Compute substitution counts and dwell times

```python
counts = Counts(alignment, tree, model_hky)
# Shape: (4, 4, 6) — one 4×4 matrix per column

# For column 0:
col0 = counts[:, :, 0]
# Diagonal: dwell times E[w_A], E[w_C], E[w_G], E[w_T]
dwell_times = np.diag(col0)
# Off-diagonal: substitution counts E[s_AC], E[s_AG], ...
```

The counts tensor packs both dwell times (diagonal) and substitution counts (off-diagonal) into a single `(A, A, C)` array.

### F81 fast path

For F81 and Jukes-Cantor models, there's an $O(CRA^2)$ alternative that avoids the eigenbasis:

```python
counts_fast = Counts(alignment, tree, model_jc, f81_fast=True)
# Same result, different algorithm
```

## Step 6: Compute root posterior

```python
root_post = RootProb(alignment, tree, model_hky)
# Shape: (4, 6) — posterior P(root=a | column c)
print(np.sum(root_post, axis=0))  # sums to 1 for each column
```

## Step 7: Rate heterogeneity with mixture models

Real sequences evolve at different rates across sites. Model this with a discretized gamma distribution:

```python
alpha = 0.5  # shape parameter (lower = more rate variation)
K = 4        # number of rate categories

rates, weights = gamma_rate_categories(alpha, K)
print(f"Rates: {rates}")    # e.g., [0.03, 0.26, 0.82, 2.89]
print(f"Weights: {weights}") # [0.25, 0.25, 0.25, 0.25]

# Create K scaled models
models = [scale_model(model_hky, r) for r in rates]
log_weights = np.log(weights)

# Compute posterior over rate categories
posteriors = MixturePosterior(alignment, tree, models, log_weights)
# Shape: (4, 6) — P(rate category k | column c)
print(np.sum(posteriors, axis=0))  # sums to 1
```

Fast-evolving columns will have higher posterior weight on larger rate categories.

## Step 8: Branch masking

Not all branches are informative for every column. The Steiner tree mask identifies which branches connect ungapped leaves:

```python
mask = compute_branch_mask(alignment, tree['parentIndex'], A=4)
# Shape: (5, 6) — boolean per branch per column

# Column 5 has a gap at root and unobserved at node 1:
print(mask[:, 5])
# Branches connecting ungapped leaves (2, 3, 4) are active;
# root branch is always inactive (mask[0] = False)
```

## Step 9: Assemble feature vectors

For gene annotation, concatenate features from multiple tokenization schemes:

```python
def compute_features(alignment, tree, model, K=4, alpha=0.5):
    """Compute feature vector for one tokenization scheme."""
    A = len(model['pi'])
    C = alignment.shape[1]

    # Rate-heterogeneous counts
    rates, weights = gamma_rate_categories(alpha, K)
    models = [scale_model(model, r) for r in rates]
    log_weights = np.log(weights)

    # Average counts across rate categories (weighted by posterior)
    posteriors = MixturePosterior(alignment, tree, models, log_weights)  # (K, C)

    all_counts = []
    for k in range(K):
        c_k = Counts(alignment, tree, models[k])  # (A, A, C)
        all_counts.append(c_k)

    # Posterior-weighted average
    avg_counts = np.zeros((A, A, C))
    for k in range(K):
        avg_counts += posteriors[k:k+1, :][np.newaxis, np.newaxis, :] * all_counts[k]  # broadcast (1,1,C) * (A,A,C)

    # Extract features per column
    features = np.zeros((C, A * (A - 1) + A + K))
    for c in range(C):
        idx = 0
        # Off-diagonal: substitution counts
        for i in range(A):
            for j in range(A):
                if i != j:
                    features[c, idx] = avg_counts[i, j, c]
                    idx += 1
        # Diagonal: dwell times
        for i in range(A):
            features[c, idx] = avg_counts[i, i, c]
            idx += 1
        # Rate category posteriors
        for k in range(K):
            features[c, idx] = posteriors[k, c]
            idx += 1

    return features

# HKY85 features: 12 subs + 4 dwell + 4 posteriors = 20 features per column
hky_features = compute_features(alignment, tree, model_hky)
print(f"HKY85 features shape: {hky_features.shape}")  # (6, 20)
```

## Step 10: Browser deployment

For in-browser inference, the same computation runs on WebGPU or WASM. Format parsers are available in JS for converting standard files to subby's internal arrays:

```javascript
import {
  createPhyloEngine,
  parseNewick, parseFasta, combineTreeAlignment, jukesCantor,
} from './subby/webgpu/index.js';

const { engine, backend } = await createPhyloEngine({
  shaderBasePath: './subby/webgpu/shaders/',
  wasmUrl: './phylo_wasm_bg.wasm',
});
console.log(`Using ${backend} backend`);

// Parse tree and alignment from standard formats
const tree = parseNewick('((A:0.1,B:0.2):0.05,C:0.3);');
const aln = parseFasta('>A\nACGT\n>B\nTGCA\n>C\nGGGG\n');
const combined = combineTreeAlignment(tree, aln);

// Get model parameters
const model = jukesCantor(4);

const logLike = await engine.LogLike(
  combined.alignment, combined.parentIndex, combined.distanceToParent,
  model.eigenvalues, model.eigenvectors, model.pi
);

engine.destroy();
```

You can also supply sequences as a dictionary:

```javascript
const aln = parseDict({ A: 'ACGT', B: 'TGCA', C: 'GGGG' });
```

## Intermediates deep dive

For debugging or advanced use, you can inspect every intermediate quantity:

```python
# Substitution matrices M(t) for each branch
sub_mats = compute_sub_matrices(model_hky, distanceToParent)
# Shape: (5, 4, 4) — M[n] is the 4×4 matrix for branch n

# Verify: rows sum to 1, M(0) = I
print(np.sum(sub_mats[1], axis=1))  # [1, 1, 1, 1]

# Inside vectors and log-normalizers
U, logNormU, logLike = upward_pass(alignment, tree, sub_mats, model_hky['pi'])
# U: (5, 6, 4) — rescaled inside vector per node per column
# logNormU: (5, 6) — log-normalizer per node per column

# Outside vectors
D, logNormD = downward_pass(U, logNormU, tree, sub_mats, model_hky['pi'], alignment)
# D: (5, 6, 4) — rescaled outside vector per node per column

# Verify sum-product consistency for any non-root branch n, column c:
# sum_{a,b} D[n,c,a] * M[n,a,b] * U[n,c,b] * exp(logNormD[n,c] + logNormU[n,c])
#   = P(x_c) = exp(logLike[c])
n, c = 1, 0
check = 0.0
for a in range(4):
    for b in range(4):
        check += D[n, c, a] * sub_mats[n, a, b] * U[n, c, b]
check *= np.exp(logNormD[n, c] + logNormU[n, c])
print(f"Sum-product: {check:.6e}, P(x): {np.exp(logLike[c]):.6e}")
# These should be equal (up to numerical precision)
```
