# JAX API Reference

The JAX implementation provides GPU-accelerated, differentiable phylogenetic computations at f64 precision. All operations support batched models via optional leading `*H` dimensions.

## Types

### `Tree`

```python
from subby.jax.types import Tree

tree = Tree(
    parentIndex=jnp.array([-1, 0, 0, 1, 1], dtype=jnp.int32),
    distanceToParent=jnp.array([0.0, 0.1, 0.2, 0.15, 0.25]),
)
```

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `parentIndex` | int32 | `(R,)` | Preorder parent indices; `parentIndex[0] = -1` |
| `distanceToParent` | float | `(R,)` | Branch lengths |

### `DiagModel`

```python
from subby.jax.types import DiagModel

model = DiagModel(
    eigenvalues=jnp.array([0.0, -1.333, -1.333, -1.333]),
    eigenvectors=V,  # (A, A) orthogonal matrix
    pi=jnp.array([0.25, 0.25, 0.25, 0.25]),
)
```

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `eigenvalues` | float | `(*H, A)` | Eigenvalues of symmetrized rate matrix |
| `eigenvectors` | float | `(*H, A, A)` | `v[a,k]` = component $a$ of eigenvector $k$ |
| `pi` | float | `(*H, A)` | Equilibrium distribution |

### `IrrevDiagModel`

For irreversible rate matrices (non-symmetric, complex eigendecomposition).

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `eigenvalues` | complex128 | `(*H, A)` | Complex eigenvalues of rate matrix |
| `eigenvectors` | complex128 | `(*H, A, A)` | Right eigenvectors $V$ |
| `eigenvectors_inv` | complex128 | `(*H, A, A)` | $V^{-1}$ |
| `pi` | float | `(*H, A)` | Stationary distribution |

### `RateModel`

```python
from subby.jax.types import RateModel
```

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `subRate` | float | `(*H, A, A)` | Rate matrix $Q$ |
| `rootProb` | float | `(*H, A)` | Equilibrium distribution |

Automatically diagonalized when passed to high-level functions. Reversibility is auto-detected.

---

## High-level API

### `LogLike(alignment, tree, model, maxChunkSize=128)`

Compute per-column log-likelihoods via Felsenstein pruning.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `Tree` | Phylogenetic tree |
| `model` | `DiagModel`, `RateModel`, list, or grid | Substitution model (see below) |
| `maxChunkSize` | `int` | Column chunk size for memory control |

**Returns:** `(*H, C)` float array of log-likelihoods.

**Model parameter forms:**

| Form | Description |
|------|-------------|
| Single model | Same model for all columns and branches |
| `[m_0, m_1, ..., m_{C-1}]` | Per-column: each column uses its own model |
| `[[m_0, m_1, ..., m_{R-1}]]` | (1, R) per-row: each branch uses its own model, broadcast to all columns |
| `[[...], [...], ...]` (C lists of R) | (C, R) grid: different model at every column and branch |

The (1, R) form broadcasts the same per-branch configuration to all columns. The (C, R) form allows fully independent models at every (column, branch) pair.

When `model` is a list of C models, each column uses its own substitution model (per-column substitution matrices). This enables position-specific rates, e.g., from a neural network predicting rates per site.

**Example — CNN-predicted per-column rates:**

A 1D convolutional network reads the one-hot-encoded leaf sequences of an MSA and predicts per-column rate multipliers for a Jukes-Cantor model. Gradients flow from `LogLike` through the per-column model list back into the CNN parameters.

```python
import jax
import jax.numpy as jnp
from subby.jax import LogLike
from subby.jax.types import Tree
from subby.jax.models import jukes_cantor_model, scale_model

# --- 1D CNN: leaf one-hots -> per-column rates ---
def conv1d(x, w, b):
    out = jax.lax.conv_general_dilated(
        x[None, ...], w, window_strides=(1,), padding="SAME")[0]
    return out + b[:, None]

def predict_rates(params, leaf_one_hot):
    """(n_leaves * A, C) -> (C,) positive rates."""
    h = jax.nn.relu(conv1d(leaf_one_hot, params["w1"], params["b1"]))
    h = jax.nn.relu(conv1d(h, params["w2"], params["b2"]))
    h = conv1d(h, params["w3"], params["b3"])  # (1, C)
    return jax.nn.softplus(h[0])

# --- Loss: negative log-likelihood with per-column rates ---
def loss_fn(params, alignment, tree, base_model, leaf_idx):
    A = base_model.pi.shape[0]
    leaves = alignment[leaf_idx]                         # (n_leaves, C)
    oh = jax.nn.one_hot(leaves, A)                       # (n_leaves, C, A)
    oh_input = oh.transpose(0, 2, 1).reshape(-1, alignment.shape[1])
    rates = predict_rates(params, oh_input)              # (C,)
    models = [scale_model(base_model, rates[c])
              for c in range(rates.shape[0])]
    return -jnp.sum(LogLike(alignment, tree, models))

# --- Training loop (SGD, overfitting one example) ---
grad_fn = jax.grad(loss_fn)
for step in range(500):
    grads = grad_fn(params, alignment, tree, base_model, leaf_idx)
    params = jax.tree.map(lambda p, g: p - 1e-3 * g, params, grads)
```

See [`examples/conv_rate_prediction.py`](https://github.com/ihh/subby/blob/main/examples/conv_rate_prediction.py) for a complete runnable script that simulates an alignment with slow/fast columns and trains the CNN to recover the rate pattern.

**Example — fitting per-branch transition/transversion ratio:**

Each branch of the tree can have its own kappa (transition/transversion ratio) while sharing the same equilibrium frequencies. Wrap R models in a single-element list `[[m_0, ..., m_{R-1}]]` to create a (1, R) per-row grid.

```python
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from subby.jax import LogLike, BranchCounts
from subby.jax.types import Tree
from subby.jax.models import hky85_diag

# Star tree: 4 leaves, all branches directly from root
tree = Tree(
    parentIndex=jnp.array([-1, 0, 0, 0, 0], dtype=jnp.int32),
    distanceToParent=jnp.array([0.0, 0.15, 0.25, 0.1, 0.2]),
)
R, A = 5, 4
pi = jnp.array([0.3, 0.2, 0.25, 0.25])

# --- Simulate 10000 columns with known per-branch kappas ---
true_kappas = [2.0, 1.5, 4.0, 0.8, 3.0]
C = 10000
key = jax.random.PRNGKey(42)

# Substitution matrix M(t) = exp(Q*t) from eigendecomposition
def sub_matrix(model, t):
    V, mu = model.eigenvectors, model.eigenvalues
    S = jnp.einsum('ak,k,bk->ab', V, jnp.exp(mu * t), V)
    sp = jnp.sqrt(model.pi)
    return S * (1.0 / sp)[:, None] * sp[None, :]

true_models = [hky85_diag(k, pi) for k in true_kappas]
Ms = [sub_matrix(true_models[r], float(tree.distanceToParent[r]))
      for r in range(R)]

# Propagate states root → leaves
key, k1 = jax.random.split(key)
states = [jax.random.categorical(k1, jnp.log(pi), shape=(C,))]
for n in range(1, R):
    key, k = jax.random.split(key)
    states.append(jax.random.categorical(
        k, jnp.log(jnp.clip(Ms[n], 1e-30))[states[0]], axis=-1,
    ))

# Alignment: leaves observed, root unobserved (token A)
alignment = jnp.stack([jnp.full(C, A, dtype=jnp.int32)] +
                       [states[n] for n in range(1, R)])

# --- Fit kappa per branch by maximum likelihood ---
def neg_ll(log_kappas):
    models = [hky85_diag(float(np.exp(lk)), pi) for lk in log_kappas]
    return -float(jnp.sum(LogLike(alignment, tree, [models])))

result = minimize(neg_ll, x0=np.log(2.0) * np.ones(R), method='Nelder-Mead')
fitted_kappas = np.exp(result.x)

# Branch 0 (root, t=0) is unidentifiable; leaf branches recover true values
for r in range(1, R):
    print(f"Branch {r}: true κ={true_kappas[r]:.1f}, fitted κ={fitted_kappas[r]:.2f}")
# Branch 1: true κ=1.5, fitted κ≈1.36
# Branch 2: true κ=4.0, fitted κ≈3.75
# Branch 3: true κ=0.8, fitted κ≈0.88
# Branch 4: true κ=3.0, fitted κ≈2.74

# Per-branch substitution counts at the fitted values
models_row = [hky85_diag(k, pi) for k in fitted_kappas]
bc = BranchCounts(alignment, tree, [models_row])  # (R, 4, 4, C)
```

### `LogLikeCustomGrad(alignment, tree, model, maxChunkSize=128)`

Like `LogLike` but with a custom VJP for faster distance gradients.

Uses the Fisher identity: the gradient of log-likelihood w.r.t. branch lengths equals a contraction of expected substitution counts, computed via the downward pass and eigenbasis projection without tracing through the full computation graph. The forward pass is identical to `LogLike`; only the backward pass differs.

**Parameters:** Same as `LogLike` (single model only, not per-column).

**Returns:** `(*H, C)` float array of log-likelihoods.

**Example:**

```python
import jax
from subby.jax import LogLikeCustomGrad
from subby.jax.types import Tree

def loss(distances):
    tree = Tree(parentIndex=parent_idx, distanceToParent=distances)
    return jnp.sum(LogLikeCustomGrad(alignment, tree, model))

# Gradient via Fisher identity (faster than autograd)
grad = jax.grad(loss)(distances)
```

### `Counts(alignment, tree, model, maxChunkSize=128, f81_fast_flag=False)`

Compute expected substitution counts and dwell times per column.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `Tree` | Phylogenetic tree |
| `model` | model, list, or grid | Substitution model (same forms as `LogLike`) |
| `maxChunkSize` | `int` | Column chunk size |
| `f81_fast_flag` | `bool` | Use $O(CRA^2)$ fast path (F81/JC only; not with per-row models) |

**Returns:** `(*H, A, A, C)` float tensor. Diagonal entries are dwell times $E[w_i(c)]$; off-diagonal entries are substitution counts $E[s_{ij}(c)]$.

### `BranchCounts(alignment, tree, model, maxChunkSize=128, f81_fast_flag=False)`

Compute per-branch expected substitution counts and dwell times per column. Returns the same quantities as `Counts` but broken down per branch rather than summed.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `Tree` | Phylogenetic tree |
| `model` | model, list, or grid | Substitution model (same forms as `LogLike`) |
| `maxChunkSize` | `int` | Column chunk size |
| `f81_fast_flag` | `bool` | Use $O(CRA^2)$ fast path (F81/JC only; not with per-row models) |

**Returns:** `(*H, R, A, A, C)` float tensor. Branch 0 (root) is zeros. Diagonal entries are dwell times; off-diagonal entries are substitution counts. Summing over the `R` axis recovers `Counts`.

### `ExpectedCounts(model, t)`

Expected substitution counts and dwell times for a single CTMC branch, independent of any alignment or tree.

Computes $\mathbb{E}[N_{i \to j}(t) \mid X(0)=a, X(t)=b]$ (off-diagonal) and $\mathbb{E}[T_i(t) \mid X(0)=a, X(t)=b]$ (diagonal) for all $(a, b, i, j)$.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `model` | `DiagModel`, `IrrevDiagModel`, or `RateModel` | Substitution model |
| `t` | `float` | Branch length |

**Returns:** `(*H, A, A, A, A)` float tensor. `result[..., a, b, i, j]` is the expected number of $i \to j$ substitutions (off-diagonal) or the expected dwell time in state $i$ (diagonal), conditioned on endpoint states $a$ and $b$.

**Properties:**
- Dwell times sum to $t$: $\sum_i \text{result}[a, b, i, i] = t$ for every reachable $(a, b)$.
- All entries are non-negative.
- At $t = 0$, all entries are zero.

#### `expected_counts_eigen(eigenvalues, eigenvectors, pi, t)`

Inner function for reversible models. Takes pre-computed eigendecomposition, so it can be called repeatedly for different $t$ without re-diagonalizing.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `eigenvalues` | float | `(*H, A)` | Eigenvalues of symmetrized rate matrix |
| `eigenvectors` | float | `(*H, A, A)` | Orthogonal eigenvectors |
| `pi` | float | `(*H, A)` | Equilibrium distribution |
| `t` | float | scalar | Branch length |

#### `expected_counts_eigen_irrev(eigenvalues, eigenvectors, eigenvectors_inv, pi, t)`

Inner function for irreversible models. Takes pre-computed complex eigendecomposition.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `eigenvalues` | complex128 | `(*H, A)` | Complex eigenvalues |
| `eigenvectors` | complex128 | `(*H, A, A)` | Right eigenvectors $V$ |
| `eigenvectors_inv` | complex128 | `(*H, A, A)` | $V^{-1}$ |
| `pi` | float | `(*H, A)` | Stationary distribution |
| `t` | float | scalar | Branch length |

### `RootProb(alignment, tree, model, maxChunkSize=128)`

Compute posterior root state distribution per column.

$$q_a(c) = \frac{\pi_a \cdot U^{(0)}_a(c)}{P(x_c)}$$

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `Tree` | Phylogenetic tree |
| `model` | model, list, or grid | Substitution model (same forms as `LogLike`) |
| `maxChunkSize` | `int` | Column chunk size |

**Returns:** `(*H, A, C)` float array. Sums to 1 over the $A$ dimension for each column.

### `MixturePosterior(alignment, tree, models, log_weights, maxChunkSize=128)`

Compute posterior over mixture components per column.

$$P(k \mid x_c) = \text{softmax}_k(\log P(x_c \mid k) + \log w_k)$$

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `Tree` | Phylogenetic tree |
| `models` | `list[DiagModel]` | $K$ substitution models (e.g., rate-scaled) |
| `log_weights` | `float (K,)` | Log prior weights |
| `maxChunkSize` | `int` | Column chunk size |

**Returns:** `(K, C)` float array of posterior probabilities.

---

## Model constructors

### `hky85_diag(kappa, pi)`

HKY85 model with closed-form eigendecomposition.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kappa` | scalar | Transition/transversion ratio |
| `pi` | `float (4,)` | Equilibrium frequencies $[\pi_A, \pi_C, \pi_G, \pi_T]$ |

**Returns:** `DiagModel` with 4 distinct eigenvalues.

### `jukes_cantor_model(A)`

Jukes-Cantor model for an $A$-state alphabet. Equal rates, uniform equilibrium.

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | int | Alphabet size |

**Returns:** `DiagModel` with eigenvalues $\mu_0 = 0$, $\mu_k = -A/(A-1)$ for $k \geq 1$.

### `f81_model(pi)`

F81 model: $R_{ij} = \mu \cdot \pi_j$ for $i \neq j$, normalized to expected rate 1.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pi` | `float (A,)` | Equilibrium frequencies |

**Returns:** `DiagModel` with eigenvalues $\mu_0 = 0$, $\mu_k = -\mu$ for $k \geq 1$.

### `gamma_rate_categories(alpha, K)`

Yang (1994) discretized gamma rate categories using quantile medians.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | scalar | Shape parameter (lower = more rate variation) |
| `K` | int | Number of categories |

**Returns:** `(rates, weights)` — each `(K,)`. Rates are mean-normalized; weights are uniform $1/K$.

### `gy94_model(omega, kappa, pi=None)`

Goldman-Yang (1994) codon substitution model. Operates on 61 sense codons.

Rate matrix:
- $Q_{ij} = 0$ if codons differ at more than 1 nucleotide position
- $Q_{ij} = \pi_j \cdot \kappa^{\mathbb{1}[\text{transition}]} \cdot \omega^{\mathbb{1}[\text{nonsynonymous}]}$
- Diagonal: $Q_{ii} = -\sum_{j \neq i} Q_{ij}$
- Normalized so $-\sum_i \pi_i Q_{ii} = 1$

| Parameter | Type | Description |
|-----------|------|-------------|
| `omega` | scalar | dN/dS ratio (Ka/Ks) |
| `kappa` | scalar | Transition/transversion ratio |
| `pi` | `float (61,)` or `None` | Codon equilibrium frequencies (default: uniform $1/61$) |

**Returns:** `DiagModel` with $A = 61$ states.

```python
from subby.jax.models import gy94_model
model = gy94_model(omega=0.5, kappa=2.0)
# DiagModel with 61 sense codons
```

### `scale_model(model, rate_multiplier)`

Scale eigenvalues by a rate multiplier. If `rate_multiplier` is `(K,)`, adds $K$ as a leading batch dimension.

---

## Preset models

### `cherryml_siteRM()`

Load the CherryML 400x400 site-pair coevolution model (Prillo et al., Nature Methods 2023). Returns a `DiagModel` with $A = 400$ states representing pairs of amino acids at structurally contacting sites.

State ordering: pair $(i, j) \to i \cdot 20 + j$ using the ARNDCQEGHILKMFPSTWYV alphabet.

**Returns:** `DiagModel` with $A = 400$ states.

```python
from subby.jax.presets import cherryml_siteRM
model_400 = cherryml_siteRM()
# model_400.pi has shape (400,)
```

---

## Format utilities

### `genetic_code()`

Return the standard genetic code as a structured dict. Codons are in ACGT lexicographic order (AAA, AAC, AAG, ..., TTT). Stop codons (TAA=48, TAG=50, TGA=56) are marked with `'*'`.

**Returns:** dict with:

| Key | Type | Description |
|-----|------|-------------|
| `codons` | `list[str]` | 64 codon strings |
| `amino_acids` | `list[str]` | 64 amino acid letters (stop = `'*'`) |
| `sense_mask` | `(64,) bool` | True for sense codons |
| `sense_indices` | `(61,) int` | Indices of sense codons in 0..63 |
| `codon_to_sense` | `(64,) int` | Maps codon index to sense index (stop -> -1) |
| `sense_codons` | `list[str]` | 61 sense codon strings |
| `sense_amino_acids` | `list[str]` | 61 amino acid letters |

```python
from subby.formats import genetic_code
gc = genetic_code()
print(gc['sense_codons'][:5])  # ['AAA', 'AAC', 'AAG', 'AAT', 'ACA']
```

### `codon_to_sense(alignment, A=64)`

Remap a 64-codon tokenized alignment to 61-sense-codon tokens. Stop codons become the gap token. Unobserved and gap tokens are remapped to the new alphabet size.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `int32 (N, C)` | Tokens 0..63 for codons, 64 for ungapped-unobserved, 65 for gap |
| `A` | `int` | Input codon alphabet size (default 64) |

**Returns:** dict with `alignment` (`int32 (N, C)` with $A_\text{sense} = 61$), `A_sense` (61), `alphabet` (list of 61 sense codon strings).

### `split_paired_columns(alignment, paired_columns, A=20)`

Split an alignment into paired and single-column alignments. For coevolution models that operate on pairs of columns (e.g., CherryML SiteRM with $A = 400 = 20 \times 20$ amino acid pairs). Internally uses `kmer_tokenize` with k=2 column tuples for pairs and k=1 for singles.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alignment` | `int32 (N, C)` | Token-encoded alignment with $A$-state tokens |
| `paired_columns` | `list[(int, int)]` | List of `(col_i, col_j)` tuples |
| `A` | `int` | Single-column alphabet size (default 20 for amino acids) |

**Returns:** dict with:

| Key | Type | Description |
|-----|------|-------------|
| `paired_alignment` | `int32 (N, P)` | $A_\text{paired} = A \times A$ states |
| `singles_alignment` | `int32 (N, S)` | $A_\text{singles} = A$ states |
| `paired_columns` | `list[(int, int)]` | Echoed back |
| `single_columns` | `list[int]` | Columns not in any pair |
| `A_paired` | `int` | $A \times A$ |
| `A_singles` | `int` | $A$ |
| `paired_index` | `KmerIndex` | Tuple ↔ index mapping for paired columns |
| `singles_index` | `KmerIndex` | Tuple ↔ index mapping for single columns |

### `merge_paired_columns(paired_posterior, singles_posterior, split_info)`

Reassemble per-column posteriors from paired and single results. Marginalizes the $A_\text{paired} = A \times A$ dimensional paired posteriors into two $A$-dimensional single-column posteriors, then reassembles into the original column order.

| Parameter | Type | Description |
|-----------|------|-------------|
| `paired_posterior` | `float (A_paired, P)` | Posterior for paired columns |
| `singles_posterior` | `float (A_singles, S)` | Posterior for single columns |
| `split_info` | `dict` | Output from `split_paired_columns` |

**Returns:** `(A, C)` array — posterior for all columns in original order.

### K-mer tokenization

See the [Oracle API reference](oracle.md#k-mer-tokenization) for full documentation of `KmerIndex`, `sliding_windows`, `all_column_ktuples`, and `kmer_tokenize`. These functions are implemented in `subby.formats` and re-exported from both `subby.jax` and `subby.oracle`.

```python
from subby.formats import kmer_tokenize, sliding_windows, all_column_ktuples, KmerIndex
```

---

## Padding utilities

### `pad_alignment(alignment, bin_size=128)`

Pad alignment columns to the next multiple of `bin_size` with gap tokens (`-1`). Gap-padded columns are mathematically neutral (logL = 0, zero counts, root prior unchanged), so padding avoids JAX recompilation when `C` varies across inputs.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `bin_size` | `int` | Round `C` up to the next multiple of this (default 128) |

**Returns:** `(padded_alignment, C_original)` — the padded `(R, C_padded)` alignment and the original column count.

### `unpad_columns(result, C_original)`

Strip padding columns from a result array: `result[..., :C_original]`.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `result` | array `(..., C_padded)` | Output from a padded computation |
| `C_original` | `int` | Original column count from `pad_alignment` |

**Returns:** `(..., C_original)` array.

**Example — JIT-friendly binning:**

```python
from subby.jax import LogLike, pad_alignment, unpad_columns

padded, C_orig = pad_alignment(alignment, bin_size=64)
ll = LogLike(padded, tree, model)       # shape reused across similar C
ll = unpad_columns(ll, C_orig)          # back to original C
```

---

## InsideOutside

### `InsideOutside(alignment, tree, model, maxChunkSize=128)`

Runs the inside (upward) and outside (downward) passes once and stores the resulting DP tables, enabling efficient queries for log-likelihoods, expected substitution counts, node state posteriors, and branch endpoint joint posteriors without recomputation.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `Tree` | Phylogenetic tree |
| `model` | `DiagModel`, `IrrevDiagModel`, `RateModel`, or `list` | Substitution model |
| `maxChunkSize` | `int` | Column chunk size |

**Properties:**

| Name | Type | Description |
|------|------|-------------|
| `log_likelihood` | `(*H, C)` float | Per-column log-likelihoods |

**Methods:**

#### `counts(f81_fast_flag=False, branch_mask="auto")`

Expected substitution counts and dwell times, reusing stored DP tables.

**Returns:** `(*H, A, A, C)` float tensor.

#### `branch_counts(f81_fast_flag=False)`

Per-branch expected substitution counts and dwell times, reusing stored DP tables.

**Returns:** `(*H, R, A, A, C)` float tensor. Branch 0 (root) is zeros.

#### `node_posterior(node=None)`

Posterior state distribution at node(s).

For root: $P(X_0 = a \mid \text{data}) \propto \pi_a \cdot U^{(0)}_a(c)$

For non-root: $P(X_n = j \mid \text{data}) \propto \left[\sum_a D^{(n)}_a(c) \cdot M^{(n)}_{aj}\right] \cdot U^{(n)}_j(c)$

| Argument | Type | Description |
|----------|------|-------------|
| `node` | `int` or `None` | Node index, or `None` for all nodes |

**Returns:** `(*H, A, C)` if `node` is int; `(*H, R, A, C)` if `None`.

#### `branch_posterior(node=None)`

Joint posterior of parent-child states on a branch.

$$P(X_{\text{parent}(n)}=i,\, X_n=j \mid \text{data}, c) \propto D^{(n)}_i(c) \cdot M^{(n)}_{ij} \cdot U^{(n)}_j(c)$$

| Argument | Type | Description |
|----------|------|-------------|
| `node` | `int` or `None` | Child node index (must be > 0), or `None` for all |

**Returns:** `(*H, A, A, C)` if `node` is int; `(*H, R, A, A, C)` if `None`. Branch 0 is zeros.

**Example:**

```python
from subby.jax import InsideOutside

io = InsideOutside(alignment, tree, model)

ll = io.log_likelihood                  # (*H, C)
root_post = io.node_posterior(0)        # (*H, A, C)
all_posts = io.node_posterior()         # (*H, R, A, C)
branch_joint = io.branch_posterior(3)   # (*H, A, A, C)
counts = io.counts()                    # (*H, A, A, C)
per_branch = io.branch_counts()         # (*H, R, A, A, C)
```

---

## Low-level functions

### `diagonalize_rate_matrix(subRate, rootProb)`

Convert a `RateModel` to `DiagModel` via eigendecomposition of the symmetrized rate matrix.

### `compute_sub_matrices(model, distanceToParent)`

Compute transition probability matrices $M_{ij}(t_n)$ for each branch.

**Returns:** `(*H, R, A, A)` — rows sum to 1, $M(0) = I$.

### `upward_pass(alignment, tree, subMatrices, rootProb, maxChunkSize, per_column=False)`

Felsenstein pruning (postorder, leaves to root) via `jax.lax.scan`.

When `per_column=True`, `subMatrices` has shape `(*H, R, C, A, A)` — a different substitution matrix per column. Default: `(*H, R, A, A)`.

**Returns:** `(U, logNormU, logLike)` where:
- `U`: `(*H, R, C, A)` rescaled inside vectors
- `logNormU`: `(*H, R, C)` log-normalizers
- `logLike`: `(*H, C)` per-column log-likelihoods

### `downward_pass(U, logNormU, tree, subMatrices, rootProb, alignment, per_column=False)`

Outside algorithm (preorder, root to leaves) via `jax.lax.scan`.

When `per_column=True`, `subMatrices` has shape `(*H, R, C, A, A)`. Default: `(*H, R, A, A)`.

**Returns:** `(D, logNormD)` where:
- `D`: `(*H, R, C, A)` rescaled outside vectors
- `logNormD`: `(*H, R, C)` log-normalizers

### `compute_J(eigenvalues, distanceToParent)`

$J$ interaction matrix for eigensubstitution accumulation.

$$J_{kl}(t) = \begin{cases} t \cdot e^{\mu_k t} & \text{if } \mu_k \approx \mu_l \\ \frac{e^{\mu_k t} - e^{\mu_l t}}{\mu_k - \mu_l} & \text{otherwise} \end{cases}$$

**Returns:** `(*H, R, A, A)`.

### `eigenbasis_project(U, D, model)`

Project inside/outside vectors into the eigenbasis.

**Returns:** `(U_tilde, D_tilde)` — each `(*H, R, C, A)`.

### `accumulate_C(D_tilde, U_tilde, J, logNormU, logNormD, logLike, parentIndex)`

Sum eigenbasis contributions over all non-root branches.

**Returns:** `(*H, A, A, C)` eigenbasis counts.

### `back_transform(C, model)`

Transform eigenbasis counts to natural-basis substitution counts and dwell times.

**Returns:** `(*H, A, A, C)`.

### `accumulate_C_per_branch(...)` / `back_transform_per_branch(...)`

Per-branch variants of `accumulate_C` and `back_transform`. Instead of summing over branches, each branch's contribution is stored separately.

**Returns:** `(*H, R, A, A, C)`.

### `f81_counts(U, D, logNormU, logNormD, logLike, distances, pi, parentIndex)`

$O(CRA^2)$ direct computation for F81/JC models, avoiding the eigenbasis.

**Returns:** `(*H, A, A, C)`.

### `f81_counts_per_branch(U, D, logNormU, logNormD, logLike, distances, pi, parentIndex)`

Per-branch variant of `f81_counts`.

**Returns:** `(*H, R, A, A, C)`.

### `mixture_posterior(log_likes, log_weights)`

Numerically stable softmax over mixture components.

**Returns:** `(K, C)`.
