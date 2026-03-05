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
| `model` | `DiagModel`, `RateModel`, or `list` | Substitution model (or list of C models for per-column rates) |
| `maxChunkSize` | `int` | Column chunk size for memory control |

**Returns:** `(*H, C)` float array of log-likelihoods.

When `model` is a list of C models, each column uses its own substitution model (per-column substitution matrices). This enables position-specific rates, e.g., from a neural network predicting rates per site.

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
| `model` | `DiagModel`, `RateModel`, or `list` | Substitution model (or list of C models for per-column rates) |
| `maxChunkSize` | `int` | Column chunk size |
| `f81_fast_flag` | `bool` | Use $O(CRA^2)$ fast path (F81/JC only) |

**Returns:** `(*H, A, A, C)` float tensor. Diagonal entries are dwell times $E[w_i(c)]$; off-diagonal entries are substitution counts $E[s_{ij}(c)]$.

### `RootProb(alignment, tree, model, maxChunkSize=128)`

Compute posterior root state distribution per column.

$$q_a(c) = \frac{\pi_a \cdot U^{(0)}_a(c)}{P(x_c)}$$

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `alignment` | `int32 (R, C)` | Token-encoded alignment |
| `tree` | `Tree` | Phylogenetic tree |
| `model` | `DiagModel`, `RateModel`, or `list` | Substitution model (or list of C models for per-column rates) |
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

### `scale_model(model, rate_multiplier)`

Scale eigenvalues by a rate multiplier. If `rate_multiplier` is `(K,)`, adds $K$ as a leading batch dimension.

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

### `f81_counts(U, D, logNormU, logNormD, logLike, distances, pi, parentIndex)`

$O(CRA^2)$ direct computation for F81/JC models, avoiding the eigenbasis.

**Returns:** `(*H, A, A, C)`.

### `mixture_posterior(log_likes, log_weights)`

Numerically stable softmax over mixture components.

**Returns:** `(K, C)`.
