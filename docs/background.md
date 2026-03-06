# Mathematical Background

This page describes the eigensubstitution accumulation algorithm implemented by the library, following [Holmes & Rubin (2002)](https://doi.org/10.1089/10665270252935467).

## Problem statement

Given:
- A multiple sequence alignment $x$ of $R$ sequences over $C$ columns
- A binary phylogenetic tree $T$ with branch lengths $t_n$
- A continuous-time Markov substitution model with rate matrix $Q$ and equilibrium distribution $\pi$

Compute, for each column $c$:
- **Log-likelihood** $\log P(x_c \mid T, Q)$
- **Expected substitution counts** $E[s_{ij,c}]$ for each state pair $(i,j)$, $i \ne j$
- **Expected dwell times** $E[w_{i,c}]$ for each state $i$
- **Posterior root state** $P(\text{root} = a \mid x_c)$

## Substitution model

### General CTMC

A continuous-time Markov chain on $A$ states is parameterized by:
- **Rate matrix** $Q$ with $Q_{ij} \ge 0$ for $i \ne j$ and $\sum_j Q_{ij} = 0$
- **Equilibrium distribution** $\pi$ satisfying $\pi^T Q = 0$

### Eigendecomposition

The rate matrix is diagonalized as $Q = V \Lambda V^{-1}$ where $\Lambda = \text{diag}(\mu_0, \ldots, \mu_{A-1})$ are eigenvalues and $V$ contains the corresponding right eigenvectors as columns. In general, $\mu_k$ and $V$ may be complex-valued. We always have $\mu_0 = 0$ (corresponding to the stationary distribution), and $\text{Re}(\mu_k) \le 0$ for all $k$.

The **transition probability matrix** is:
$$M_{ab}(t) = \left[ e^{Qt} \right]_{ab} = \sum_k V_{ak} \, e^{\mu_k t} \, V^{-1}_{kb}$$

Note: $V^{-1}$ is the matrix of left eigenvectors (rows). For a general rate matrix, $V^{-1} \ne V^T$; the left and right eigenvectors are biorthogonal ($V^{-1} V = I$) but not individually orthonormal.

### Reversible models (special case)

A model is **reversible** if it satisfies detailed balance: $\pi_i Q_{ij} = \pi_j Q_{ji}$. In this case, the **symmetrized rate matrix**
$$S_{ij} = \sqrt{\pi_i} \, Q_{ij} / \sqrt{\pi_j}$$
is real symmetric. Its eigendecomposition $S = V \Lambda V^T$ yields real eigenvalues $\mu_k$ and an orthogonal eigenvector matrix $V$ (i.e. $V^{-1} = V^T$).

The transition probability matrix simplifies to:
$$M_{ab}(t) = \sqrt{\frac{\pi_b}{\pi_a}} \sum_k V_{ak} \, e^{\mu_k t} \, V_{bk}$$

This is the basis for the symmetrized formulation used throughout the reversible path.

### Supported models

| Model | States | Reversible | Eigendecomposition |
|-------|--------|------------|--------------------|
| **HKY85** | 4 (A,C,G,T) | Yes | Closed-form (4 analytical eigenvectors) |
| **Jukes-Cantor** | $A$ (any) | Yes | $\mu_0 = 0$, $\mu_k = -A/(A-1)$ for $k \ge 1$; QR complement |
| **F81** | $A$ (any) | Yes | $\mu_0 = 0$, $\mu_k = -\mu$ for $k \ge 1$; QR complement |
| **General** | $A$ (any) | No | Full complex eigendecomposition via `eig` |

Auto-detection of reversibility is supported via a detailed balance check with configurable tolerance. A rate matrix satisfying detailed balance is automatically routed to the real symmetric (eigh) path; otherwise the complex (eig) path is used.

### Rate heterogeneity

A discretized gamma distribution (Yang 1994) with shape $\alpha$ and $K$ categories provides rate multipliers $r_k$ at quantile medians, with equal weights $1/K$. Each rate category scales the eigenvalues: $\mu'_k = r_k \cdot \mu_k$.

## Algorithm

Steps 1 and 2 operate on the transition matrices $M(t)$ directly and are identical for reversible and irreversible models. Steps 3 and 4 work in the eigenbasis and differ in how vectors are projected and back-transformed.

### Step 1: Felsenstein pruning (upward/inside pass)

For each node $n$ in postorder (leaves to root), the inside vector $U^{(n)}_a(c)$ gives the likelihood of the data below node $n$ given that $n$ is in state $a$ at column $c$.

**Leaf initialization:**
- $U^{(n)}_a(c) = \delta_{a, x_{n,c}}$ if observed
- $U^{(n)}_a(c) = 1$ for all $a$ if gapped or unobserved

**Internal node recursion:**
$$U^{(p)}_b(c) = \prod_{\text{child } n \text{ of } p} \left[ \sum_j M_{bj}(t_n) \cdot U^{(n)}_j(c) \right]$$

**Rescaling:** After each multiplication, divide by $\max_b U^{(p)}_b(c)$ and accumulate the log-normalizer $\lambda^{(p)}(c)$ to prevent underflow.

**Log-likelihood:**
$$\log P(x_c) = \lambda^{(0)}(c) + \log \left[ \sum_a \pi_a \cdot U^{(0)}_a(c) \right]$$

### Step 2: Outside algorithm (downward pass)

For each non-root node $n$ in preorder, the outside vector $D^{(n)}_a(c)$ gives the likelihood of all data *except* the subtree below $n$, given that $n$ is in state $a$.

For node $n$ with parent $p$ and sibling $s$:
$$D^{(n)}_a(c) = \left[ \sum_j M_{aj}(t_s) \cdot U^{(s)}_j(c) \right] \cdot L(x_p \mid a) \cdot \begin{cases} \pi_a & \text{if } p = \text{root} \\ \sum_i D^{(p)}_i(c) \cdot M_{ia}(t_p) & \text{otherwise} \end{cases}$$

where $L(x_p \mid a)$ is the observation likelihood at the parent node.

### Step 3: Eigensubstitution accumulation

The key insight of Holmes & Rubin (2002) is that the branch integral can be performed in the eigenbasis of $Q$, deferring the expensive back-transformation until after summing over branches.

#### J interaction matrix

For each branch $n$ with length $t_n$, define the pairwise interaction between eigenmodes $k$ and $l$:
$$J_{kl}(t_n) = \begin{cases} t_n \cdot e^{\mu_k t_n} & \text{if } \mu_k \approx \mu_l \\ \frac{e^{\mu_k t_n} - e^{\mu_l t_n}}{\mu_k - \mu_l} & \text{otherwise} \end{cases}$$

This is the integral $\int_0^{t_n} e^{\mu_k s} \, e^{\mu_l(t_n - s)} \, ds$. For irreversible models, $\mu_k$ and $J_{kl}$ are complex-valued.

#### Eigenbasis projection

**General (irreversible) case.** Project inside/outside vectors using the right eigenvectors $V$ and their inverse $V^{-1}$:
$$\tilde{U}^{(n)}_l(c) = \sum_b U^{(n)}_b(c) \cdot V_{bl}$$
$$\tilde{D}^{(n)}_k(c) = \sum_a D^{(n)}_a(c) \cdot (V^{-1})^T_{ak} = \sum_a D^{(n)}_a(c) \cdot V^{-1}_{ka}$$

These are complex-valued when $V$ is complex.

**Reversible simplification.** When $V^{-1} = V^T$ (orthogonal eigenvectors from the symmetrized decomposition), the projections absorb a $\sqrt{\pi}$ weighting to account for the symmetrization:
$$\tilde{U}^{(n)}_l(c) = \sum_b U^{(n)}_b(c) \cdot V_{bl} \cdot \sqrt{\pi_b}$$
$$\tilde{D}^{(n)}_k(c) = \sum_a D^{(n)}_a(c) \cdot V_{ak} / \sqrt{\pi_a}$$

These are always real-valued.

#### Accumulation

Sum over all non-root branches:
$$C_{kl}(c) = \sum_{n > 0} \tilde{D}^{(n)}_k(c) \cdot J_{kl}(t_n) \cdot \tilde{U}^{(n)}_l(c) \cdot \text{scale}(n,c)$$

where $\text{scale}(n,c) = \exp(\lambda_D^{(n)}(c) + \lambda_U^{(n)}(c) - \log P(x_c))$.

For irreversible models, $C_{kl}$ is complex; for reversible models, it is real. The formula is identical in both cases.

### Step 4: Back-transformation

Transform eigenbasis counts to natural basis and recover the rate matrix.

**General (irreversible) case.** Use $V$ and $V^{-1}$:
$$\text{VCV}_{ij}(c) = \sum_{kl} V_{ik} \cdot C_{kl}(c) \cdot V^{-1}_{lj}$$

Reconstruct the rate matrix: $Q_{ij} = \sum_k V_{ik} \, \mu_k \, V^{-1}_{kj}$.

- **Dwell times** (diagonal): $E[w_i(c)] = \text{Re}(\text{VCV}_{ii}(c))$
- **Substitution counts** (off-diagonal): $E[s_{ij}(c)] = \text{Re}(Q_{ij} \cdot \text{VCV}_{ij}(c))$

The final result is guaranteed real because complex eigenvalues and eigenvectors come in conjugate pairs.

**Reversible simplification.** Since $V^{-1} = V^T$, the back-transformation uses $V$ on both sides:
$$\text{VCV}_{ij}(c) = \sum_{kl} V_{ik} \cdot C_{kl}(c) \cdot V_{jl}$$

The rate matrix is replaced by its symmetrized form $S_{ij} = \sum_k V_{ik} \, \mu_k \, V_{jk}$:

- **Dwell times**: $E[w_i(c)] = \text{VCV}_{ii}(c)$
- **Substitution counts**: $E[s_{ij}(c)] = S_{ij} \cdot \text{VCV}_{ij}(c)$

All quantities are real throughout.

## Standalone branch integrals (ExpectedCounts)

Independent of any alignment or tree, we can compute the expected substitution counts and dwell times for a single CTMC branch of length $t$, conditioned on the endpoint states:

$$E[N_{i \to j}(t) \mid X(0)=a,\, X(t)=b] \quad (i \ne j)$$
$$E[T_i(t) \mid X(0)=a,\, X(t)=b]$$

### Branch integral tensor

Define the **branch integral** $W_{abij}(t) = \int_0^t M_{ai}(s) \, M_{jb}(t-s) \, ds$.

**General (irreversible) case.** In the eigenbasis of $Q = V \Lambda V^{-1}$:
$$W_{abij}(t) = \sum_{kl} V_{ak} \, V^{-1}_{ki} \cdot J_{kl}(t) \cdot V_{jl} \, V^{-1}_{lb}$$

The expected counts are:
- **Dwell**: $\text{result}[a,b,i,i] = \text{Re}(W_{abii}) / M_{ab}$
- **Subs**: $\text{result}[a,b,i,j] = \text{Re}(Q_{ij} \cdot W_{abij}) / M_{ab}$ for $i \ne j$

where $M_{ab} = M_{ab}(t) = \sum_k V_{ak} e^{\mu_k t} V^{-1}_{kb}$ is the transition probability.

**Reversible simplification.** In the symmetrized eigenbasis ($V^{-1} = V^T$):
$$W_{abij}(t) = \sum_{kl} V_{ak} \, V_{ik} \cdot J_{kl}(t) \cdot V_{bl} \, V_{jl}$$

The expected counts are:
- **Dwell**: $\text{result}[a,b,i,i] = W_{abii} / S_t[a,b]$
- **Subs**: $\text{result}[a,b,i,j] = S_{ij} \cdot W_{abij} / S_t[a,b]$ for $i \ne j$

where $S_t[a,b] = \sum_k V_{ak} e^{\mu_k t} V_{bk}$ is the **symmetrized** transition matrix (not the actual transition matrix $M_{ab} = \sqrt{\pi_b/\pi_a} \cdot S_t[a,b]$).

### Key properties

For all reachable $(a,b)$ with $M_{ab}(t) > 0$:
- **Dwell times sum to $t$**: $\sum_i \text{result}[a,b,i,i] = t$
- **Non-negative**: all entries $\ge 0$
- **$t=0$ limit**: all entries are zero

### Relationship to tree-based counts

`BranchCounts` (the tree-based per-branch expected counts from the inside-outside algorithm) returns the EM sufficient statistics, which differ from `ExpectedCounts` by a factor of the transition probability:

$$\text{BranchCounts}[r, i, j, c] = M_{ab}(t_r) \cdot \text{ExpectedCounts}[a,b,i,j]$$

where $(a,b)$ are the posterior endpoint states. This normalization convention is correct for the EM M-step, where the transition probability cancels in the ratio $E[N_{ij}] / E[T_i]$.

## F81/Jukes-Cantor fast path

For F81 (and its special case Jukes-Cantor), all non-zero eigenvalues are equal ($\mu_k = -\mu$ for $k \ge 1$), enabling a direct $O(CRA^2)$ computation without the eigenbasis. The transition probability is:
$$M_{ij}(t) = \delta_{ij} \cdot e^{-\mu t} + \pi_j \cdot (1 - e^{-\mu t})$$

The integral $I^{ab}_{ij}(t)$ decomposes into three closed-form terms parameterized by:
- $\alpha(t) = t \cdot e^{-\mu t}$
- $\beta(t) = (1 - e^{-\mu t})/\mu - t \cdot e^{-\mu t}$
- $\gamma(t) = t(1 + e^{-\mu t}) - 2(1 - e^{-\mu t})/\mu$

## Gap handling: Steiner tree branch masking

Not all branches are informative for every column. The **minimum Steiner tree** of ungapped leaves identifies which branches carry signal:

1. Classify leaves as ungapped (tokens $0 \ldots A-1$ or $A$) or gapped ($-1$, $A+1$)
2. Upward pass: propagate "has ungapped descendant" flags
3. Identify Steiner nodes: ungapped leaves, branching points ($\ge 2$ ungapped children), and pass-through nodes
4. A branch $p \to n$ is active iff both $p$ and $n$ are in the Steiner tree

## Mixture model posterior

For a mixture of $K$ rate-scaled models with prior weights $w_k$:
$$P(k \mid x_c) = \frac{w_k \cdot P(x_c \mid k)}{\sum_{k'} w_{k'} \cdot P(x_c \mid k')} = \text{softmax}_k(\log P(x_c \mid k) + \log w_k)$$

## References

- Holmes, I. & Rubin, G.M. (2002). An Expectation Maximization Algorithm for Training Hidden Substitution Models. *Journal of Molecular Biology*, 317(5), 753-764.
- Felsenstein, J. (1981). Evolutionary trees from DNA sequences: a maximum likelihood approach. *Journal of Molecular Evolution*, 17, 368-376.
- Yang, Z. (1994). Maximum likelihood phylogenetic estimation from DNA sequences with variable rates over sites. *Journal of Molecular Evolution*, 39, 306-314.
- Hasegawa, M., Kishino, H., & Yano, T. (1985). Dating of the human-ape splitting by a molecular clock of mitochondrial DNA. *Journal of Molecular Evolution*, 22, 160-174.
