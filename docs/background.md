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

A reversible continuous-time Markov model is parameterized by:
- Rate matrix $Q$ with $Q_{ij} \ge 0$ for $i \ne j$ and $\sum_j Q_{ij} = 0$
- Equilibrium distribution $\pi$ satisfying $\pi^T Q = 0$ and detailed balance $\pi_i Q_{ij} = \pi_j Q_{ji}$

### Eigendecomposition

The symmetrized matrix $S_{ij} = Q_{ij} \sqrt{\pi_i / \pi_j}$ is real symmetric, so $S = V \Lambda V^T$ where $V$ is orthogonal and $\Lambda = \text{diag}(\mu_0, \ldots, \mu_{A-1})$ with $\mu_0 = 0 \ge \mu_1 \ge \cdots \ge \mu_{A-1}$.

The transition probability matrix is:
$$M_{ij}(t) = \sqrt{\frac{\pi_j}{\pi_i}} \sum_k V_{ik} \, e^{\mu_k t} \, V_{jk}$$

### Supported models

| Model | States | Eigendecomposition |
|-------|--------|--------------------|
| **HKY85** | 4 (A,C,G,T) | Closed-form (4 analytical eigenvectors) |
| **Jukes-Cantor** | $A$ (any) | $\mu_0 = 0$, $\mu_k = -A/(A-1)$ for $k \ge 1$; QR complement |
| **F81** | $A$ (any) | $\mu_0 = 0$, $\mu_k = -\mu$ for $k \ge 1$; QR complement |

### Rate heterogeneity

A discretized gamma distribution (Yang 1994) with shape $\alpha$ and $K$ categories provides rate multipliers $r_k$ at quantile medians, with equal weights $1/K$. Each rate category scales the eigenvalues: $\mu'_k = r_k \cdot \mu_k$.

## Algorithm

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

#### J interaction matrix

For each branch $n$ with length $t_n$, define:
$$J_{kl}(t_n) = \begin{cases} t_n \cdot e^{\mu_k t_n} & \text{if } \mu_k \approx \mu_l \\ \frac{e^{\mu_k t_n} - e^{\mu_l t_n}}{\mu_k - \mu_l} & \text{otherwise} \end{cases}$$

#### Eigenbasis projection

Project inside/outside vectors into the eigenbasis:
$$\tilde{U}^{(n)}_l(c) = \sum_b U^{(n)}_b(c) \cdot V_{bl} \cdot \sqrt{\pi_b}$$
$$\tilde{D}^{(n)}_k(c) = \sum_a D^{(n)}_a(c) \cdot V_{ak} / \sqrt{\pi_a}$$

#### Accumulation

Sum over all non-root branches:
$$C_{kl}(c) = \sum_{n > 0} \tilde{D}^{(n)}_k(c) \cdot J_{kl}(t_n) \cdot \tilde{U}^{(n)}_l(c) \cdot \text{scale}(n,c)$$

where $\text{scale}(n,c) = \exp(\lambda_D^{(n)}(c) + \lambda_U^{(n)}(c) - \log P(x_c))$.

### Step 4: Back-transformation

Transform eigenbasis counts to natural basis:
$$\text{VCV}_{ij}(c) = \sum_{kl} V_{ik} \cdot C_{kl}(c) \cdot V_{jl}$$

Reconstruct the rate matrix in the symmetrized basis: $S = V \Lambda V^T$.

- **Dwell times** (diagonal): $E[w_i(c)] = \text{VCV}_{ii}(c)$
- **Substitution counts** (off-diagonal): $E[s_{ij}(c)] = S_{ij} \cdot \text{VCV}_{ij}(c)$

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
