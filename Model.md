# EFFICIENT COMPUTATION OF PHYLOGENETIC SUFFICIENT STATISTICS

Suppose we have an MSA with C columns, R rows, and a rooted binary tree with branch lengths.
There is one MSA row for each leaf node in the tree.
Each (leafRow,column) position in the MSA contains a value from 0 to A-1 (representing ungapped positions containing observed tokens), A (representing ungapped and unobserved), or A+1 (representing gapped and unobserved).

Let Model denote the model, comprising a rate matrix and equilibrium distribution for an A-state stationary continuous-time Markov chain. The model may be reversible (satisfying detailed balance: pi_i R_ij = pi_j R_ji) or irreversible. For reversible models, the symmetrized matrix S = diag(sqrt(pi)) R diag(1/sqrt(pi)) is diagonalized via eigh (real eigenvalues, orthonormal eigenvectors); for irreversible models, R is diagonalized directly via eig (complex eigenvalues, V^{-1} stored explicitly). Auto-detection of reversibility is supported via a detailed balance check with configurable tolerance.

For any given column of the MSA, we can identify the "unambiguously ungapped” branches: the minimum subset of branches between ungapped nodes such that deleting any more branches fragments the subtree.

Each subtree has a root. By placing a prior pi on the state at the root and a conditional probability distribution exp(R*t)_{ij} for child state j given parent state i on a branch of length t, and constraining the states at (observed) leaf nodes to their values given by the MSA, we obtain a tree-structured factor graph, applying the standard sum-product algorithm to which yields a posterior distribution for the joint distribution of the states on the endpoint of each branch.

Further, if we condition on the endpoint states {i,j} of a branch of length t, we can write down (in terms of the eigenvalues and eigenvectors of R) closed-form expressions for the a posteriori expected number s_{kl} of substitutions k->l (l \neq k) on that branch, and the expected dwell or “wait” time w_k in each state k.

Thus, by combining the posterior distribution P(i,j|MSA,tree) and the expectations E[s_{kl}|i,j,t] and E[w_k|i,j,t], and summing over unambiguously ungapped branches we can obtain E[s_{kl}|MSA,tree] and E[w_k|MSA,tree] for the (ungapped part of the) column.

In doing so we will already have computed the marginal likelihood of the observed leaves in every column.

Holmes and Rubin (“An expectation maximization algorithm for training hidden substitution models”, 2002) show how the sum-product algorithm and the accumulation of posterior-expected counts and dwell times can be performed in the eigenbasis of R, counting “eigensubstitutions”, with the expensive transformation back to the natural basis postponed until the sum-product recursion over the tree is complete. This extends naturally to irreversible models using complex eigenvalues; all intermediate quantities become complex-valued, with the final counts guaranteed real by conjugate pairing.

The accompanying latex file Holmes_Rubin_2002.tex details that eigensubstitution-accumulation algorithm.
(NB this file was reverse-engineered from a PDF and may have issues.)

Our first task is to express this algorithm using parallel operations that can operate on every column of the matrix simultaneously. The accompanying Jax file toy_felsenstein_pruning.py, and in particular the function subLogLikeForMatrices, indicates how this can be done for the leaf-to-root phase of the recursion (the Felsenstein pruning recursion), using various tree index arrays (postorderBranches, parentIndex). (This implementation also uses rescaling to prevent underflow, and padding to avoid triggering JIT recompilation, though the padding is unnecessary and inefficient for fixed-depth MSAs. The implementation does not properly deal with multiple ungapped components per column.)

The task is to extend this to the root-to-leaf phase of the sum-product recursion, accumulate eigensubstitutions, and transform back to obtain per-column expectations for s_{kl} and w_k, all using operations that are as parallelizable as possible (map, scan, etc). The core function will assume that R is supplied in diagonalized form (an additional wrapper to diagonalize it should also be provided).

We will implement this in Jax; in WebGPU with a WebAssembly fallback (with a vanilla JavaScript wrapper around both WebGPU and WASM); and in Rust.

Denote by LogLike(MSA,Tree,Model) the function that returns the C column log-likelihoods. This algorithm is O(C*R*A^3) in time and O(C*R*A) in memory.

Denote by Counts(MSA,Tree,Model) the function that takes the MSA and tree and returns the expected counts s_{kl} and dwell times w_k for each column. The return value is a tensor with shape [A,A,C] where the per-column dwell times are the diagonals w_k=[k,k,:] and the substitution counts the off-diagonals s_{kl}=[k,l,:]. This algorithm is O(C*R*A^6) in time and O(C*R*A^2) in memory, although it can be optimized to O(C*R*A^4) time for Markov chains where the instantaneous rate from source state i to destination state j is independent of the source i (i.e. the F81 or Jukes-Cantor models); in those models we do not need to accumulate eigensubstitutions, as we can simply write down E[w_k|i,j,t] and E[s_{kl}|i,j,t] directly for each branch endpoint configuration. (This fact encourages the use of F81 for models with larger state spaces A.)

Note also that we do not need to consider (k,l) combinations where we know the k->l rate is zero, e.g. codon models that only allow substitution of one nucleotide at a time. This can further tame the time complexity for such sparse models.

Denote by RootProb(MSA,Tree,Model) the posterior state distribution at the root node of the unambiguously ungapped subtree.

In addition to the substitution counts, dwell times, and (occasionally) the root probabilities, we will sometimes use the posterior distribution over mixture component labels assuming an independent mixture model of substitution processes in each column (which we can easily get via a SoftMax of the LogLike(MSA,Tree,Component) values for each of the components, appropriately biased by the logs of the component prior weights).

# IMPLEMENTATION

We implement this in JAX; in WebGPU with a WebAssembly fallback (with a vanilla JavaScript wrapper around both WebGPU and WASM); and in Rust.
