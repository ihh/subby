EFFICIENT COMPUTATION OF PHYLOGENETIC SUFFICIENT STATISTICS

Suppose we have an MSA with C columns, R rows, and a binary tree with branch lengths, rooted at the target node to be annotated.
There is one MSA row for each leaf node in the tree.
Each (leafRow,column) position in the MSA contains a value from 0 to A-1 (representing ungapped positions containing observed tokens), A (representing ungapped and unobserved), or A+1 (representing gapped and unobserved).

Let Model denote the model, comprising a rate matrix and equilibrium distribution for an A-state stationary reversible continuous-time Markov chain.

For any given column of the MSA, we can identify the "unambiguously ungapped” branches: the minimum subset of branches between ungapped nodes such that deleting any more branches fragments the subtree.

Each subtree has a root. By placing a prior pi on the state at the root and a conditional probability distribution exp(R*t)_{ij} for child state j given parent state i on a branch of length t, and constraining the states at (observed) leaf nodes to their values given by the MSA, we obtain a tree-structured factor graph, applying the standard sum-product algorithm to which yields a posterior distribution for the joint distribution of the states on the endpoint of each branch.

Further, if we condition on the endpoint states {i,j} of a branch of length t, we can write down (in terms of the eigenvalues and eigenvectors of R) closed-form expressions for the a posteriori expected number s_{kl} of substitutions k->l (l \neq k) on that branch, and the expected dwell or “wait” time w_k in each state k.

Thus, by combining the posterior distribution P(i,j|MSA,tree) and the expectations E[s_{kl}|i,j,t] and E[w_k|i,j,t], and summing over unambiguously ungapped branches we can obtain E[s_{kl}|MSA,tree] and E[w_k|MSA,tree] for the (ungapped part of the) column.

In doing so we will already have computed the marginal likelihood of the observed leaves in every column.

Holmes and Rubin (“An expectation maximization algorithm for training hidden substitution models”, 2002) show how the sum-product algorithm and the accumulation of posterior-expected counts and dwell times can be performed in the eigenbasis of R, counting “eigensubstitutions”, with the expensive transformation back to the natural basis postponed until the sum-product recursion over the tree is complete.

The accompanying latex file Holmes_Rubin_2002.tex details that eigensubstitution-accumulation algorithm.
(NB this file was reverse-engineered from a PDF and may have issues.)

Our first task is to express this algorithm using parallel operations that can operate on every column of the matrix simultaneously. The accompanying Jax file toy_felsenstein_pruning.py, and in particular the function subLogLikeForMatrices, indicates how this can be done for the leaf-to-root phase of the recursion (the Felsenstein pruning recursion), using various tree index arrays (postorderBranches, parentIndex). (This implementation also uses rescaling to prevent underflow, and padding to avoid triggering JIT recompilation, though the padding is unnecessary and inefficient for fixed-depth MSAs. The implementation does not properly deal with multiple ungapped components per column.)

The task is to extend this to the root-to-leaf phase of the sum-product recursion, accumulate eigensubstitutions, and transform back to obtain per-column expectations for s_{kl} and w_k, all using operations that are as parallelizable as possible (map, scan, etc). The core function will assume that R is supplied in diagonalized form (an additional wrapper to diagonalize it should also be provided).

We will implement this in Jax; in WebGPU with a WebAssembly fallback (with a vanilla JavaScript wrapper around both WebGPU and WASM); and in Rust.

Denote by LogLike(MSA,Tree,Model) the function that returns the C column log-likelihoods. This algorithm is O(C*R*A^3) in time and O(C*R*A) in memory.

Denote by Counts(MSA,Tree,Model) the function that takes the MSA and tree and returns the expected counts s_{kl} and dwell times w_k for each column. The return value is a tensor with shape [A,A,C] where the per-column dwell times are the diagonals w_k=[k,k,:] and the substitution counts the off-diagonals s_{kl}=[k,l,:]. This algorithm is O(C*R*A^6) in time and O(C*R*A^2) in memory, although it can be optimized to O(C*R*A^4) time for Markov chains where the instantaneous rate from source state i to destination state j is independent of the source i (i.e. the F81 or Jukes-Cantor models); in those models we do not need to accumulate eigensubstitutions, as we can simply write down E[w_k|i,j,t] and E[s_{kl}|i,j,t] directly for each branch endpoint configuration. (This fact encourages the use of F81 for models with larger state spaces A.)

Note also that we do not need to consider (k,l) combinations where we know the k->l rate is zero, e.g. codon models that only allow substitution of one nucleotide at a time. This can further tame the time complexity for such sparse models.

Denote by RootProb(MSA,Tree,Model) the posterior state distribution at the root node of the unambiguously ungapped subtree.

These implementations will be used to provide just-in-time (in-browser) and preprocessed featurization of MSAs, via various tokenizations of the MSA - and corresponding models - described in the next section.

In addition to the substitution counts, dwell times, and (occasionally) the root probabilities, we will sometimes use the posterior distribution over mixture component labels assuming an independent mixture model of substitution processes in each column (which we can easily get via a SoftMax of the LogLike(MSA,Tree,Component) values for each of the components, appropriately biased by the logs of the component prior weights). 

FEATURIZATION

We will obtain summary statistics for each of the following state spaces, tokenization schemes, models, and posterior summaries, concatenating all the summary stats vectors. We will use ablation studies to investigate the cost/benefit trade-offs.

The input MSA has tokens {A,C,G,T,N,-}.

[SUBS] Purpose: detailed profile of substitutions in a column. State space: the nucleotide alphabet {A,C,G,T}. Tokenization: N’s become ungapped-unobserved; others are as in the original MSA. Model: mixture of K rate-scaled HKY85’s, representing a discretized gamma distribution over the rate multiplier, with nucleotide frequencies estimated naively from the entire MSA (no correction for phylogenetic bias), and transition/transversion ratio set to a sensible default (e.g. 2.0). Note that eigenvalues and eigenvectors are available in closed form. Summaries: the substitution counts (12 features), dwell times (4 features), and posterior over rate categories (K features).

[TRIPLETS] Purpose: nucleotide triplet statistics, unbiased by the tree. State space: the 64 nucleotide triplets. Tokenization: at each column of the original MSA, combine the nucleotide in that column, the previous nucleotide in that row (skipping over gaps), and the next nucleotide in that row (skipping over gaps). If the central position is gapped, or represents the first or last nucleotide in that row, or the triplet contains an N, then flag the position as gapped in the tokenized MSA. Model: Jukes-Cantor (64-state). Summaries: the substitution counts to each triplet (64 features) and the dwell times for each triplet (64 features).

[PHASE] Purpose: indicate whether gaps are frame-preserving. State space: the 4 tokens {NUC,0,1,2}. Tokenization: NUC represents ungapped sites in the original MSA. {0,1,2} are for gapped sites, and represent the length of the gap modulo 3. Model: Jukes-Cantor. Summaries: the substitution counts (12 features).

[ANNOT] Purpose: transfer annotations from other species. State space: the M annotation labels. Tokenization: no annotation = gapped, except for the root (target) node which is flagged as ungapped and unobserved. Model: Jukes-Cantor. Summaries: the substitution counts to each of the M labels (M features), the dwell times for each label (M features), and the root node posterior (M features).

The required tokenizations can be produced in a single pass of a finite state machine over each row. The FSM's state must include the last non-gap character (or, initially, a beginning-of-sequence sentinel token BOS) and the current gap length modulo 3. The rest of the necessary information can be obtained by looking at the adjacent column positions.

REVERSE COMPLEMENT EQUIVARIANCE

Revcomp equivariance is not baked into the model, but provided by data augmentation: all inputs (including MSA and RNA-Seq) and outputs are randomly reverse-complemented during training.

MODEL AND LOSS

We will use a tower of bidirectional Mamba layers, with RmsNorm pre-normalization.

The output of this stack will be a distribution over annotation labels (15 such labels to emulate Tiberius).
The final decoder head will be a Forward-Backward pass over an HMM, specified and implemented using Machine Boss.

In cases where RNA-Seq data are available, these will be provided as paired forward- and reverse-strand tensors of shape [3,C]. The three channels consist of read coverage at each position, number of junction reads starting at the position (donors), and number of junction reads ending at the position (acceptors), all compressed into 16-bit floats via the Borzoi transformation: $f(x) = \min(x^a, b) + \sqrt{\max(0, x^a - b)}$, with $a = 3/4$ and $b = 384$.

Each RNA-Seq track is passed through a track encoder: a bidirectional Mamba operating on the concatenated forward and reverse channel tensors (6 input channels per position), with weights shared across tracks. 

This produces a per-site per-track embedding. The Main Mamba tower incorporates interleaved cross-attention: at selected layers, each genomic position queries over the set of track embeddings at that position (one key-value pair per track). The output is a weighted pool that gets added to the residual stream.

The loss will be a categorical cross-entropy loss over the annotation labels.

IMPLEMENTATION

In addition to the code for efficient computation of phylogenetic sufficient statistics (which is of sufficiently broad use to be packaged as a separate module), we will require synchronized WebGPU/WASM and Jax versions of bidirectional revcomp-invariant Mamba. These can be based on the Jax implementation.
