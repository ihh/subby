import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from tensorflow_probability.substrates.jax.distributions import Gamma

def alignmentIsValid (alignment, alphabetSize):
    assert jnp.all(alignment >= -1)
    assert jnp.all(alignment < alphabetSize)

def treeIsValid (distanceToParent, parentIndex, alignmentRows):
    assert jnp.all(distanceToParent >= 0)
    assert jnp.all(parentIndex[1:] >= -1)
    assert jnp.all(parentIndex <= jnp.arange(alignmentRows))

# Compute substitution log-likelihood of a multiple sequence alignment and phylogenetic tree by pruning
# Parameters:
#  - alignment: (R,C) integer tokens. C is the length of the alignment (#cols), R is the number of sequences (#rows). A token of -1 indicates a gap.
#  - distanceToParent: (R,) floats, distance to parent node
#  - parentIndex: (R,) integers, index of parent node. Nodes are sorted in preorder so parentIndex[i] <= i for all i. parentIndex[0] = -1
#  - subRate: (*H,A,A) substitution rate matrix/matrices. Leading H axes (if any) are "hidden" substitution rate categories, A is alphabet size
#  - rootProb: (*H,A) root frequencies (typically equilibrium frequencies for substitution rate matrix)
# To pad rows, set parentIndex[paddingRow:] = arange(paddingRow,R)
# To pad columns, set alignment[paddingRow,paddingCol:] = -1

def subLogLike (alignment, distanceToParent, parentIndex, subRate, rootProb):
    subMatrix = computeSubMatrixForTimes (distanceToParent, subRate)
    return subLogLikeForMatrices (alignment, parentIndex, subMatrix, rootProb)

def computeSubMatrixForTimes (distanceToParent, subRate):
    assert distanceToParent.ndim == 1
    assert subRate.ndim >= 2
    R, = distanceToParent.shape
    *H, A = subRate.shape[0:-1]
    assert subRate.shape == (*H,A,A)
    # Compute transition matrices per branch
    subMatrix = expm (jnp.einsum('...ij,r->...rij', subRate, distanceToParent))  # (*H,R,A,A)
    return subMatrix

def subLogLikeForMatrices (alignment, parentIndex, subMatrix, rootProb, maxChunkSize = 128):
    assert alignment.ndim == 2
    assert subMatrix.ndim >= 3
    *H, R, A = subMatrix.shape[0:-1]
    C = alignment.shape[-1]
    assert parentIndex.shape == (R,)
    assert rootProb.shape == (*H,A)
    assert subMatrix.shape == (*H,R,A,A)
    assert alignment.dtype == jnp.int32
    assert parentIndex.dtype == jnp.int32
    # If too big, split into chunks
    if C > maxChunkSize:
#        jax.debug.print('Splitting %d x %d alignment into %d chunks of size %d x %d' % (R,C,C//maxChunkSize,R,maxChunkSize))
        return jnp.concatenate ([subLogLikeForMatrices (alignment[:,i:i+maxChunkSize], parentIndex, subMatrix, rootProb) for i in range(0,C,maxChunkSize)], axis=-1)
    # Initialize pruning matrix
    tokenLookup = jnp.concatenate([jnp.ones(A)[None,:],jnp.eye(A)])
    likelihood = tokenLookup[alignment + 1]  # (R,C,A)
    if len(H) > 0:
        likelihood += jnp.zeros((*H,R,C,A))
    logNorm = jnp.zeros((*H,C))  # (*H,C)
    # Compute log-likelihood for all columns in parallel by iterating over nodes in postorder
    postorderBranches = (jnp.arange(R-1,0,-1),  # child indices
                         jnp.flip(parentIndex[1:]), # parent indices
                         jnp.flip(jnp.moveaxis(subMatrix,-3,0)[1:,...],axis=0))  # substitution matrices
    (likelihood, logNorm), _dummy = jax.lax.scan (computeLogLikeForBranch, (likelihood, logNorm), postorderBranches)
    logNorm = logNorm + jnp.log(jnp.einsum('...ci,...i->...c', likelihood[...,0,:,:], rootProb))  # (*H,C)
    return logNorm

def computeLogLikeForBranch (vars, branch):
    likelihood, logNorm = vars
    child, parent, subMatrix = branch
    likelihood = likelihood.at[...,parent,:,:].multiply (jnp.einsum('...ij,...cj->...ci', subMatrix, likelihood[...,child,:,:]))
    maxLike = jnp.max(likelihood[...,parent,:,:], axis=-1)  # (*H,C)
    likelihood = likelihood.at[...,parent,:,:].divide (maxLike[...,None])  # guard against underflow
    logNorm = logNorm + jnp.log(maxLike)
    return (likelihood, logNorm), None

def padDimension (len, multiplier):
    return jnp.where (multiplier == 2,  # handle this case specially to avoid precision errors
                      1 << (len-1).bit_length(),
                      int (jnp.ceil (multiplier ** jnp.ceil (jnp.log(len) / jnp.log(multiplier)))))

def padAlignment (alignment, parentIndex, distanceToParent, transCounts, nRows: int = None, nCols: int = None, colMultiplier = 2, rowMultiplier = 2):
    unpaddedRows, unpaddedCols = alignment.shape
    if nCols is None:
        nCols = padDimension (unpaddedCols, colMultiplier)
    if nRows is None:
        nRows = padDimension (unpaddedRows, rowMultiplier)
    # To pad rows, set parentIndex[paddingRow:] = arange(paddingRow,R) and pad alignment and distanceToParent with any value (-1 and 0 for predictability),
    # and pad transCounts with zeros
    if nRows > unpaddedRows:
        alignment = jnp.concatenate ([alignment, -1*jnp.ones((nRows - unpaddedRows, unpaddedCols), dtype=alignment.dtype)], axis=0)
        distanceToParent = jnp.concatenate ([distanceToParent, jnp.zeros(nRows - unpaddedRows, dtype=distanceToParent.dtype)], axis=0)
        parentIndex = jnp.concatenate ([parentIndex, jnp.arange(unpaddedRows,nRows, dtype=parentIndex.dtype)], axis=0)
        transCounts = jnp.concatenate ([transCounts, jnp.zeros((nRows - unpaddedRows, *transCounts.shape[1:]), dtype=transCounts.dtype)], axis=0)
    # To pad columns, set alignment[paddingRow,paddingCol:] = -1
    if nCols > unpaddedCols:
        alignment = jnp.concatenate ([alignment, -1*jnp.ones((nRows, nCols - unpaddedCols), dtype=alignment.dtype)], axis=1)
    return alignment, parentIndex, distanceToParent, transCounts

def normalizeSubRate (subRate):
    subRate = jnp.abs (subRate)
    return subRate - jnp.diag(jnp.sum(subRate, axis=-1))

def zeroDiagonal (matrix):
    return matrix - jnp.diag(jnp.diag(matrix))

def logitsToProbs (logits):
    return jax.nn.softmax(logits,axis=-1)

def probsToLogits (probs):
    return jnp.log (probs)

def logitToProb (logit):
    return 1 / (1 + jnp.exp (logit))

def probToLogit (prob):
    return jnp.log (1/jnp.clip(prob,0,1) - 1)

def normalizeRootProb (rootProb):
    return rootProb / jnp.sum(rootProb)

def normalizeSubModel (subRate, rootProb):
    return normalizeSubRate(subRate), normalizeRootProb(rootProb)

def parametricSubModel (subRate, rootLogits):
    return normalizeSubRate(subRate), logitsToProbs(rootLogits)

def exchangeabilityMatrixToSubMatrix (exchangeRate, rootProb):
    sqrtRootProb = jnp.sqrt(rootProb)
    return jnp.einsum('i,...ij,j->...ij', 1/sqrtRootProb, exchangeRate, sqrtRootProb)

def subMatrixToExchangeabilityMatrix (subMatrix, rootProb):
    sqrtRootProb = jnp.sqrt(rootProb)
    return jnp.einsum('i,...ij,j->...ij', sqrtRootProb, subMatrix, 1/sqrtRootProb)

def symmetrizeSubRate (matrix):
    return normalizeSubRate (0.5 * (matrix + matrix.swapaxes(-1,-2)))

def parametricReversibleSubModel (subRate, rootLogits):
    rootProb = logitsToProbs (rootLogits)
    subRate = exchangeabilityMatrixToSubMatrix (symmetrizeSubRate(subRate), rootProb)
    return subRate, rootProb
