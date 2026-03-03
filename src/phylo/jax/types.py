from typing import NamedTuple
import jax.numpy as jnp


class Tree(NamedTuple):
    parentIndex: jnp.ndarray       # (R,) int32, preorder: parentIndex[i] < i, parentIndex[0] = -1
    distanceToParent: jnp.ndarray  # (R,) float


class DiagModel(NamedTuple):
    eigenvalues: jnp.ndarray    # (*H, A)
    eigenvectors: jnp.ndarray   # (*H, A, A)  v[a,k] = component a of eigenvector k
    pi: jnp.ndarray             # (*H, A)     equilibrium distribution


class RateModel(NamedTuple):
    subRate: jnp.ndarray   # (*H, A, A)
    rootProb: jnp.ndarray  # (*H, A)
