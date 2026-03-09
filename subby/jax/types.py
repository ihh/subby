from __future__ import annotations
from typing import NamedTuple, Union
import jax.numpy as jnp

from ..formats import Tree


class DiagModel(NamedTuple):
    eigenvalues: jnp.ndarray    # (*H, A)
    eigenvectors: jnp.ndarray   # (*H, A, A)  v[a,k] = component a of eigenvector k
    pi: jnp.ndarray             # (*H, A)     equilibrium distribution


class IrrevDiagModel(NamedTuple):
    eigenvalues: jnp.ndarray        # (*H, A) complex128
    eigenvectors: jnp.ndarray       # (*H, A, A) complex128 — right eigenvectors V
    eigenvectors_inv: jnp.ndarray   # (*H, A, A) complex128 — V^{-1}
    pi: jnp.ndarray                 # (*H, A) real — stationary distribution


class RateModel(NamedTuple):
    subRate: jnp.ndarray   # (*H, A, A)
    rootProb: jnp.ndarray  # (*H, A)


# Union type for any diagonalized model
AnyDiagModel = Union[DiagModel, IrrevDiagModel]
AnyModel = Union[DiagModel, IrrevDiagModel, RateModel]
