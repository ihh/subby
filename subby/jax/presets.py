"""Preset substitution models from published rate matrices.

Currently includes CherryML's SiteRM 400x400 site-pair coevolution model.
"""
from __future__ import annotations

import os
import numpy as np
import jax.numpy as jnp
from .types import DiagModel
from .diagonalize import diagonalize_rate_matrix_auto


def cherryml_siteRM() -> DiagModel:
    """Load CherryML's 400x400 site-pair coevolution model.

    Returns a DiagModel with A=400 states representing pairs of amino acids
    at structurally contacting sites. State ordering: pair (i,j) -> i*20+j
    using the ARNDCQEGHILKMFPSTWYV alphabet.

    Reference:
        Prillo, S. et al. "CherryML: scalable maximum likelihood estimation
        of phylogenetic models." Nature Methods (2023).

    Returns:
        DiagModel with A=400 states
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    data = np.load(os.path.join(data_dir, 'cherryml_q2.npz'))
    Q = jnp.asarray(data['Q'], dtype=jnp.float64)
    pi = jnp.asarray(data['pi'], dtype=jnp.float64)
    return diagonalize_rate_matrix_auto(Q, pi, reversible=True)
