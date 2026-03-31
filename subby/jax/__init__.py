from __future__ import annotations
from typing import Optional, Union

import jax.numpy as jnp

from .types import Tree, DiagModel, IrrevDiagModel, RateModel, AnyDiagModel, AnyModel
from .diagonalize import (
    diagonalize_rate_matrix, compute_sub_matrices,
    diagonalize_irreversible, compute_sub_matrices_irrev,
    check_detailed_balance, diagonalize_rate_matrix_auto,
)
from .pruning import upward_pass
from .outside import downward_pass
from .eigensub import (
    compute_J, eigenbasis_project, accumulate_C, back_transform,
    compute_J_complex, eigenbasis_project_irrev, accumulate_C_complex, back_transform_irrev,
    accumulate_C_per_branch, back_transform_per_branch,
    accumulate_C_complex_per_branch, back_transform_irrev_per_branch,
    compute_J_per_row, compute_J_per_row_complex,
    eigenbasis_project_per_row, eigenbasis_project_per_row_irrev,
    back_transform_per_row, back_transform_per_row_irrev,
)
from ._utils import (
    GEOM_BINS,
    pad_to_geom_bin,
    pad_alignment,
    pad_tree,
    pad_tree_and_alignment,
    unpad_columns,
)
from .f81_fast import f81_counts, f81_counts_per_branch
from .mixture import mixture_posterior
from .vjp import make_loglike_custom_grad, make_loglike_pade_grad, _extract_rate_matrix
from .models import (
    hky85_diag, jukes_cantor_model, f81_model,
    gamma_rate_categories, scale_model,
    irrev_model_from_rate_matrix, model_from_rate_matrix,
    gy94_model,
)
from .ctmc import ExpectedCounts, expected_counts_eigen, expected_counts_eigen_irrev
from .presets import cherryml_siteRM


def _ensure_diag(model: AnyModel) -> AnyDiagModel:
    """Auto-diagonalize if given a RateModel."""
    if isinstance(model, RateModel):
        return diagonalize_rate_matrix(model.subRate, model.rootProb)
    if isinstance(model, IrrevDiagModel):
        return model
    return model


def _is_model_list(model) -> bool:
    """Check if model is a list/tuple of models (per-column)."""
    return isinstance(model, (list, tuple)) and len(model) > 0 and not isinstance(model, (Tree, DiagModel, IrrevDiagModel, RateModel))


def _is_model_grid(model) -> bool:
    """Check if model is a 2D list of models (per-column-per-row grid).

    A model grid is a list of lists: model[c][r] gives the model for
    column c, row (branch) r.  Shape (C, R) or (1, R).
    """
    if not isinstance(model, (list, tuple)) or len(model) == 0:
        return False
    first = model[0]
    # Model NamedTuples are tuples but not grids
    if isinstance(first, (DiagModel, IrrevDiagModel, RateModel)):
        return False
    return isinstance(first, (list, tuple))


def _compute_sub_matrices_per_row(
    models_row: list[AnyDiagModel], distances: jnp.ndarray, is_irrev: bool,
) -> jnp.ndarray:
    """Compute substitution matrices where each branch has its own model.

    Args:
        models_row: list of R DiagModel or IrrevDiagModel
        distances: (R,) branch lengths
        is_irrev: True for irreversible models

    Returns:
        (R, A, A) substitution matrices
    """
    eigenvalues = jnp.stack([m.eigenvalues for m in models_row])  # (R, A)
    V = jnp.stack([m.eigenvectors for m in models_row])           # (R, A, A)

    exp_mu_t = jnp.exp(eigenvalues * distances[:, None])          # (R, A)

    if is_irrev:
        V_inv = jnp.stack([m.eigenvectors_inv for m in models_row])
        M = jnp.einsum('rak,rk,rkj->raj', V, exp_mu_t, V_inv)
        return M.real
    else:
        pi = jnp.stack([m.pi for m in models_row])
        S = jnp.einsum('rak,rk,rbk->rab', V, exp_mu_t, V)
        sqrt_pi = jnp.sqrt(pi)
        M = S * (1.0 / sqrt_pi)[:, :, None] * sqrt_pi[:, None, :]
        return M


def _branch_counts_per_row(
    alignment: jnp.ndarray,
    tree: Tree,
    models_row: list[AnyDiagModel],
    maxChunkSize: int = 128,
) -> jnp.ndarray:
    """Compute per-branch counts where each branch has its own model.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        models_row: list of R diagonalized models

    Returns:
        (R, A, A, C) per-branch counts (branch 0 = zeros)
    """
    R = alignment.shape[0]
    assert len(models_row) == R
    is_irrev = isinstance(models_row[0], IrrevDiagModel)

    eigenvalues_row = jnp.stack([m.eigenvalues for m in models_row])
    V_row = jnp.stack([m.eigenvectors for m in models_row])
    pi_row = jnp.stack([m.pi for m in models_row])

    subMatrices = _compute_sub_matrices_per_row(models_row, tree.distanceToParent, is_irrev)
    rootProb = pi_row[0]

    U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, rootProb, maxChunkSize)
    D, logNormD = downward_pass(U, logNormU, tree, subMatrices, rootProb, alignment)

    if is_irrev:
        V_inv_row = jnp.stack([m.eigenvectors_inv for m in models_row])
        J = compute_J_per_row_complex(eigenvalues_row, tree.distanceToParent)
        U_tilde, D_tilde = eigenbasis_project_per_row_irrev(U, D, V_row, V_inv_row)
        C_pb = accumulate_C_per_branch(
            D_tilde, U_tilde, J, logNormU, logNormD, logLike, tree.parentIndex,
        )
        return back_transform_per_row_irrev(C_pb, V_row, V_inv_row, eigenvalues_row)
    else:
        J = compute_J_per_row(eigenvalues_row, tree.distanceToParent)
        U_tilde, D_tilde = eigenbasis_project_per_row(U, D, V_row, pi_row)
        C_pb = accumulate_C_per_branch(
            D_tilde, U_tilde, J, logNormU, logNormD, logLike, tree.parentIndex,
        )
        return back_transform_per_row(C_pb, V_row, eigenvalues_row)


def _stack_models(models: list[AnyModel]) -> tuple[AnyDiagModel, bool]:
    """Stack a list of models into a single model with leading C dimension.

    Returns:
        (stacked_model, is_irrev)
    """
    models = [_ensure_diag(m) for m in models]
    is_irrev = isinstance(models[0], IrrevDiagModel)
    if is_irrev:
        return IrrevDiagModel(
            eigenvalues=jnp.stack([m.eigenvalues for m in models]),
            eigenvectors=jnp.stack([m.eigenvectors for m in models]),
            eigenvectors_inv=jnp.stack([m.eigenvectors_inv for m in models]),
            pi=jnp.stack([m.pi for m in models]),
        ), True
    else:
        return DiagModel(
            eigenvalues=jnp.stack([m.eigenvalues for m in models]),
            eigenvectors=jnp.stack([m.eigenvectors for m in models]),
            pi=jnp.stack([m.pi for m in models]),
        ), False


def _compute_sub_matrices_per_column(
    stacked_model: AnyDiagModel, distances: jnp.ndarray, is_irrev: bool
) -> jnp.ndarray:
    """Compute per-column substitution matrices from a stacked model.

    stacked_model fields have leading C dimension.
    Returns: (R, C, A, A) substitution matrices.
    """
    if is_irrev:
        V = stacked_model.eigenvectors       # (C, A, A)
        V_inv = stacked_model.eigenvectors_inv  # (C, A, A)
        mu = stacked_model.eigenvalues        # (C, A)
        # exp(mu_k * t_r): (C, R, A)
        exp_mu_t = jnp.exp(mu[:, None, :] * distances[None, :, None])
        # M_ij(c,t) = sum_k V_ik(c) * exp(mu_k(c)*t) * V_inv_kj(c)
        M = jnp.einsum('cak,crk,ckj->craj', V, exp_mu_t, V_inv)
        # Transpose to (R, C, A, A)
        M = jnp.moveaxis(M, 0, 1)
        return M.real
    else:
        V = stacked_model.eigenvectors   # (C, A, A)
        mu = stacked_model.eigenvalues   # (C, A)
        pi = stacked_model.pi            # (C, A)
        exp_mu_t = jnp.exp(mu[:, None, :] * distances[None, :, None])
        S_t = jnp.einsum('cak,crk,cbk->crab', V, exp_mu_t, V)
        sqrt_pi = jnp.sqrt(pi)
        inv_sqrt_pi = 1.0 / sqrt_pi
        # M_ij = sqrt(pi_j/pi_i) * S_ij
        M = S_t * inv_sqrt_pi[:, None, :, None] * sqrt_pi[:, None, None, :]
        # Transpose to (R, C, A, A)
        M = jnp.moveaxis(M, 0, 1)
        return M


def LogLike(
    alignment: jnp.ndarray,
    tree: Tree,
    model,
    maxChunkSize: int = 128,
) -> jnp.ndarray:
    """Compute per-column log-likelihoods.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel, IrrevDiagModel, RateModel, list of C models
            (per-column), or list of lists (model grid):
            - ``[[m0, m1, ..., m_{R-1}]]`` — (1, R) per-row, broadcast to all columns
            - ``[[...], [...], ...]`` — (C, R) per-column-per-row
        maxChunkSize: chunk columns for memory

    Returns:
        (*H, C) log-likelihoods
    """
    if _is_model_grid(model):
        grid = model
        C_grid = len(grid)
        R, C = alignment.shape

        if C_grid == 1:
            # (1, R): per-row, broadcast to all columns
            models_row = [_ensure_diag(m) for m in grid[0]]
            assert len(models_row) == R, f"Expected {R} per-row models, got {len(models_row)}"
            is_irrev = isinstance(models_row[0], IrrevDiagModel)
            subMatrices = _compute_sub_matrices_per_row(
                models_row, tree.distanceToParent, is_irrev
            )
            rootProb = models_row[0].pi
            _, _, logLike = upward_pass(
                alignment, tree, subMatrices, rootProb, maxChunkSize
            )
            return logLike
        else:
            # (C, R): per-column-per-row
            assert C_grid == C, f"Model grid has {C_grid} columns but alignment has {C}"
            grid_diag = [[_ensure_diag(m) for m in grid[c]] for c in range(C)]
            is_irrev = isinstance(grid_diag[0][0], IrrevDiagModel)
            sub_cols = []
            for c in range(C):
                assert len(grid_diag[c]) == R
                M_c = _compute_sub_matrices_per_row(
                    grid_diag[c], tree.distanceToParent, is_irrev
                )
                sub_cols.append(M_c)  # (R, A, A)
            subMatrices = jnp.stack(sub_cols, axis=1)  # (R, C, A, A)

            # Upward pass with per_column=True; rootProb is a dummy for the pass
            # (U and logNormU don't depend on rootProb)
            rootProb_dummy = grid_diag[0][0].pi
            U, logNormU, _ = upward_pass(
                alignment, tree, subMatrices, rootProb_dummy, maxChunkSize,
                per_column=True,
            )
            # Recompute logLike with correct per-column rootProb
            rootProb_per_col = jnp.stack([grid_diag[c][0].pi for c in range(C)])
            root_like = U[..., 0, :, :]  # (*H, C, A)
            logLike = logNormU[..., 0, :] + jnp.log(
                jnp.sum(root_like * rootProb_per_col, axis=-1)
            )
            return logLike

    if _is_model_list(model):
        stacked, is_irrev = _stack_models(model)
        subMatrices = _compute_sub_matrices_per_column(
            stacked, tree.distanceToParent, is_irrev
        )
        # Use first model's pi for root (all should be same or broadcastable)
        rootProb = stacked.pi[0]
        _, _, logLike = upward_pass(
            alignment, tree, subMatrices, rootProb, maxChunkSize, per_column=True
        )
        return logLike
    model = _ensure_diag(model)
    if isinstance(model, IrrevDiagModel):
        subMatrices = compute_sub_matrices_irrev(model, tree.distanceToParent)
    else:
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)
    _, _, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
    return logLike


def LogLikeCustomGrad(
    alignment: jnp.ndarray,
    tree: Tree,
    model: AnyModel,
    maxChunkSize: int = 128,
    method: str = 'eigen',
) -> jnp.ndarray:
    """Like LogLike but with custom VJP for faster distance gradients.

    Two methods are available:

    - ``'eigen'`` (default): Uses eigenbasis projection. The backward pass
      projects inside/outside vectors into the eigenbasis and contracts with
      μ_k·exp(μ_k·t). Requires eigendecomposition of the rate matrix.

    - ``'pade'``: Uses Padé matrix exponential (``jax.scipy.linalg.expm``)
      for the forward pass and Q·M(t) contraction for the backward pass.
      Avoids eigendecomposition entirely.

    Both methods compute the same gradient with the same asymptotic
    complexity O(CRA²). They differ in constant factors.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel, IrrevDiagModel, or RateModel
        maxChunkSize: chunk columns for memory
        method: 'eigen' or 'pade'

    Returns:
        (*H, C) log-likelihoods
    """
    if method == 'pade':
        Q, pi = _extract_rate_matrix(
            model if isinstance(model, (DiagModel, IrrevDiagModel, RateModel))
            else _ensure_diag(model)
        )
        f = make_loglike_pade_grad(Q, pi, alignment, tree.parentIndex, maxChunkSize)
    else:
        model = _ensure_diag(model)
        f = make_loglike_custom_grad(model, alignment, tree.parentIndex, maxChunkSize)
    return f(tree.distanceToParent)


def Counts(
    alignment: jnp.ndarray,
    tree: Tree,
    model,
    maxChunkSize: int = 128,
    f81_fast_flag: bool = False,
    branch_mask: Optional[Union[str, jnp.ndarray]] = "auto",
) -> jnp.ndarray:
    """Compute expected substitution counts and dwell times per column.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel, IrrevDiagModel, RateModel, or list of models
            (one per column, len must equal C)
        maxChunkSize: chunk columns for memory
        f81_fast_flag: if True, use O(CRA^2) F81/JC fast path
        branch_mask: "auto" (compute from alignment), None (no masking),
            or (*H, R, C) bool array

    Returns:
        (*H, A, A, C) counts tensor (diag=dwell, off-diag=substitution counts)
    """
    if _is_model_grid(model):
        grid = model
        C_grid = len(grid)
        R, C = alignment.shape

        if C_grid == 1:
            # (1, R): per-row, broadcast to all columns
            models_row = [_ensure_diag(m) for m in grid[0]]
            assert len(models_row) == R, f"Expected {R} per-row models, got {len(models_row)}"
            assert not f81_fast_flag, "F81 fast path not supported with per-row models"

            if isinstance(branch_mask, str) and branch_mask == "auto":
                from .components import compute_branch_mask
                A = models_row[0].pi.shape[-1]
                branch_mask = compute_branch_mask(alignment, tree.parentIndex, A)

            bc = _branch_counts_per_row(alignment, tree, models_row, maxChunkSize)
            if branch_mask is not None:
                bc = bc * branch_mask[:, None, None, :]
            return jnp.sum(bc, axis=0)
        else:
            # (C, R): per-column-per-row
            assert C_grid == C, f"Model grid has {C_grid} columns but alignment has {C}"
            results = []
            for c in range(C):
                col_counts = Counts(
                    alignment[:, c:c+1], tree, [grid[c]],
                    maxChunkSize=maxChunkSize,
                    f81_fast_flag=f81_fast_flag,
                    branch_mask=branch_mask[..., c:c+1] if branch_mask is not None and not isinstance(branch_mask, str) else branch_mask,
                )
                results.append(col_counts)
            return jnp.concatenate(results, axis=-1)

    if _is_model_list(model):
        # Per-column models: compute column-by-column since the eigensub
        # pipeline requires model fields to align with *H batch dims,
        # which conflicts with the C (column) dimension.
        models_diag = [_ensure_diag(m) for m in model]
        C = alignment.shape[1]
        assert len(models_diag) == C, f"Expected {C} models, got {len(models_diag)}"
        results = []
        for c in range(C):
            col_counts = Counts(
                alignment[:, c:c+1], tree, models_diag[c],
                maxChunkSize=maxChunkSize,
                f81_fast_flag=f81_fast_flag,
                branch_mask=branch_mask[..., c:c+1] if branch_mask is not None and not isinstance(branch_mask, str) else branch_mask,
            )
            results.append(col_counts)
        return jnp.concatenate(results, axis=-1)

    model = _ensure_diag(model)
    is_irrev = isinstance(model, IrrevDiagModel)

    if isinstance(branch_mask, str) and branch_mask == "auto":
        from .components import compute_branch_mask
        A = model.pi.shape[-1]
        branch_mask = compute_branch_mask(alignment, tree.parentIndex, A)

    if is_irrev:
        assert not f81_fast_flag, "F81 fast path is reversible-only"
        subMatrices = compute_sub_matrices_irrev(model, tree.distanceToParent)
    else:
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)

    U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
    D, logNormD = downward_pass(U, logNormU, tree, subMatrices, model.pi, alignment)

    if f81_fast_flag:
        return f81_counts(
            U, D, logNormU, logNormD, logLike,
            tree.distanceToParent, model.pi, tree.parentIndex,
            branch_mask=branch_mask,
        )
    elif is_irrev:
        J = compute_J_complex(model.eigenvalues, tree.distanceToParent)
        U_tilde, D_tilde = eigenbasis_project_irrev(U, D, model)
        C = accumulate_C_complex(D_tilde, U_tilde, J, logNormU, logNormD, logLike,
                                 tree.parentIndex, branch_mask=branch_mask)
        return back_transform_irrev(C, model)
    else:
        J = compute_J(model.eigenvalues, tree.distanceToParent)
        U_tilde, D_tilde = eigenbasis_project(U, D, model)
        C = accumulate_C(D_tilde, U_tilde, J, logNormU, logNormD, logLike, tree.parentIndex,
                         branch_mask=branch_mask)
        return back_transform(C, model)


def BranchCounts(
    alignment: jnp.ndarray,
    tree: Tree,
    model,
    maxChunkSize: int = 128,
    f81_fast_flag: bool = False,
) -> jnp.ndarray:
    """Compute per-branch expected substitution counts and dwell times.

    Returns the (A, A) matrix of a posteriori expected transition counts
    and dwell times for a discrete-state continuous-time Markov chain on
    each branch of the tree, for each alignment column.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel, IrrevDiagModel, RateModel, or list of models
        maxChunkSize: chunk columns for memory
        f81_fast_flag: if True, use O(CRA^2) F81/JC fast path

    Returns:
        (*H, R, A, A, C) per-branch counts (diag=dwell, off-diag=substitutions).
        Branch 0 (root) is zeros.
    """
    if _is_model_grid(model):
        grid = model
        C_grid = len(grid)
        R, C = alignment.shape

        if C_grid == 1:
            # (1, R): per-row, broadcast to all columns
            models_row = [_ensure_diag(m) for m in grid[0]]
            assert len(models_row) == R
            assert not f81_fast_flag, "F81 fast path not supported with per-row models"
            return _branch_counts_per_row(alignment, tree, models_row, maxChunkSize)
        else:
            # (C, R): per-column-per-row
            assert C_grid == C
            results = []
            for c in range(C):
                col_bc = BranchCounts(
                    alignment[:, c:c+1], tree, [grid[c]],
                    maxChunkSize=maxChunkSize, f81_fast_flag=f81_fast_flag,
                )
                results.append(col_bc)
            return jnp.concatenate(results, axis=-1)

    if _is_model_list(model):
        models_diag = [_ensure_diag(m) for m in model]
        C = alignment.shape[1]
        assert len(models_diag) == C
        results = []
        for c in range(C):
            col_bc = BranchCounts(
                alignment[:, c:c+1], tree, models_diag[c],
                maxChunkSize=maxChunkSize, f81_fast_flag=f81_fast_flag,
            )
            results.append(col_bc)
        return jnp.concatenate(results, axis=-1)

    model = _ensure_diag(model)
    is_irrev = isinstance(model, IrrevDiagModel)

    if is_irrev:
        assert not f81_fast_flag, "F81 fast path is reversible-only"
        subMatrices = compute_sub_matrices_irrev(model, tree.distanceToParent)
    else:
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)

    U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)
    D, logNormD = downward_pass(U, logNormU, tree, subMatrices, model.pi, alignment)

    if f81_fast_flag:
        return f81_counts_per_branch(
            U, D, logNormU, logNormD, logLike,
            tree.distanceToParent, model.pi, tree.parentIndex,
        )
    elif is_irrev:
        J = compute_J_complex(model.eigenvalues, tree.distanceToParent)
        U_tilde, D_tilde = eigenbasis_project_irrev(U, D, model)
        C = accumulate_C_complex_per_branch(
            D_tilde, U_tilde, J, logNormU, logNormD, logLike, tree.parentIndex,
        )
        return back_transform_irrev_per_branch(C, model)
    else:
        J = compute_J(model.eigenvalues, tree.distanceToParent)
        U_tilde, D_tilde = eigenbasis_project(U, D, model)
        C = accumulate_C_per_branch(
            D_tilde, U_tilde, J, logNormU, logNormD, logLike, tree.parentIndex,
        )
        return back_transform_per_branch(C, model)


def RootProb(
    alignment: jnp.ndarray,
    tree: Tree,
    model,
    maxChunkSize: int = 128,
) -> jnp.ndarray:
    """Compute posterior root state distribution per column.

    q_b = pi_b * U^(root)_b / P(x)

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel, IrrevDiagModel, RateModel, or list of models
            (one per column, len must equal C)
        maxChunkSize: chunk columns for memory

    Returns:
        (*H, A, C) posterior root distribution
    """
    if _is_model_grid(model):
        grid = model
        C_grid = len(grid)
        R, C = alignment.shape

        if C_grid == 1:
            # (1, R): per-row, broadcast to all columns
            models_row = [_ensure_diag(m) for m in grid[0]]
            assert len(models_row) == R
            is_irrev = isinstance(models_row[0], IrrevDiagModel)
            subMatrices = _compute_sub_matrices_per_row(
                models_row, tree.distanceToParent, is_irrev
            )
            rootProb = models_row[0].pi
            U, logNormU, logLike = upward_pass(
                alignment, tree, subMatrices, rootProb, maxChunkSize
            )
            root_U = U[..., 0, :, :]
            log_scale = logNormU[..., 0, :] - logLike
            scale = jnp.exp(log_scale)
            q = rootProb[..., None, :] * root_U * scale[..., None]
            q = jnp.moveaxis(q, -1, -2)
            return q
        else:
            # (C, R): per-column-per-row — vectorize via upward pass
            assert C_grid == C
            grid_diag = [[_ensure_diag(m) for m in grid[c]] for c in range(C)]
            is_irrev = isinstance(grid_diag[0][0], IrrevDiagModel)
            sub_cols = []
            for c in range(C):
                assert len(grid_diag[c]) == R
                M_c = _compute_sub_matrices_per_row(
                    grid_diag[c], tree.distanceToParent, is_irrev
                )
                sub_cols.append(M_c)
            subMatrices = jnp.stack(sub_cols, axis=1)  # (R, C, A, A)

            rootProb_dummy = grid_diag[0][0].pi
            U, logNormU, _ = upward_pass(
                alignment, tree, subMatrices, rootProb_dummy, maxChunkSize,
                per_column=True,
            )
            rootProb_per_col = jnp.stack([grid_diag[c][0].pi for c in range(C)])
            root_like = U[..., 0, :, :]
            logLike = logNormU[..., 0, :] + jnp.log(
                jnp.sum(root_like * rootProb_per_col, axis=-1)
            )
            root_U = root_like
            log_scale = logNormU[..., 0, :] - logLike
            scale = jnp.exp(log_scale)
            q = rootProb_per_col * root_U * scale[..., None]
            q = jnp.moveaxis(q, -1, -2)
            return q

    if _is_model_list(model):
        stacked, is_irrev = _stack_models(model)
        subMatrices = _compute_sub_matrices_per_column(
            stacked, tree.distanceToParent, is_irrev
        )
        rootProb = stacked.pi[0]
        U, logNormU, logLike = upward_pass(
            alignment, tree, subMatrices, rootProb, maxChunkSize, per_column=True
        )
        # Per-column pi: stacked.pi is (C, A)
        pi_per_col = stacked.pi  # (C, A)
        root_U = U[..., 0, :, :]  # (*H, C, A)
        log_scale = logNormU[..., 0, :] - logLike
        scale = jnp.exp(log_scale)
        q = pi_per_col * root_U * scale[..., None]
        q = jnp.moveaxis(q, -1, -2)
        return q

    model = _ensure_diag(model)
    if isinstance(model, IrrevDiagModel):
        subMatrices = compute_sub_matrices_irrev(model, tree.distanceToParent)
    else:
        subMatrices = compute_sub_matrices(model, tree.distanceToParent)
    U, logNormU, logLike = upward_pass(alignment, tree, subMatrices, model.pi, maxChunkSize)

    # q_b = pi_b * U_root_b / P(x)
    # With rescaling: q_b = pi_b * U_rescaled_root_b * exp(logNormU_root) / exp(logLike)
    root_U = U[..., 0, :, :]  # (*H, C, A)
    log_scale = logNormU[..., 0, :] - logLike  # (*H, C)
    scale = jnp.exp(log_scale)
    q = model.pi[..., None, :] * root_U * scale[..., None]  # (*H, C, A)
    # Transpose to (*H, A, C)
    q = jnp.moveaxis(q, -1, -2)
    return q


def MixturePosterior(
    alignment: jnp.ndarray,
    tree: Tree,
    models: list[AnyModel],
    log_weights: jnp.ndarray,
    maxChunkSize: int = 128,
) -> jnp.ndarray:
    """Compute posterior over mixture components per column.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        models: list of K DiagModel, IrrevDiagModel, or RateModel instances
        log_weights: (K,) log prior weights
        maxChunkSize: chunk columns for memory

    Returns:
        (K, C) posterior probabilities
    """
    log_likes = jnp.stack([
        LogLike(alignment, tree, m, maxChunkSize) for m in models
    ])  # (K, C) or (K, *H, C)
    # If there are hidden dims H, sum over them for mixture
    while log_likes.ndim > 2:
        log_likes = jnp.sum(log_likes, axis=1)
    return mixture_posterior(log_likes, log_weights)


class InsideOutside:
    """Inside-outside DP tables for querying posteriors.

    Runs the upward (inside) and downward (outside) passes once and stores the
    resulting vectors, enabling efficient queries for log-likelihoods, expected
    substitution counts, node state posteriors, and branch endpoint joint
    posteriors without recomputation.

    Args:
        alignment: (R, C) int32 tokens
        tree: Tree
        model: DiagModel, IrrevDiagModel, RateModel, or list of models
        maxChunkSize: chunk columns for memory

    Example::

        io = InsideOutside(alignment, tree, model)
        ll = io.log_likelihood               # (*H, C)
        root_post = io.node_posterior(0)      # (*H, A, C)
        branch_post = io.branch_posterior(5)  # (*H, A, A, C)
        counts = io.counts()                  # (*H, A, A, C)
    """

    def __init__(
        self,
        alignment: jnp.ndarray,
        tree: Tree,
        model,
        maxChunkSize: int = 128,
    ):
        self._alignment = alignment
        self._tree = tree
        self._maxChunkSize = maxChunkSize

        if _is_model_grid(model):
            raise NotImplementedError(
                "InsideOutside does not support model grids (2D model arrays). "
                "Use LogLike, Counts, BranchCounts, RootProb directly."
            )

        if _is_model_list(model):
            stacked, is_irrev = _stack_models(model)
            subMatrices = _compute_sub_matrices_per_column(
                stacked, tree.distanceToParent, is_irrev
            )
            rootProb = stacked.pi[0]
            per_column = True
            self._model = stacked
            self._is_irrev = is_irrev
            self._is_model_list = True
            self._models_list = [_ensure_diag(m) for m in model]
        else:
            model = _ensure_diag(model)
            is_irrev = isinstance(model, IrrevDiagModel)
            if is_irrev:
                subMatrices = compute_sub_matrices_irrev(model, tree.distanceToParent)
            else:
                subMatrices = compute_sub_matrices(model, tree.distanceToParent)
            rootProb = model.pi
            per_column = False
            self._model = model
            self._is_irrev = is_irrev
            self._is_model_list = False
            self._models_list = None

        self._subMatrices = subMatrices
        self._per_column = per_column
        self._rootProb = rootProb

        U, logNormU, logLike = upward_pass(
            alignment, tree, subMatrices, rootProb, maxChunkSize,
            per_column=per_column,
        )
        self._U = U
        self._logNormU = logNormU
        self._logLike = logLike

        D, logNormD = downward_pass(
            U, logNormU, tree, subMatrices, rootProb, alignment,
            per_column=per_column,
        )

        # Set root's outside vector to the prior pi (downward pass leaves it zero)
        if self._is_model_list:
            root_pi = self._model.pi  # (C, A)
        else:
            root_pi = self._model.pi  # (*H, A)
        D_root = jnp.broadcast_to(
            root_pi[..., None, :] if root_pi.ndim == 1 or not self._is_model_list
            else root_pi,
            D[..., 0, :, :].shape,
        )
        max_val = jnp.maximum(jnp.max(D_root, axis=-1), 1e-300)
        D = D.at[..., 0, :, :].set(D_root / max_val[..., None])
        logNormD = logNormD.at[..., 0, :].set(jnp.log(max_val))

        self._D = D
        self._logNormD = logNormD

    @property
    def log_likelihood(self) -> jnp.ndarray:
        """Per-column log-likelihoods. Shape: (*H, C)."""
        return self._logLike

    def counts(
        self,
        f81_fast_flag: bool = False,
        branch_mask: Optional[Union[str, jnp.ndarray]] = "auto",
    ) -> jnp.ndarray:
        """Expected substitution counts and dwell times per column.

        Reuses stored inside-outside vectors (avoids re-running the DP passes).

        Args:
            f81_fast_flag: if True, use O(CRA^2) F81/JC fast path
            branch_mask: "auto", None, or (*H, R, C) bool array

        Returns:
            (*H, A, A, C) counts tensor
        """
        if self._is_model_list:
            return Counts(
                self._alignment, self._tree, self._models_list,
                maxChunkSize=self._maxChunkSize,
                f81_fast_flag=f81_fast_flag,
                branch_mask=branch_mask,
            )

        model = self._model

        if isinstance(branch_mask, str) and branch_mask == "auto":
            from .components import compute_branch_mask
            A = model.pi.shape[-1]
            branch_mask = compute_branch_mask(
                self._alignment, self._tree.parentIndex, A
            )

        if f81_fast_flag:
            return f81_counts(
                self._U, self._D, self._logNormU, self._logNormD,
                self._logLike, self._tree.distanceToParent, model.pi,
                self._tree.parentIndex, branch_mask=branch_mask,
            )
        elif self._is_irrev:
            assert not f81_fast_flag, "F81 fast path is reversible-only"
            J = compute_J_complex(model.eigenvalues, self._tree.distanceToParent)
            U_tilde, D_tilde = eigenbasis_project_irrev(self._U, self._D, model)
            C = accumulate_C_complex(
                D_tilde, U_tilde, J, self._logNormU, self._logNormD,
                self._logLike, self._tree.parentIndex, branch_mask=branch_mask,
            )
            return back_transform_irrev(C, model)
        else:
            J = compute_J(model.eigenvalues, self._tree.distanceToParent)
            U_tilde, D_tilde = eigenbasis_project(self._U, self._D, model)
            C = accumulate_C(
                D_tilde, U_tilde, J, self._logNormU, self._logNormD,
                self._logLike, self._tree.parentIndex, branch_mask=branch_mask,
            )
            return back_transform(C, model)

    def branch_counts(
        self, f81_fast_flag: bool = False,
    ) -> jnp.ndarray:
        """Per-branch expected substitution counts and dwell times.

        Returns the (A, A) matrix of a posteriori expected transition counts
        and dwell times for the CTMC on each branch, for each column.

        Args:
            f81_fast_flag: if True, use O(CRA^2) F81/JC fast path

        Returns:
            (*H, R, A, A, C) per-branch counts (branch 0 = zeros)
        """
        if self._is_model_list:
            return BranchCounts(
                self._alignment, self._tree, self._models_list,
                maxChunkSize=self._maxChunkSize,
                f81_fast_flag=f81_fast_flag,
            )

        model = self._model

        if f81_fast_flag:
            return f81_counts_per_branch(
                self._U, self._D, self._logNormU, self._logNormD,
                self._logLike, self._tree.distanceToParent, model.pi,
                self._tree.parentIndex,
            )
        elif self._is_irrev:
            J = compute_J_complex(model.eigenvalues, self._tree.distanceToParent)
            U_tilde, D_tilde = eigenbasis_project_irrev(self._U, self._D, model)
            C = accumulate_C_complex_per_branch(
                D_tilde, U_tilde, J, self._logNormU, self._logNormD,
                self._logLike, self._tree.parentIndex,
            )
            return back_transform_irrev_per_branch(C, model)
        else:
            J = compute_J(model.eigenvalues, self._tree.distanceToParent)
            U_tilde, D_tilde = eigenbasis_project(self._U, self._D, model)
            C = accumulate_C_per_branch(
                D_tilde, U_tilde, J, self._logNormU, self._logNormD,
                self._logLike, self._tree.parentIndex,
            )
            return back_transform_per_branch(C, model)

    def node_posterior(self, node: Optional[int] = None) -> jnp.ndarray:
        """Posterior state distribution at node(s).

        For root: P(X_0 = a | data) ∝ pi_a * U(0,c,a)
        For non-root: P(X_n = j | data) ∝ [sum_a D(n,c,a) * M(n,a,j)] * U(n,c,j)

        D(n,a) is indexed by the parent's state a, so we transform through
        the branch matrix M(n) to get the child state j.

        Args:
            node: int node index, or None for all nodes.

        Returns:
            If node is int: (*H, A, C) posterior distribution
            If node is None: (*H, R, A, C) posteriors for all nodes
        """
        if node is not None:
            if node == 0:
                # Root: D[0] = pi (set in __init__), product is pi * U / Z
                U_0 = self._U[..., 0, :, :]
                D_0 = self._D[..., 0, :, :]
                log_scale = (
                    self._logNormU[..., 0, :]
                    + self._logNormD[..., 0, :]
                    - self._logLike
                )
                posterior = U_0 * D_0 * jnp.exp(log_scale)[..., None]
            else:
                # Non-root: transform D through M
                D_n = self._D[..., node, :, :]   # (*H, C, A_parent)
                U_n = self._U[..., node, :, :]   # (*H, C, A_child)
                if self._per_column:
                    M = self._subMatrices[..., node, :, :, :]  # (*H, C, A, A)
                    D_transformed = jnp.einsum('...ca,...caj->...cj', D_n, M)
                else:
                    M = self._subMatrices[..., node, :, :]     # (*H, A, A)
                    D_transformed = jnp.einsum('...ca,...aj->...cj', D_n, M)
                log_scale = (
                    self._logNormU[..., node, :]
                    + self._logNormD[..., node, :]
                    - self._logLike
                )
                posterior = D_transformed * U_n * jnp.exp(log_scale)[..., None]
            posterior = posterior / jnp.sum(posterior, axis=-1, keepdims=True)
            return jnp.moveaxis(posterior, -1, -2)  # (*H, A, C)
        else:
            # All nodes: compute via branch_posterior marginals for non-root,
            # direct formula for root
            *H, R, C, A = self._U.shape
            # Root
            D_0 = self._D[..., 0, :, :]
            U_0 = self._U[..., 0, :, :]
            log_scale_0 = (
                self._logNormU[..., 0, :]
                + self._logNormD[..., 0, :]
                - self._logLike
            )
            root_post = D_0 * U_0 * jnp.exp(log_scale_0)[..., None]
            root_post = root_post / jnp.sum(root_post, axis=-1, keepdims=True)

            # Non-root: D_transformed = D @ M, then * U * scale
            log_scale = (
                self._logNormU + self._logNormD
                - self._logLike[..., None, :]
            )  # (*H, R, C)
            if self._per_column:
                D_transformed = jnp.einsum(
                    '...rca,...rcaj->...rcj', self._D, self._subMatrices
                )
            else:
                D_transformed = jnp.einsum(
                    '...rca,...raj->...rcj', self._D, self._subMatrices
                )
            posterior = D_transformed * self._U * jnp.exp(log_scale)[..., None]
            posterior = posterior / jnp.sum(posterior, axis=-1, keepdims=True)
            # Replace root row with correct root posterior
            posterior = posterior.at[..., 0, :, :].set(root_post)
            return jnp.moveaxis(posterior, -1, -2)  # (*H, R, A, C)

    def branch_posterior(self, node: Optional[int] = None) -> jnp.ndarray:
        """Joint posterior of parent-child states on branch(es).

        P(X_{parent(n)}=i, X_n=j | data, c) ∝ D(n,c,i) * M(n,i,j) * U(n,c,j)

        D(n,a) is indexed by the parent's state a (the outside context at
        branch n), so D(n) is used directly (not D at the parent node).

        Args:
            node: int child node index (must be > 0), or None for all branches.

        Returns:
            If node is int: (*H, A, A, C) where [i,j,c] = P(parent=i, child=j)
            If node is None: (*H, R, A, A, C) for all branches (branch 0 is zeros)
        """
        if node is not None:
            D_n = self._D[..., node, :, :]     # (*H, C, A_parent)
            U_n = self._U[..., node, :, :]     # (*H, C, A_child)

            if self._per_column:
                M = self._subMatrices[..., node, :, :, :]  # (*H, C, A, A)
            else:
                M = self._subMatrices[..., node, :, :]     # (*H, A, A)

            log_scale = (
                self._logNormD[..., node, :]
                + self._logNormU[..., node, :]
                - self._logLike
            )  # (*H, C)
            scale = jnp.exp(log_scale)

            if self._per_column:
                joint = (
                    D_n[..., :, :, None]
                    * M
                    * U_n[..., :, None, :]
                    * scale[..., :, None, None]
                )
            else:
                joint = (
                    D_n[..., :, :, None]
                    * M[..., None, :, :]
                    * U_n[..., :, None, :]
                    * scale[..., :, None, None]
                )

            joint = joint / jnp.sum(joint, axis=(-2, -1), keepdims=True)
            return jnp.moveaxis(joint, -3, -1)  # (*H, A, A, C)
        else:
            # All branches: use D[n] directly (not D[parent])
            log_scale = (
                self._logNormD + self._logNormU
                - self._logLike[..., None, :]
            )  # (*H, R, C)
            scale = jnp.exp(log_scale)

            if self._per_column:
                joint = (
                    self._D[..., :, :, None]
                    * self._subMatrices
                    * self._U[..., :, None, :]
                    * scale[..., :, None, None]
                )
            else:
                joint = (
                    self._D[..., :, :, None]
                    * self._subMatrices[..., None, :, :]
                    * self._U[..., :, None, :]
                    * scale[..., :, None, None]
                )

            joint = joint / jnp.sum(joint, axis=(-2, -1), keepdims=True)
            joint = joint.at[..., 0, :, :, :].set(0.0)
            return jnp.moveaxis(joint, -3, -1)  # (*H, R, A, A, C)
