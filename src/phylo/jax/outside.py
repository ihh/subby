import jax
import jax.numpy as jnp
from ._utils import children_of, token_to_likelihood


def downward_pass(U, logNormU, tree, subMatrices, rootProb, alignment):
    """Compute outside (D) vectors for all nodes.

    D^(n)_a = [sum_j M_{aj}(t_{p,s}) U^(s)_j] * L(x_p|a) *
              [pi_a if p=root else sum_i D^(p)_i M_{ia}(t_{g,p})]

    where s is n's sibling, p is n's parent, g is p's parent,
    and L(x_p|a) is the observation likelihood at the parent node.

    Args:
        U: (*H, R, C, A) inside vectors from upward_pass (rescaled)
        logNormU: (*H, R, C) per-node subtree log-normalizers
        tree: Tree(parentIndex, distanceToParent)
        subMatrices: (*H, R, A, A) substitution probability matrices
        rootProb: (*H, A) equilibrium frequencies
        alignment: (R, C) int32 token alignment

    Returns:
        D: (*H, R, C, A) outside vectors per node (rescaled)
        logNormD: (*H, R, C) per-node outside log-normalizers
    """
    parentIndex = tree.parentIndex
    *H, R, C, A = U.shape

    _, _, sibling = children_of(parentIndex)

    # Observation likelihoods per node: (R, C, A)
    obs_like = token_to_likelihood(alignment, A)

    D = jnp.zeros((*H, R, C, A))
    logNormD = jnp.zeros((*H, R, C))

    # Process nodes 1, 2, ..., R-1 in preorder
    preorder_nodes = jnp.arange(1, R, dtype=jnp.int32)
    parent_of = parentIndex[1:]
    sibling_of = sibling[1:]

    init = (D, logNormD)

    (D, logNormD), _ = jax.lax.scan(
        lambda carry, xs: _downward_step(
            carry, xs, U, logNormU, subMatrices, rootProb, parentIndex, obs_like
        ),
        init,
        (preorder_nodes, parent_of, sibling_of),
    )

    return D, logNormD


def _downward_step(carry, xs, U, logNormU, subMatrices, rootProb, parentIndex, obs_like):
    """Single step of the downward (outside) scan."""
    D, logNormD = carry
    node, parent, sib = xs

    # Sibling contribution: sum_j M_{aj}(t_{p,sib}) * U^(sib)_j
    sib_M = subMatrices[..., sib, :, :]  # (*H, A, A)
    sib_U = U[..., sib, :, :]            # (*H, C, A)
    sib_contrib = jnp.einsum('...ij,...cj->...ci', sib_M, sib_U)  # (*H, C, A)

    # Parent contribution: pi_a if parent is root, else sum_i D^(p)_i M_{ia}(t_{g,p})
    parent_is_root = (parent == 0)

    # Non-root case: sum_i D^(p)_i M_{ia}(t_{g,p})
    parent_M = subMatrices[..., parent, :, :]  # (*H, A, A)
    parent_D = D[..., parent, :, :]            # (*H, C, A)
    prop_down = jnp.einsum('...ci,...ia->...ca', parent_D, parent_M)  # (*H, C, A)

    # Root case: just rootProb
    root_contrib = jnp.broadcast_to(rootProb[..., None, :], prop_down.shape)

    parent_contrib = jnp.where(parent_is_root, root_contrib, prop_down)

    # Include parent's observation likelihood L(x_p|a)
    parent_obs = obs_like[parent]  # (C, A) — broadcast to (*H, C, A)
    parent_contrib = parent_contrib * parent_obs

    # D^(n) = sib_contrib * parent_contrib
    D_raw = sib_contrib * parent_contrib  # (*H, C, A)

    # Compute logNormD for this node
    log_norm_from_sib = logNormU[..., sib, :]         # (*H, C)
    log_norm_from_parent = logNormD[..., parent, :]    # (*H, C)
    log_norm_prior = jnp.where(parent_is_root, 0.0, log_norm_from_parent)
    accumulated = log_norm_from_sib + log_norm_prior

    # Rescale
    maxD = jnp.max(D_raw, axis=-1)
    maxD = jnp.maximum(maxD, 1e-300)
    D_rescaled = D_raw / maxD[..., None]
    log_rescale = jnp.log(maxD)

    logNormD_node = accumulated + log_rescale

    D = D.at[..., node, :, :].set(D_rescaled)
    logNormD = logNormD.at[..., node, :].set(logNormD_node)

    return (D, logNormD), None
