import jax.numpy as jnp


def mixture_posterior(log_likes, log_weights):
    """Compute posterior probabilities over mixture components.

    P(component k | column c) = softmax_k(log_likes[k,c] + log_weights[k])

    Args:
        log_likes: (K, C) log-likelihoods per component per column
        log_weights: (K,) log prior weights

    Returns:
        (K, C) posterior probabilities (sum to 1 over K for each column)
    """
    log_joint = log_likes + log_weights[:, None]  # (K, C)
    # Softmax over K dimension
    log_joint_max = jnp.max(log_joint, axis=0, keepdims=True)
    log_joint_shifted = log_joint - log_joint_max
    posteriors = jnp.exp(log_joint_shifted) / jnp.sum(
        jnp.exp(log_joint_shifted), axis=0, keepdims=True
    )
    return posteriors
