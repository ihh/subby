/// Mixture posterior computation.

/// Compute posterior probabilities over mixture components.
/// P(k | c) = softmax_k(log_likes[k*c + c] + log_weights[k])
///
/// Returns (K*C) flat f64 array.
pub fn mixture_posterior(
    log_likes: &[f64],   // (K*C) flat
    log_weights: &[f64],  // (K,)
    k: usize,
    c: usize,
) -> Vec<f64> {
    let mut posteriors = vec![0.0; k * c];

    for col in 0..c {
        let mut max_val = f64::NEG_INFINITY;
        for comp in 0..k {
            let lj = log_likes[comp * c + col] + log_weights[comp];
            if lj > max_val { max_val = lj; }
        }

        let mut denom = 0.0;
        for comp in 0..k {
            let val = (log_likes[comp * c + col] + log_weights[comp] - max_val).exp();
            posteriors[comp * c + col] = val;
            denom += val;
        }

        for comp in 0..k {
            posteriors[comp * c + col] /= denom;
        }
    }

    posteriors
}
