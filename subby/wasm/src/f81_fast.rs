/// F81/JC fast path: O(CRA^2) direct computation.

/// Compute expected counts using closed-form F81 integrals.
/// Returns (A*A*C) flat array: diag=dwell, off-diag=subs.
pub fn f81_counts(
    u: &[f64],
    d: &[f64],
    log_norm_u: &[f64],
    log_norm_d: &[f64],
    log_like: &[f64],
    distances: &[f64],
    pi: &[f64],
    _parent_index: &[i32],
    r: usize,
    c: usize,
    a: usize,
) -> Vec<f64> {
    let pi_sq_sum: f64 = pi.iter().map(|&p| p * p).sum();
    let mu = 1.0 / (1.0 - pi_sq_sum);

    let mut result = vec![0.0; a * a * c];

    for n in 1..r {
        let t = distances[n];
        let mu_t = mu * t;
        let e_t = (-mu_t).exp();
        let p = 1.0 - e_t;

        let alpha_n = t * e_t;
        let beta_n = p / mu - t * e_t;
        let gamma_n = t * (1.0 + e_t) - 2.0 * p / mu;

        for col in 0..c {
            let log_s = log_norm_d[n * c + col] + log_norm_u[n * c + col] - log_like[col];
            let scale = log_s.exp();

            let base = (n * c + col) * a;

            // piU = sum_b pi[b] * U[n,col,b]
            let mut pi_u = 0.0;
            for b in 0..a {
                pi_u += pi[b] * u[base + b];
            }

            // Dsum = sum_a D[n,col,a] * scale
            let mut d_sum = 0.0;
            for aa in 0..a {
                d_sum += d[base + aa] * scale;
            }

            for i in 0..a {
                let d_i_scaled = d[base + i] * scale;
                for j in 0..a {
                    let i_sum = alpha_n * d_i_scaled * u[base + j]
                        + beta_n * (d_i_scaled * pi_u + pi[i] * d_sum * u[base + j])
                        + gamma_n * pi[i] * d_sum * pi_u;

                    if i == j {
                        result[i * a * c + j * c + col] += i_sum;
                    } else {
                        result[i * a * c + j * c + col] += mu * pi[j] * i_sum;
                    }
                }
            }
        }
    }

    result
}
