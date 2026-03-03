/// Upward pass (Felsenstein pruning).

use crate::token::token_to_likelihood;

/// Compute inside (U) vectors for all nodes.
///
/// Returns (U, logNormU, logLike):
///   U: (R*C*A) flat f64
///   logNormU: (R*C) flat f64
///   logLike: (C,) f64
pub fn upward_pass(
    alignment: &[i32],
    parent_index: &[i32],
    sub_matrices: &[f64],  // (R*A*A)
    root_prob: &[f64],     // (A,)
    r: usize,
    c: usize,
    a: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut u = token_to_likelihood(alignment, r, c, a);
    let mut log_norm_u = vec![0.0; r * c];

    // Postorder: children R-1 down to 1
    for n in (1..r).rev() {
        let p = parent_index[n] as usize;
        let m_base = n * a * a;

        for col in 0..c {
            // child_contrib[b] = sum_j M[b,j] * U[n,col,j]
            let child_base = (n * c + col) * a;
            let parent_base = (p * c + col) * a;

            for b in 0..a {
                let mut s = 0.0;
                for j in 0..a {
                    s += sub_matrices[m_base + b * a + j] * u[child_base + j];
                }
                u[parent_base + b] *= s;
            }

            // Rescale
            let mut max_val = u[parent_base];
            for b in 1..a {
                if u[parent_base + b] > max_val {
                    max_val = u[parent_base + b];
                }
            }
            if max_val < 1e-300 {
                max_val = 1e-300;
            }
            for b in 0..a {
                u[parent_base + b] /= max_val;
            }
            let log_rescale = max_val.ln();
            log_norm_u[p * c + col] += log_norm_u[n * c + col] + log_rescale;
        }
    }

    // Log-likelihood
    let mut log_like = vec![0.0; c];
    for col in 0..c {
        let mut s = 0.0;
        for aa in 0..a {
            s += root_prob[aa] * u[(0 * c + col) * a + aa];
        }
        log_like[col] = log_norm_u[0 * c + col] + s.ln();
    }

    (u, log_norm_u, log_like)
}
