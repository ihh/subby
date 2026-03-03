/// Downward pass (outside algorithm).

use crate::tree::children_of;
use crate::token::token_to_likelihood;

/// Compute outside (D) vectors for all nodes.
///
/// Returns (D, logNormD):
///   D: (R*C*A) flat f64
///   logNormD: (R*C) flat f64
pub fn downward_pass(
    u: &[f64],           // (R*C*A) inside vectors
    log_norm_u: &[f64],  // (R*C)
    parent_index: &[i32],
    sub_matrices: &[f64], // (R*A*A)
    root_prob: &[f64],    // (A,)
    alignment: &[i32],    // (R*C)
    r: usize,
    c: usize,
    a: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (_, _, sibling) = children_of(parent_index);
    let obs_like = token_to_likelihood(alignment, r, c, a);

    let mut d = vec![0.0; r * c * a];
    let mut log_norm_d = vec![0.0; r * c];

    // Preorder: nodes 1..R-1
    for n in 1..r {
        let p = parent_index[n] as usize;
        let sib = sibling[n] as usize;
        let sib_m_base = sib * a * a;
        let par_m_base = p * a * a;

        for col in 0..c {
            let sib_u_base = (sib * c + col) * a;
            let par_d_base = (p * c + col) * a;
            let par_obs_base = (p * c + col) * a;

            let mut d_raw = vec![0.0; a];
            for aa in 0..a {
                // Sibling contribution
                let mut sib_contrib = 0.0;
                for j in 0..a {
                    sib_contrib += sub_matrices[sib_m_base + aa * a + j] * u[sib_u_base + j];
                }

                // Parent contribution
                let parent_contrib;
                if p == 0 {
                    parent_contrib = root_prob[aa];
                } else {
                    let mut pc = 0.0;
                    for i in 0..a {
                        pc += d[par_d_base + i] * sub_matrices[par_m_base + i * a + aa];
                    }
                    parent_contrib = pc;
                }

                d_raw[aa] = sib_contrib * parent_contrib * obs_like[par_obs_base + aa];
            }

            // logNormD
            let log_norm_sib = log_norm_u[sib * c + col];
            let log_norm_prior = if p == 0 { 0.0 } else { log_norm_d[p * c + col] };
            let accumulated = log_norm_sib + log_norm_prior;

            // Rescale
            let mut max_val = d_raw[0];
            for aa in 1..a {
                if d_raw[aa] > max_val { max_val = d_raw[aa]; }
            }
            if max_val < 1e-300 { max_val = 1e-300; }

            let node_d_base = (n * c + col) * a;
            for aa in 0..a {
                d[node_d_base + aa] = d_raw[aa] / max_val;
            }
            log_norm_d[n * c + col] = accumulated + max_val.ln();
        }
    }

    (d, log_norm_d)
}
