/// Eigensubstitution accumulation (Holmes & Rubin 2002).

use crate::model::DiagModel;

/// Compute J^{kl}(T) interaction matrix.
/// Returns flat (R*A*A) f64 array.
pub fn compute_j(eigenvalues: &[f64], distances: &[f64]) -> Vec<f64> {
    let a = eigenvalues.len();
    let r = distances.len();
    let mut j = vec![0.0; r * a * a];

    for br in 0..r {
        let t = distances[br];
        for k in 0..a {
            let mu_k = eigenvalues[k];
            let exp_k = (mu_k * t).exp();
            for l in 0..a {
                let mu_l = eigenvalues[l];
                let diff = mu_k - mu_l;
                if diff.abs() < 1e-8 {
                    j[br * a * a + k * a + l] = t * exp_k;
                } else {
                    j[br * a * a + k * a + l] = (exp_k - (mu_l * t).exp()) / diff;
                }
            }
        }
    }
    j
}

/// Project inside/outside vectors into eigenbasis.
/// Returns (U_tilde, D_tilde), each (R*C*A) flat.
pub fn eigenbasis_project(
    u: &[f64],
    d: &[f64],
    model: &DiagModel,
    r: usize,
    c: usize,
    a: usize,
) -> (Vec<f64>, Vec<f64>) {
    let v = &model.eigenvectors;
    let pi = &model.pi;
    let sqrt_pi: Vec<f64> = pi.iter().map(|&p| p.sqrt()).collect();
    let inv_sqrt_pi: Vec<f64> = sqrt_pi.iter().map(|&s| 1.0 / s).collect();

    let mut u_tilde = vec![0.0; r * c * a];
    let mut d_tilde = vec![0.0; r * c * a];

    for row in 0..r {
        for col in 0..c {
            let base = (row * c + col) * a;
            for k in 0..a {
                let mut su = 0.0;
                let mut sd = 0.0;
                for b in 0..a {
                    su += u[base + b] * v[b * a + k] * sqrt_pi[b];
                    sd += d[base + b] * v[b * a + k] * inv_sqrt_pi[b];
                }
                u_tilde[base + k] = su;
                d_tilde[base + k] = sd;
            }
        }
    }

    (u_tilde, d_tilde)
}

/// Accumulate eigenbasis counts C_{kl} over branches.
/// Returns (A*A*C) flat array.
pub fn accumulate_c(
    d_tilde: &[f64],
    u_tilde: &[f64],
    j: &[f64],
    log_norm_u: &[f64],
    log_norm_d: &[f64],
    log_like: &[f64],
    _parent_index: &[i32],
    r: usize,
    c: usize,
    a: usize,
) -> Vec<f64> {
    let mut c_out = vec![0.0; a * a * c];

    for n in 1..r {
        for col in 0..c {
            let log_s = log_norm_d[n * c + col] + log_norm_u[n * c + col] - log_like[col];
            let scale = log_s.exp();
            let base = (n * c + col) * a;

            for k in 0..a {
                for l in 0..a {
                    c_out[k * a * c + l * c + col] +=
                        d_tilde[base + k] * j[n * a * a + k * a + l] * u_tilde[base + l] * scale;
                }
            }
        }
    }

    c_out
}

/// Transform eigenbasis counts to natural basis.
/// Returns (A*A*C) flat array: diag=dwell, off-diag=subs.
pub fn back_transform(c_eigen: &[f64], model: &DiagModel, c: usize) -> Vec<f64> {
    let a = model.pi.len();
    let v = &model.eigenvectors;
    let mu = &model.eigenvalues;

    // S = V diag(mu) V^T
    let mut s_mat = vec![0.0; a * a];
    for i in 0..a {
        for j in 0..a {
            let mut s = 0.0;
            for k in 0..a {
                s += v[i * a + k] * mu[k] * v[j * a + k];
            }
            s_mat[i * a + j] = s;
        }
    }

    // VCV = V C V^T per column
    let mut counts = vec![0.0; a * a * c];
    for col in 0..c {
        for i in 0..a {
            for j in 0..a {
                let mut vcv = 0.0;
                for k in 0..a {
                    for l in 0..a {
                        vcv += v[i * a + k] * c_eigen[k * a * c + l * c + col] * v[j * a + l];
                    }
                }
                if i == j {
                    counts[i * a * c + j * c + col] = vcv;
                } else {
                    counts[i * a * c + j * c + col] = s_mat[i * a + j] * vcv;
                }
            }
        }
    }

    counts
}
