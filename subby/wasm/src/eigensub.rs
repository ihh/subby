/// Eigensubstitution accumulation (Holmes & Rubin 2002).

use crate::model::{DiagModel, IrrevDiagModel};
use crate::complex::*;

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

// ---- Per-branch variants ----

/// Accumulate eigenbasis counts C_{kl} per branch.
/// Returns (R*A*A*C) flat array with layout result[n*A*A*C + k*A*C + l*C + col].
/// Branch 0 = zeros.
pub fn accumulate_c_per_branch(
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
    let mut c_out = vec![0.0; r * a * a * c];

    for n in 1..r {
        for col in 0..c {
            let log_s = log_norm_d[n * c + col] + log_norm_u[n * c + col] - log_like[col];
            let scale = log_s.exp();
            let base = (n * c + col) * a;

            for k in 0..a {
                for l in 0..a {
                    c_out[n * a * a * c + k * a * c + l * c + col] =
                        d_tilde[base + k] * j[n * a * a + k * a + l] * u_tilde[base + l] * scale;
                }
            }
        }
    }

    c_out
}

/// Transform eigenbasis counts to natural basis, per branch.
/// Takes (R*A*A*C) flat, returns (R*A*A*C) flat.
pub fn back_transform_per_branch(c_eigen: &[f64], model: &DiagModel, r: usize, c: usize) -> Vec<f64> {
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

    // VCV = V C V^T per branch per column
    let mut counts = vec![0.0; r * a * a * c];
    for n in 1..r {
        let br_offset = n * a * a * c;
        for col in 0..c {
            for i in 0..a {
                for j in 0..a {
                    let mut vcv = 0.0;
                    for k in 0..a {
                        for l in 0..a {
                            vcv += v[i * a + k] * c_eigen[br_offset + k * a * c + l * c + col] * v[j * a + l];
                        }
                    }
                    if i == j {
                        counts[br_offset + i * a * c + j * c + col] = vcv;
                    } else {
                        counts[br_offset + i * a * c + j * c + col] = s_mat[i * a + j] * vcv;
                    }
                }
            }
        }
    }

    counts
}

// ---- Complex / irreversible variants ----

/// Compute J^{kl}(T) for complex eigenvalues.
/// eigenvalues_complex: interleaved (2*A). Output: interleaved (2*R*A*A).
pub fn compute_j_complex(eigenvalues_complex: &[f64], distances: &[f64]) -> Vec<f64> {
    let a = eigenvalues_complex.len() / 2;
    let r = distances.len();
    let mut j = vec![0.0; 2 * r * a * a];

    for br in 0..r {
        let t = distances[br];
        for k in 0..a {
            let mu_k = cload(eigenvalues_complex, k);
            let exp_k = cexp(cscale(mu_k, t));
            for l in 0..a {
                let mu_l = cload(eigenvalues_complex, l);
                let diff = csub(mu_k, mu_l);
                let idx = br * a * a + k * a + l;
                if cabs(diff) < 1e-8 {
                    cstore(&mut j, idx, cscale(exp_k, t));
                } else {
                    let exp_l = cexp(cscale(mu_l, t));
                    cstore(&mut j, idx, cdiv(csub(exp_k, exp_l), diff));
                }
            }
        }
    }
    j
}

/// Project inside/outside vectors into eigenbasis for irreversible model.
/// Returns (U_tilde, D_tilde), each interleaved (2*R*C*A).
pub fn eigenbasis_project_irrev(
    u: &[f64],
    d: &[f64],
    model: &IrrevDiagModel,
    r: usize,
    c: usize,
    a: usize,
) -> (Vec<f64>, Vec<f64>) {
    let v = &model.eigenvectors_complex;
    let v_inv = &model.eigenvectors_inv_complex;

    let mut u_tilde = vec![0.0; 2 * r * c * a];
    let mut d_tilde = vec![0.0; 2 * r * c * a];

    for row in 0..r {
        for col in 0..c {
            let base_real = (row * c + col) * a;
            let base_complex = row * c + col;
            for k in 0..a {
                let mut su = CZERO;
                let mut sd = CZERO;
                for b in 0..a {
                    let u_b = (u[base_real + b], 0.0);
                    let d_b = (d[base_real + b], 0.0);
                    // V[b,k] — row-major: v[b*A+k]
                    let v_bk = cload(v, b * a + k);
                    su = cadd(su, cmul(u_b, v_bk));
                    // V_inv[k,b] — row-major: v_inv[k*A+b]
                    let vinv_kb = cload(v_inv, k * a + b);
                    sd = cadd(sd, cmul(d_b, vinv_kb));
                }
                cstore(&mut u_tilde, base_complex * a + k, su);
                cstore(&mut d_tilde, base_complex * a + k, sd);
            }
        }
    }

    (u_tilde, d_tilde)
}

/// Accumulate eigenbasis counts for irreversible model (complex).
/// Returns interleaved (2*A*A*C).
pub fn accumulate_c_complex(
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
    let mut c_out = vec![0.0; 2 * a * a * c];

    for n in 1..r {
        for col in 0..c {
            let log_s = log_norm_d[n * c + col] + log_norm_u[n * c + col] - log_like[col];
            let scale = log_s.exp();
            let base = (n * c + col) * a;

            for k in 0..a {
                for l in 0..a {
                    let dt_k = cload(d_tilde, base + k);
                    let ut_l = cload(u_tilde, base + l);
                    let j_kl = cload(j, n * a * a + k * a + l);
                    let contrib = cscale(cmul(cmul(dt_k, j_kl), ut_l), scale);
                    let idx = k * a * c + l * c + col;
                    let prev = cload(&c_out, idx);
                    cstore(&mut c_out, idx, cadd(prev, contrib));
                }
            }
        }
    }

    c_out
}

/// Transform eigenbasis counts to natural basis for irreversible model.
/// Takes Re() of final result. Returns (A*A*C) real.
pub fn back_transform_irrev(c_eigen: &[f64], model: &IrrevDiagModel, c: usize) -> Vec<f64> {
    let a = model.pi.len();
    let v = &model.eigenvectors_complex;
    let v_inv = &model.eigenvectors_inv_complex;
    let mu = &model.eigenvalues_complex;

    // R = V diag(mu) V^{-1}
    let mut r_mat = vec![0.0; 2 * a * a];
    for i in 0..a {
        for j in 0..a {
            let mut s = CZERO;
            for k in 0..a {
                let v_ik = cload(v, i * a + k);
                let mu_k = cload(mu, k);
                let vinv_kj = cload(v_inv, k * a + j);
                s = cadd(s, cmul(cmul(v_ik, mu_k), vinv_kj));
            }
            cstore(&mut r_mat, i * a + j, s);
        }
    }

    // VCV = V C V^{-1} per column
    let mut counts = vec![0.0; a * a * c];
    for col in 0..c {
        for i in 0..a {
            for j in 0..a {
                let mut vcv = CZERO;
                for k in 0..a {
                    let v_ik = cload(v, i * a + k);
                    for l in 0..a {
                        let c_kl = cload(c_eigen, k * a * c + l * c + col);
                        let vinv_lj = cload(v_inv, l * a + j);
                        vcv = cadd(vcv, cmul(cmul(v_ik, c_kl), vinv_lj));
                    }
                }
                if i == j {
                    counts[i * a * c + j * c + col] = vcv.0; // Re
                } else {
                    let r_ij = cload(&r_mat, i * a + j);
                    counts[i * a * c + j * c + col] = cmul(r_ij, vcv).0; // Re
                }
            }
        }
    }

    counts
}

// ---- Per-branch complex / irreversible variants ----

/// Accumulate eigenbasis counts for irreversible model per branch (complex).
/// Returns interleaved (2*R*A*A*C). Branch 0 = zeros.
pub fn accumulate_c_complex_per_branch(
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
    let mut c_out = vec![0.0; 2 * r * a * a * c];

    for n in 1..r {
        for col in 0..c {
            let log_s = log_norm_d[n * c + col] + log_norm_u[n * c + col] - log_like[col];
            let scale = log_s.exp();
            let base = (n * c + col) * a;

            for k in 0..a {
                for l in 0..a {
                    let dt_k = cload(d_tilde, base + k);
                    let ut_l = cload(u_tilde, base + l);
                    let j_kl = cload(j, n * a * a + k * a + l);
                    let contrib = cscale(cmul(cmul(dt_k, j_kl), ut_l), scale);
                    let idx = n * a * a * c + k * a * c + l * c + col;
                    cstore(&mut c_out, idx, contrib);
                }
            }
        }
    }

    c_out
}

/// Transform eigenbasis counts to natural basis for irreversible model, per branch.
/// Takes interleaved (2*R*A*A*C), returns (R*A*A*C) real.
pub fn back_transform_irrev_per_branch(c_eigen: &[f64], model: &IrrevDiagModel, r: usize, c: usize) -> Vec<f64> {
    let a = model.pi.len();
    let v = &model.eigenvectors_complex;
    let v_inv = &model.eigenvectors_inv_complex;
    let mu = &model.eigenvalues_complex;

    // R = V diag(mu) V^{-1}
    let mut r_mat = vec![0.0; 2 * a * a];
    for i in 0..a {
        for j in 0..a {
            let mut s = CZERO;
            for k in 0..a {
                let v_ik = cload(v, i * a + k);
                let mu_k = cload(mu, k);
                let vinv_kj = cload(v_inv, k * a + j);
                s = cadd(s, cmul(cmul(v_ik, mu_k), vinv_kj));
            }
            cstore(&mut r_mat, i * a + j, s);
        }
    }

    // VCV = V C V^{-1} per branch per column
    let mut counts = vec![0.0; r * a * a * c];
    for n in 1..r {
        let br_offset = n * a * a * c;
        for col in 0..c {
            for i in 0..a {
                for j in 0..a {
                    let mut vcv = CZERO;
                    for k in 0..a {
                        let v_ik = cload(v, i * a + k);
                        for l in 0..a {
                            let c_kl = cload(c_eigen, br_offset + k * a * c + l * c + col);
                            let vinv_lj = cload(v_inv, l * a + j);
                            vcv = cadd(vcv, cmul(cmul(v_ik, c_kl), vinv_lj));
                        }
                    }
                    if i == j {
                        counts[br_offset + i * a * c + j * c + col] = vcv.0; // Re
                    } else {
                        let r_ij = cload(&r_mat, i * a + j);
                        counts[br_offset + i * a * c + j * c + col] = cmul(r_ij, vcv).0; // Re
                    }
                }
            }
        }
    }

    counts
}
