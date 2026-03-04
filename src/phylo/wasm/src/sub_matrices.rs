/// Compute substitution probability matrices M(t) from eigendecomposition.

use crate::model::{DiagModel, IrrevDiagModel};
use crate::complex::*;

/// M_ij(t) = sqrt(pi_j/pi_i) * sum_k v_ik * exp(mu_k * t) * v_jk
///
/// Returns flat (R*A*A) f64 array, row-major.
pub fn compute_sub_matrices(model: &DiagModel, distances: &[f64]) -> Vec<f64> {
    let r = distances.len();
    let a = model.pi.len();
    let v = &model.eigenvectors;
    let mu = &model.eigenvalues;
    let pi = &model.pi;

    let sqrt_pi: Vec<f64> = pi.iter().map(|&p| p.sqrt()).collect();
    let inv_sqrt_pi: Vec<f64> = sqrt_pi.iter().map(|&s| 1.0 / s).collect();

    let mut m = vec![0.0; r * a * a];
    for br in 0..r {
        let t = distances[br];
        for i in 0..a {
            for j in 0..a {
                let mut s = 0.0;
                for k in 0..a {
                    s += v[i * a + k] * (mu[k] * t).exp() * v[j * a + k];
                }
                m[br * a * a + i * a + j] = inv_sqrt_pi[i] * sqrt_pi[j] * s;
            }
        }
    }
    m
}

/// Compute substitution probability matrices for irreversible model.
/// M_ij(t) = Re(sum_k V_ik * exp(mu_k * t) * V_inv_kj)
/// Returns flat (R*A*A) real f64 array.
pub fn compute_sub_matrices_irrev(model: &IrrevDiagModel, distances: &[f64]) -> Vec<f64> {
    let r = distances.len();
    let a = model.pi.len();
    let v = &model.eigenvectors_complex;
    let v_inv = &model.eigenvectors_inv_complex;
    let mu = &model.eigenvalues_complex;

    let mut m = vec![0.0; r * a * a];
    for br in 0..r {
        let t = distances[br];
        for i in 0..a {
            for j in 0..a {
                let mut s = CZERO;
                for k in 0..a {
                    let v_ik = cload(v, i * a + k);
                    let mu_k = cload(mu, k);
                    let exp_k = cexp(cscale(mu_k, t));
                    let vinv_kj = cload(v_inv, k * a + j);
                    s = cadd(s, cmul(cmul(v_ik, exp_k), vinv_kj));
                }
                m[br * a * a + i * a + j] = s.0; // Re
            }
        }
    }
    m
}
