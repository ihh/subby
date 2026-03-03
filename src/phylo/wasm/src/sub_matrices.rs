/// Compute substitution probability matrices M(t) from eigendecomposition.

use crate::model::DiagModel;

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
