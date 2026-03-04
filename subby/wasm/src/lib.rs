//! Phylogenetic sufficient statistics library.
//!
//! Single crate, dual target: cdylib (WASM) + rlib (native Rust tests).
//! Feature flag `wasm` gates wasm-bindgen dependencies.

pub mod tree;
pub mod model;
pub mod token;
pub mod sub_matrices;
pub mod pruning;
pub mod outside;
pub mod eigensub;
pub mod f81_fast;
pub mod mixture;
pub mod branch_mask;
pub mod complex;

use model::{DiagModel, IrrevDiagModel};

/// Compute per-column log-likelihoods.
pub fn log_like(
    alignment: &[i32],
    parent_index: &[i32],
    distances: &[f64],
    model: &DiagModel,
) -> Vec<f64> {
    let r = parent_index.len();
    let a = model.pi.len();
    let c = alignment.len() / r;

    let sub_mats = sub_matrices::compute_sub_matrices(model, distances);
    let (_, _, ll) = pruning::upward_pass(alignment, parent_index, &sub_mats, &model.pi, r, c, a);
    ll
}

/// Compute expected substitution counts and dwell times.
pub fn counts(
    alignment: &[i32],
    parent_index: &[i32],
    distances: &[f64],
    model: &DiagModel,
    f81_fast_flag: bool,
) -> Vec<f64> {
    let r = parent_index.len();
    let a = model.pi.len();
    let c = alignment.len() / r;

    let sub_mats = sub_matrices::compute_sub_matrices(model, distances);
    let (u, log_norm_u, ll) = pruning::upward_pass(
        alignment, parent_index, &sub_mats, &model.pi, r, c, a,
    );
    let (d, log_norm_d) = outside::downward_pass(
        &u, &log_norm_u, parent_index, &sub_mats, &model.pi, alignment, r, c, a,
    );

    if f81_fast_flag {
        f81_fast::f81_counts(
            &u, &d, &log_norm_u, &log_norm_d, &ll,
            distances, &model.pi, parent_index, r, c, a,
        )
    } else {
        let j = eigensub::compute_j(&model.eigenvalues, distances);
        let (u_tilde, d_tilde) = eigensub::eigenbasis_project(&u, &d, model, r, c, a);
        let c_eigen = eigensub::accumulate_c(
            &d_tilde, &u_tilde, &j, &log_norm_u, &log_norm_d, &ll,
            parent_index, r, c, a,
        );
        eigensub::back_transform(&c_eigen, model, c)
    }
}

/// Compute expected substitution counts and dwell times (irreversible model).
pub fn counts_irrev(
    alignment: &[i32],
    parent_index: &[i32],
    distances: &[f64],
    model: &IrrevDiagModel,
) -> Vec<f64> {
    let r = parent_index.len();
    let a = model.pi.len();
    let c = alignment.len() / r;

    let sub_mats = sub_matrices::compute_sub_matrices_irrev(model, distances);
    let (u, log_norm_u, ll) = pruning::upward_pass(
        alignment, parent_index, &sub_mats, &model.pi, r, c, a,
    );
    let (d, log_norm_d) = outside::downward_pass(
        &u, &log_norm_u, parent_index, &sub_mats, &model.pi, alignment, r, c, a,
    );

    let j = eigensub::compute_j_complex(&model.eigenvalues_complex, distances);
    let (u_tilde, d_tilde) = eigensub::eigenbasis_project_irrev(&u, &d, model, r, c, a);
    let c_eigen = eigensub::accumulate_c_complex(
        &d_tilde, &u_tilde, &j, &log_norm_u, &log_norm_d, &ll,
        parent_index, r, c, a,
    );
    eigensub::back_transform_irrev(&c_eigen, model, c)
}

/// Compute posterior root state distribution.
pub fn root_prob(
    alignment: &[i32],
    parent_index: &[i32],
    distances: &[f64],
    model: &DiagModel,
) -> Vec<f64> {
    let r = parent_index.len();
    let a = model.pi.len();
    let c = alignment.len() / r;

    let sub_mats = sub_matrices::compute_sub_matrices(model, distances);
    let (u, log_norm_u, ll) = pruning::upward_pass(
        alignment, parent_index, &sub_mats, &model.pi, r, c, a,
    );

    let mut q = vec![0.0; a * c];
    for col in 0..c {
        let log_scale = log_norm_u[col] - ll[col];
        let scale = log_scale.exp();
        for aa in 0..a {
            q[aa * c + col] = model.pi[aa] * u[col * a + aa] * scale;
        }
    }
    q
}

/// Compute mixture posteriors.
pub fn mixture_posterior_full(
    alignment: &[i32],
    parent_index: &[i32],
    distances: &[f64],
    models: &[DiagModel],
    log_weights: &[f64],
) -> Vec<f64> {
    let k = models.len();
    let r = parent_index.len();
    let c = alignment.len() / r;

    let mut log_likes = vec![0.0; k * c];
    for comp in 0..k {
        let ll = log_like(alignment, parent_index, distances, &models[comp]);
        for col in 0..c {
            log_likes[comp * c + col] = ll[col];
        }
    }

    mixture::mixture_posterior(&log_likes, log_weights, k, c)
}

// ---- WASM bindings ----
#[cfg(feature = "wasm")]
mod wasm_api {
    use wasm_bindgen::prelude::*;
    use crate::model;

    #[wasm_bindgen]
    pub fn wasm_log_like(
        alignment: &[i32],
        parent_index: &[i32],
        distances: &[f64],
        eigenvalues: &[f64],
        eigenvectors: &[f64],
        pi: &[f64],
    ) -> Vec<f64> {
        let m = model::DiagModel {
            eigenvalues: eigenvalues.to_vec(),
            eigenvectors: eigenvectors.to_vec(),
            pi: pi.to_vec(),
        };
        crate::log_like(alignment, parent_index, distances, &m)
    }

    #[wasm_bindgen]
    pub fn wasm_counts(
        alignment: &[i32],
        parent_index: &[i32],
        distances: &[f64],
        eigenvalues: &[f64],
        eigenvectors: &[f64],
        pi: &[f64],
        f81_fast: bool,
    ) -> Vec<f64> {
        let m = model::DiagModel {
            eigenvalues: eigenvalues.to_vec(),
            eigenvectors: eigenvectors.to_vec(),
            pi: pi.to_vec(),
        };
        crate::counts(alignment, parent_index, distances, &m, f81_fast)
    }

    #[wasm_bindgen]
    pub fn wasm_counts_irrev(
        alignment: &[i32],
        parent_index: &[i32],
        distances: &[f64],
        eigenvalues_complex: &[f64],
        eigenvectors_complex: &[f64],
        eigenvectors_inv_complex: &[f64],
        pi: &[f64],
    ) -> Vec<f64> {
        let m = crate::model::IrrevDiagModel {
            eigenvalues_complex: eigenvalues_complex.to_vec(),
            eigenvectors_complex: eigenvectors_complex.to_vec(),
            eigenvectors_inv_complex: eigenvectors_inv_complex.to_vec(),
            pi: pi.to_vec(),
        };
        crate::counts_irrev(alignment, parent_index, distances, &m)
    }

    #[wasm_bindgen]
    pub fn wasm_root_prob(
        alignment: &[i32],
        parent_index: &[i32],
        distances: &[f64],
        eigenvalues: &[f64],
        eigenvectors: &[f64],
        pi: &[f64],
    ) -> Vec<f64> {
        let m = model::DiagModel {
            eigenvalues: eigenvalues.to_vec(),
            eigenvectors: eigenvectors.to_vec(),
            pi: pi.to_vec(),
        };
        crate::root_prob(alignment, parent_index, distances, &m)
    }

    #[wasm_bindgen]
    pub fn wasm_branch_mask(
        alignment: &[i32],
        parent_index: &[i32],
        a: usize,
    ) -> Vec<u8> {
        let r = parent_index.len();
        let c = alignment.len() / r;
        crate::branch_mask::compute_branch_mask(alignment, parent_index, a, r, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tree_5() -> (Vec<i32>, Vec<f64>) {
        let parent_index = vec![-1, 0, 0, 1, 1];
        let distances = vec![0.0, 0.1, 0.2, 0.15, 0.25];
        (parent_index, distances)
    }

    #[test]
    fn test_log_like_jc4() {
        let (pi, dist) = make_tree_5();
        let alignment = vec![0, 1, 2, 3, 2, 3, 0, 1, 1, 0, 3, 2, 3, 2, 1, 0, 0, 0, 0, 0];
        let model = model::jukes_cantor_model(4);
        let ll = log_like(&alignment, &pi, &dist, &model);
        assert_eq!(ll.len(), 4);
        for v in &ll {
            assert!(v.is_finite());
            assert!(*v <= 0.0);
        }
    }

    #[test]
    fn test_counts_shape() {
        let (pi, dist) = make_tree_5();
        let alignment = vec![0, 1, 2, 3, 2, 3, 0, 1, 1, 0, 3, 2, 3, 2, 1, 0, 0, 0, 0, 0];
        let model = model::jukes_cantor_model(4);
        let c = counts(&alignment, &pi, &dist, &model, false);
        assert_eq!(c.len(), 4 * 4 * 4); // A*A*C
    }

    #[test]
    fn test_root_prob_sums_to_one() {
        let (pi, dist) = make_tree_5();
        let alignment = vec![0, 1, 2, 3, 2, 3, 0, 1, 1, 0, 3, 2, 3, 2, 1, 0, 0, 0, 0, 0];
        let model = model::jukes_cantor_model(4);
        let rp = root_prob(&alignment, &pi, &dist, &model);
        let ncols = 4;
        for col in 0..ncols {
            let mut sum = 0.0;
            for a in 0..4 {
                sum += rp[a * ncols + col];
            }
            assert!((sum - 1.0).abs() < 1e-8, "Sum = {} for col {}", sum, col);
        }
    }

    #[test]
    fn test_branch_mask() {
        let (parent_index, _) = make_tree_5();
        let alignment = vec![0, 1, 2, 3, 2, 3, 0, 1, 1, 0, 3, 2, 3, 2, 1, 0, 0, 0, 0, 0];
        let mask = branch_mask::compute_branch_mask(&alignment, &parent_index, 4, 5, 4);
        assert_eq!(mask.len(), 5 * 4);
        // Root branch always inactive
        for col in 0..4 {
            assert_eq!(mask[0 * 4 + col], 0);
        }
    }
}
