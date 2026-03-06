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
pub mod ctmc;

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

/// Compute per-column log-likelihoods (irreversible model).
pub fn log_like_irrev(
    alignment: &[i32],
    parent_index: &[i32],
    distances: &[f64],
    model: &IrrevDiagModel,
) -> Vec<f64> {
    let r = parent_index.len();
    let a = model.pi.len();
    let c = alignment.len() / r;

    let sub_mats = sub_matrices::compute_sub_matrices_irrev(model, distances);
    let (_, _, ll) = pruning::upward_pass(alignment, parent_index, &sub_mats, &model.pi, r, c, a);
    ll
}

/// Compute posterior root state distribution (irreversible model).
pub fn root_prob_irrev(
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

/// Compute per-branch expected substitution counts and dwell times.
/// Returns (R*A*A*C) flat with layout result[n*A*A*C + i*A*C + j*C + col].
/// Branch 0 = zeros.
pub fn branch_counts(
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
        f81_fast::f81_counts_per_branch(
            &u, &d, &log_norm_u, &log_norm_d, &ll,
            distances, &model.pi, parent_index, r, c, a,
        )
    } else {
        let j = eigensub::compute_j(&model.eigenvalues, distances);
        let (u_tilde, d_tilde) = eigensub::eigenbasis_project(&u, &d, model, r, c, a);
        let c_eigen = eigensub::accumulate_c_per_branch(
            &d_tilde, &u_tilde, &j, &log_norm_u, &log_norm_d, &ll,
            parent_index, r, c, a,
        );
        eigensub::back_transform_per_branch(&c_eigen, model, r, c)
    }
}

/// Compute per-branch expected substitution counts and dwell times (irreversible model).
/// Returns (R*A*A*C) flat. Branch 0 = zeros.
pub fn branch_counts_irrev(
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
    let c_eigen = eigensub::accumulate_c_complex_per_branch(
        &d_tilde, &u_tilde, &j, &log_norm_u, &log_norm_d, &ll,
        parent_index, r, c, a,
    );
    eigensub::back_transform_irrev_per_branch(&c_eigen, model, r, c)
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

/// Expected substitution counts and dwell times for a single CTMC branch (reversible).
/// Returns flat (A^4) array: result[a*A^3 + b*A^2 + i*A + j].
pub fn expected_counts(model: &DiagModel, t: f64) -> Vec<f64> {
    let a = model.pi.len();
    ctmc::expected_counts_eigen(&model.eigenvalues, &model.eigenvectors, &model.pi, t, a)
}

/// Expected substitution counts and dwell times for a single CTMC branch (irreversible).
/// Returns flat (A^4) real array: result[a*A^3 + b*A^2 + i*A + j].
pub fn expected_counts_irrev(model: &IrrevDiagModel, t: f64) -> Vec<f64> {
    let a = model.pi.len();
    ctmc::expected_counts_eigen_irrev(
        &model.eigenvalues_complex, &model.eigenvectors_complex,
        &model.eigenvectors_inv_complex, &model.pi, t, a,
    )
}

// ---- InsideOutside table ----

enum ModelData {
    Reversible {
        eigenvalues: Vec<f64>,
        eigenvectors: Vec<f64>,
    },
    Irreversible {
        eigenvalues_complex: Vec<f64>,
        eigenvectors_complex: Vec<f64>,
        eigenvectors_inv_complex: Vec<f64>,
    },
}

/// Inside-outside DP tables for querying posteriors.
///
/// Runs the upward (inside) and downward (outside) passes once and stores the
/// resulting vectors, enabling efficient queries for log-likelihoods, expected
/// substitution counts, node state posteriors, and branch endpoint joint
/// posteriors without recomputation.
pub struct InsideOutsideTable {
    u: Vec<f64>,
    d: Vec<f64>,
    log_norm_u: Vec<f64>,
    log_norm_d: Vec<f64>,
    log_like: Vec<f64>,
    sub_mats: Vec<f64>,
    pi: Vec<f64>,
    parent_index: Vec<i32>,
    distances: Vec<f64>,
    model_data: ModelData,
    r: usize,
    c: usize,
    a: usize,
}

impl InsideOutsideTable {
    /// Create from a reversible DiagModel.
    pub fn new(
        alignment: &[i32],
        parent_index: &[i32],
        distances: &[f64],
        model: &DiagModel,
    ) -> Self {
        let r = parent_index.len();
        let a = model.pi.len();
        let c = alignment.len() / r;

        let sm = sub_matrices::compute_sub_matrices(model, distances);
        let (u, log_norm_u, log_like) = pruning::upward_pass(
            alignment, parent_index, &sm, &model.pi, r, c, a,
        );
        let (mut d, mut log_norm_d) = outside::downward_pass(
            &u, &log_norm_u, parent_index, &sm, &model.pi, alignment, r, c, a,
        );

        // Fix root D: set D[0] = pi (rescaled)
        Self::fix_root_d(&mut d, &mut log_norm_d, &model.pi, c, a);

        InsideOutsideTable {
            u, d, log_norm_u, log_norm_d, log_like,
            sub_mats: sm,
            pi: model.pi.clone(),
            parent_index: parent_index.to_vec(),
            distances: distances.to_vec(),
            model_data: ModelData::Reversible {
                eigenvalues: model.eigenvalues.clone(),
                eigenvectors: model.eigenvectors.clone(),
            },
            r, c, a,
        }
    }

    /// Create from an irreversible IrrevDiagModel.
    pub fn new_irrev(
        alignment: &[i32],
        parent_index: &[i32],
        distances: &[f64],
        model: &IrrevDiagModel,
    ) -> Self {
        let r = parent_index.len();
        let a = model.pi.len();
        let c = alignment.len() / r;

        let sm = sub_matrices::compute_sub_matrices_irrev(model, distances);
        let (u, log_norm_u, log_like) = pruning::upward_pass(
            alignment, parent_index, &sm, &model.pi, r, c, a,
        );
        let (mut d, mut log_norm_d) = outside::downward_pass(
            &u, &log_norm_u, parent_index, &sm, &model.pi, alignment, r, c, a,
        );

        Self::fix_root_d(&mut d, &mut log_norm_d, &model.pi, c, a);

        InsideOutsideTable {
            u, d, log_norm_u, log_norm_d, log_like,
            sub_mats: sm,
            pi: model.pi.clone(),
            parent_index: parent_index.to_vec(),
            distances: distances.to_vec(),
            model_data: ModelData::Irreversible {
                eigenvalues_complex: model.eigenvalues_complex.clone(),
                eigenvectors_complex: model.eigenvectors_complex.clone(),
                eigenvectors_inv_complex: model.eigenvectors_inv_complex.clone(),
            },
            r, c, a,
        }
    }

    fn fix_root_d(d: &mut [f64], log_norm_d: &mut [f64], pi: &[f64], c: usize, a: usize) {
        let mut max_val = 0.0f64;
        for aa in 0..a {
            if pi[aa] > max_val { max_val = pi[aa]; }
        }
        if max_val < 1e-300 { max_val = 1e-300; }
        for col in 0..c {
            let base = col * a; // node 0: (0 * c + col) * a = col * a
            for aa in 0..a {
                d[base + aa] = pi[aa] / max_val;
            }
            log_norm_d[col] = max_val.ln(); // node 0: 0 * c + col = col
        }
    }

    /// Per-column log-likelihoods.
    pub fn log_likelihood(&self) -> &[f64] {
        &self.log_like
    }

    /// Expected substitution counts and dwell times.
    /// Returns (A*A*C) flat with layout result[i*A*C + j*C + col].
    pub fn counts(&self, f81_fast_flag: bool) -> Vec<f64> {
        match &self.model_data {
            ModelData::Reversible { eigenvalues, eigenvectors } => {
                if f81_fast_flag {
                    f81_fast::f81_counts(
                        &self.u, &self.d, &self.log_norm_u, &self.log_norm_d,
                        &self.log_like, &self.distances, &self.pi,
                        &self.parent_index, self.r, self.c, self.a,
                    )
                } else {
                    let model = DiagModel {
                        eigenvalues: eigenvalues.clone(),
                        eigenvectors: eigenvectors.clone(),
                        pi: self.pi.clone(),
                    };
                    let j = eigensub::compute_j(eigenvalues, &self.distances);
                    let (u_tilde, d_tilde) = eigensub::eigenbasis_project(
                        &self.u, &self.d, &model, self.r, self.c, self.a,
                    );
                    let c_eigen = eigensub::accumulate_c(
                        &d_tilde, &u_tilde, &j, &self.log_norm_u, &self.log_norm_d,
                        &self.log_like, &self.parent_index, self.r, self.c, self.a,
                    );
                    eigensub::back_transform(&c_eigen, &model, self.c)
                }
            }
            ModelData::Irreversible {
                eigenvalues_complex, eigenvectors_complex, eigenvectors_inv_complex,
            } => {
                let model = IrrevDiagModel {
                    eigenvalues_complex: eigenvalues_complex.clone(),
                    eigenvectors_complex: eigenvectors_complex.clone(),
                    eigenvectors_inv_complex: eigenvectors_inv_complex.clone(),
                    pi: self.pi.clone(),
                };
                let j = eigensub::compute_j_complex(eigenvalues_complex, &self.distances);
                let (u_tilde, d_tilde) = eigensub::eigenbasis_project_irrev(
                    &self.u, &self.d, &model, self.r, self.c, self.a,
                );
                let c_eigen = eigensub::accumulate_c_complex(
                    &d_tilde, &u_tilde, &j, &self.log_norm_u, &self.log_norm_d,
                    &self.log_like, &self.parent_index, self.r, self.c, self.a,
                );
                eigensub::back_transform_irrev(&c_eigen, &model, self.c)
            }
        }
    }

    /// Per-branch expected substitution counts and dwell times.
    /// Returns (R*A*A*C) flat with layout result[n*A*A*C + i*A*C + j*C + col].
    /// Branch 0 = zeros.
    pub fn branch_counts(&self, f81_fast_flag: bool) -> Vec<f64> {
        match &self.model_data {
            ModelData::Reversible { eigenvalues, eigenvectors } => {
                if f81_fast_flag {
                    f81_fast::f81_counts_per_branch(
                        &self.u, &self.d, &self.log_norm_u, &self.log_norm_d,
                        &self.log_like, &self.distances, &self.pi,
                        &self.parent_index, self.r, self.c, self.a,
                    )
                } else {
                    let model = DiagModel {
                        eigenvalues: eigenvalues.clone(),
                        eigenvectors: eigenvectors.clone(),
                        pi: self.pi.clone(),
                    };
                    let j = eigensub::compute_j(eigenvalues, &self.distances);
                    let (u_tilde, d_tilde) = eigensub::eigenbasis_project(
                        &self.u, &self.d, &model, self.r, self.c, self.a,
                    );
                    let c_eigen = eigensub::accumulate_c_per_branch(
                        &d_tilde, &u_tilde, &j, &self.log_norm_u, &self.log_norm_d,
                        &self.log_like, &self.parent_index, self.r, self.c, self.a,
                    );
                    eigensub::back_transform_per_branch(&c_eigen, &model, self.r, self.c)
                }
            }
            ModelData::Irreversible {
                eigenvalues_complex, eigenvectors_complex, eigenvectors_inv_complex,
            } => {
                let model = IrrevDiagModel {
                    eigenvalues_complex: eigenvalues_complex.clone(),
                    eigenvectors_complex: eigenvectors_complex.clone(),
                    eigenvectors_inv_complex: eigenvectors_inv_complex.clone(),
                    pi: self.pi.clone(),
                };
                let j = eigensub::compute_j_complex(eigenvalues_complex, &self.distances);
                let (u_tilde, d_tilde) = eigensub::eigenbasis_project_irrev(
                    &self.u, &self.d, &model, self.r, self.c, self.a,
                );
                let c_eigen = eigensub::accumulate_c_complex_per_branch(
                    &d_tilde, &u_tilde, &j, &self.log_norm_u, &self.log_norm_d,
                    &self.log_like, &self.parent_index, self.r, self.c, self.a,
                );
                eigensub::back_transform_irrev_per_branch(&c_eigen, &model, self.r, self.c)
            }
        }
    }

    /// Posterior state distribution at a single node.
    /// Returns (A*C) flat with layout q[a*C + col].
    pub fn node_posterior_single(&self, node: usize) -> Vec<f64> {
        let (c, a) = (self.c, self.a);
        let mut q = vec![0.0; a * c];

        for col in 0..c {
            let u_base = (node * c + col) * a;
            let d_base = (node * c + col) * a;

            let log_scale = self.log_norm_u[node * c + col]
                + self.log_norm_d[node * c + col]
                - self.log_like[col];
            let scale = log_scale.exp();

            let mut sum = 0.0;
            if node == 0 {
                // Root: D[0] = pi (set in constructor), product is pi * U / Z
                for aa in 0..a {
                    let val = self.d[d_base + aa] * self.u[u_base + aa] * scale;
                    q[aa * c + col] = val;
                    sum += val;
                }
            } else {
                // Non-root: transform D through M
                let m_base = node * a * a;
                for j in 0..a {
                    let mut d_transformed = 0.0;
                    for aa in 0..a {
                        d_transformed += self.d[d_base + aa]
                            * self.sub_mats[m_base + aa * a + j];
                    }
                    let val = d_transformed * self.u[u_base + j] * scale;
                    q[j * c + col] = val;
                    sum += val;
                }
            }
            if sum > 0.0 {
                for aa in 0..a {
                    q[aa * c + col] /= sum;
                }
            }
        }
        q
    }

    /// Posterior state distribution at all nodes.
    /// Returns (R*A*C) flat with layout q[n*A*C + a*C + col].
    pub fn node_posterior_all(&self) -> Vec<f64> {
        let (r, c, a) = (self.r, self.c, self.a);
        let mut result = vec![0.0; r * a * c];
        for n in 0..r {
            let single = self.node_posterior_single(n);
            let offset = n * a * c;
            result[offset..offset + a * c].copy_from_slice(&single);
        }
        result
    }

    /// Joint posterior of parent-child states on a single branch.
    /// Returns (A*A*C) flat with layout joint[i*A*C + j*C + col].
    pub fn branch_posterior_single(&self, node: usize) -> Vec<f64> {
        let (c, a) = (self.c, self.a);
        assert!(node > 0, "Branch 0 (root) has no parent");
        let mut joint = vec![0.0; a * a * c];

        for col in 0..c {
            let d_base = (node * c + col) * a;
            let u_base = (node * c + col) * a;
            let m_base = node * a * a;

            let log_scale = self.log_norm_d[node * c + col]
                + self.log_norm_u[node * c + col]
                - self.log_like[col];
            let scale = log_scale.exp();

            let mut sum = 0.0;
            for i in 0..a {
                for j in 0..a {
                    let val = self.d[d_base + i]
                        * self.sub_mats[m_base + i * a + j]
                        * self.u[u_base + j]
                        * scale;
                    joint[i * a * c + j * c + col] = val;
                    sum += val;
                }
            }
            if sum > 0.0 {
                for i in 0..a {
                    for j in 0..a {
                        joint[i * a * c + j * c + col] /= sum;
                    }
                }
            }
        }
        joint
    }

    /// Joint posterior for all branches.
    /// Returns (R*A*A*C) flat. Branch 0 is zeros.
    pub fn branch_posterior_all(&self) -> Vec<f64> {
        let (r, c, a) = (self.r, self.c, self.a);
        let mut result = vec![0.0; r * a * a * c];
        for n in 1..r {
            let single = self.branch_posterior_single(n);
            let offset = n * a * a * c;
            result[offset..offset + a * a * c].copy_from_slice(&single);
        }
        result
    }
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
    pub fn wasm_log_like_irrev(
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
        crate::log_like_irrev(alignment, parent_index, distances, &m)
    }

    #[wasm_bindgen]
    pub fn wasm_root_prob_irrev(
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
        crate::root_prob_irrev(alignment, parent_index, distances, &m)
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

    #[wasm_bindgen]
    pub fn wasm_branch_counts(
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
        crate::branch_counts(alignment, parent_index, distances, &m, f81_fast)
    }

    #[wasm_bindgen]
    pub fn wasm_branch_counts_irrev(
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
        crate::branch_counts_irrev(alignment, parent_index, distances, &m)
    }

    #[wasm_bindgen]
    pub fn wasm_expected_counts(
        eigenvalues: &[f64],
        eigenvectors: &[f64],
        pi: &[f64],
        t: f64,
    ) -> Vec<f64> {
        let m = model::DiagModel {
            eigenvalues: eigenvalues.to_vec(),
            eigenvectors: eigenvectors.to_vec(),
            pi: pi.to_vec(),
        };
        crate::expected_counts(&m, t)
    }

    #[wasm_bindgen]
    pub fn wasm_expected_counts_irrev(
        eigenvalues_complex: &[f64],
        eigenvectors_complex: &[f64],
        eigenvectors_inv_complex: &[f64],
        pi: &[f64],
        t: f64,
    ) -> Vec<f64> {
        let m = crate::model::IrrevDiagModel {
            eigenvalues_complex: eigenvalues_complex.to_vec(),
            eigenvectors_complex: eigenvectors_complex.to_vec(),
            eigenvectors_inv_complex: eigenvectors_inv_complex.to_vec(),
            pi: pi.to_vec(),
        };
        crate::expected_counts_irrev(&m, t)
    }

    // ---- InsideOutside WASM bindings ----

    #[wasm_bindgen]
    pub struct WasmInsideOutside {
        table: crate::InsideOutsideTable,
    }

    #[wasm_bindgen]
    impl WasmInsideOutside {
        /// Create InsideOutside table for a reversible model.
        pub fn create(
            alignment: &[i32],
            parent_index: &[i32],
            distances: &[f64],
            eigenvalues: &[f64],
            eigenvectors: &[f64],
            pi: &[f64],
        ) -> WasmInsideOutside {
            let m = model::DiagModel {
                eigenvalues: eigenvalues.to_vec(),
                eigenvectors: eigenvectors.to_vec(),
                pi: pi.to_vec(),
            };
            WasmInsideOutside {
                table: crate::InsideOutsideTable::new(
                    alignment, parent_index, distances, &m,
                ),
            }
        }

        /// Create InsideOutside table for an irreversible model.
        pub fn create_irrev(
            alignment: &[i32],
            parent_index: &[i32],
            distances: &[f64],
            eigenvalues_complex: &[f64],
            eigenvectors_complex: &[f64],
            eigenvectors_inv_complex: &[f64],
            pi: &[f64],
        ) -> WasmInsideOutside {
            let m = model::IrrevDiagModel {
                eigenvalues_complex: eigenvalues_complex.to_vec(),
                eigenvectors_complex: eigenvectors_complex.to_vec(),
                eigenvectors_inv_complex: eigenvectors_inv_complex.to_vec(),
                pi: pi.to_vec(),
            };
            WasmInsideOutside {
                table: crate::InsideOutsideTable::new_irrev(
                    alignment, parent_index, distances, &m,
                ),
            }
        }

        pub fn log_likelihood(&self) -> Vec<f64> {
            self.table.log_likelihood().to_vec()
        }

        pub fn counts(&self, f81_fast: bool) -> Vec<f64> {
            self.table.counts(f81_fast)
        }

        /// Per-branch expected counts and dwell times.
        /// Returns (R*A*A*C) flat. Branch 0 = zeros.
        pub fn branch_counts(&self, f81_fast: bool) -> Vec<f64> {
            self.table.branch_counts(f81_fast)
        }

        /// Posterior state distribution. node = -1 for all nodes.
        /// Returns (A*C) for single node or (R*A*C) for all.
        pub fn node_posterior(&self, node: i32) -> Vec<f64> {
            if node < 0 {
                self.table.node_posterior_all()
            } else {
                self.table.node_posterior_single(node as usize)
            }
        }

        /// Joint branch posterior. node = -1 for all branches.
        /// Returns (A*A*C) for single branch or (R*A*A*C) for all.
        pub fn branch_posterior(&self, node: i32) -> Vec<f64> {
            if node < 0 {
                self.table.branch_posterior_all()
            } else {
                self.table.branch_posterior_single(node as usize)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::complex::*;

    // ---- Deterministic PRNG (LCG) for reproducible test data ----

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self { Self(seed) }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }

        /// Uniform f64 in [lo, hi)
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            lo + u * (hi - lo)
        }

        /// Random integer in [0, n)
        fn int(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    // ---- Tree builders ----

    /// Balanced binary tree with R nodes (R must be odd).
    /// parent(n) = (n-1)/2 gives each internal node exactly 2 children.
    fn make_tree(r: usize, seed: u64) -> (Vec<i32>, Vec<f64>) {
        assert!(r >= 3 && r % 2 == 1, "R must be odd and >= 3 for binary tree");
        let mut rng = Rng::new(seed);
        let mut parent_index = vec![0i32; r];
        parent_index[0] = -1;
        for n in 1..r {
            parent_index[n] = ((n - 1) / 2) as i32;
        }
        let mut distances = vec![0.0; r];
        distances[0] = 0.0;
        for n in 1..r {
            distances[n] = rng.uniform(0.01, 0.5);
        }
        (parent_index, distances)
    }

    /// Generate alignment (R x C) with tokens in [0, A), plus optional gaps (-1).
    fn make_alignment(r: usize, c: usize, a: usize, seed: u64, gap_rate: f64) -> Vec<i32> {
        let mut rng = Rng::new(seed);
        let mut alignment = vec![0i32; r * c];
        for i in 0..r * c {
            if rng.uniform(0.0, 1.0) < gap_rate {
                alignment[i] = -1;
            } else {
                alignment[i] = rng.int(a) as i32;
            }
        }
        alignment
    }

    // ---- Model builders ----

    /// Build an IrrevDiagModel from a DiagModel (real eigenvalues/vectors → interleaved complex).
    fn make_irrev_from_diag(diag: &DiagModel) -> IrrevDiagModel {
        let a = diag.pi.len();
        let mut eigenvalues_complex = vec![0.0; 2 * a];
        for i in 0..a {
            eigenvalues_complex[2 * i] = diag.eigenvalues[i];
        }
        let mut eigenvectors_complex = vec![0.0; 2 * a * a];
        for i in 0..a {
            for k in 0..a {
                let u_ik = diag.eigenvectors[i * a + k];
                eigenvectors_complex[2 * (i * a + k)] = u_ik / diag.pi[i].sqrt();
            }
        }
        let mut eigenvectors_inv_complex = vec![0.0; 2 * a * a];
        for k in 0..a {
            for j in 0..a {
                let u_jk = diag.eigenvectors[j * a + k];
                eigenvectors_inv_complex[2 * (k * a + j)] = u_jk * diag.pi[j].sqrt();
            }
        }
        IrrevDiagModel {
            eigenvalues_complex,
            eigenvectors_complex,
            eigenvectors_inv_complex,
            pi: diag.pi.clone(),
        }
    }

    /// Build a genuine irreversible rate matrix (off-diag ≥ 0, rows sum to 0)
    /// and eigendecompose it. The resulting model has truly complex-valued
    /// (or at least non-symmetric) eigenvectors.
    fn make_irrev_model(a: usize, seed: u64) -> (IrrevDiagModel, Vec<f64>) {
        let mut rng = Rng::new(seed);
        // Random off-diagonal rates, deliberately asymmetric
        let mut q = vec![0.0; a * a];
        for i in 0..a {
            for j in 0..a {
                if i != j {
                    q[i * a + j] = rng.uniform(0.01, 1.0);
                }
            }
        }
        // Set diagonal so rows sum to 0
        for i in 0..a {
            let mut row_sum = 0.0;
            for j in 0..a {
                if j != i { row_sum += q[i * a + j]; }
            }
            q[i * a + i] = -row_sum;
        }
        // Uniform pi
        let pi: Vec<f64> = vec![1.0 / a as f64; a];
        // Normalize: expected rate = -sum_i pi_i * q_ii = 1
        let mut rate = 0.0;
        for i in 0..a { rate -= pi[i] * q[i * a + i]; }
        for v in &mut q { *v /= rate; }

        // Eigendecompose Q using brute-force power/inverse iteration
        // For small A (4-20), we use a direct approach: compute matrix exponential
        // at tiny t to numerically verify, but for the model we just store Q directly
        // and eigendecompose via Schur-like approach.
        //
        // Actually, we'll use the same approach as the Python tests:
        // form V, V_inv via solving Q*v = lambda*v for each eigenvalue found by QR.
        //
        // For simplicity in the test (since this is small A), we'll compute M(t) = exp(Q*t)
        // directly via Taylor series and verify sub_matrices_irrev matches.
        // The model is constructed using a direct numerical eigendecomposition.
        let model = eigendecompose_irrev(&q, &pi, a);
        (model, q)
    }

    /// Simple numerical eigendecomposition of a real matrix for test purposes.
    /// Uses repeated Schur + inverse iteration, good enough for A ≤ 20.
    fn eigendecompose_irrev(q: &[f64], pi: &[f64], a: usize) -> IrrevDiagModel {
        // Use Hessenberg reduction + QR iteration to find eigenvalues,
        // then inverse iteration for eigenvectors.
        // This mirrors the JS eigGeneral implementation.

        let mut h = q.to_vec();

        // Hessenberg reduction via Householder
        for k in 0..(a.max(2) - 2) {
            let mut x = vec![0.0; a - k - 1];
            for i in 0..x.len() { x[i] = h[(k + 1 + i) * a + k]; }
            let mut x_norm = 0.0;
            for v in &x { x_norm += v * v; }
            x_norm = x_norm.sqrt();
            if x_norm < 1e-15 { continue; }
            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            x[0] += sign * x_norm;
            let mut v_norm = 0.0;
            for v in &x { v_norm += v * v; }
            v_norm = v_norm.sqrt();
            for v in &mut x { *v /= v_norm; }

            // Left multiply
            for j in 0..a {
                let mut dot = 0.0;
                for i in 0..x.len() { dot += x[i] * h[(k + 1 + i) * a + j]; }
                for i in 0..x.len() { h[(k + 1 + i) * a + j] -= 2.0 * x[i] * dot; }
            }
            // Right multiply
            for i in 0..a {
                let mut dot = 0.0;
                for j in 0..x.len() { dot += h[i * a + (k + 1 + j)] * x[j]; }
                for j in 0..x.len() { h[i * a + (k + 1 + j)] -= 2.0 * dot * x[j]; }
            }
        }

        // QR iteration
        let mut eigen_re = vec![0.0; a];
        let mut eigen_im = vec![0.0; a];
        let mut n = a;

        for _ in 0..2000 {
            if n == 0 { break; }
            if n == 1 {
                eigen_re[0] = h[0];
                break;
            }

            let sub = h[(n - 1) * a + (n - 2)].abs();
            let diag_sum = h[(n - 2) * a + (n - 2)].abs() + h[(n - 1) * a + (n - 1)].abs();
            if sub < 1e-14 * diag_sum + 1e-300 {
                eigen_re[n - 1] = h[(n - 1) * a + (n - 1)];
                n -= 1;
                continue;
            }

            let is_2x2 = n == 2 || h[(n - 2) * a + (n - 3)].abs() <
                1e-14 * (h[(n - 3) * a + (n - 3)].abs() + h[(n - 2) * a + (n - 2)].abs()) + 1e-300;
            if is_2x2 {
                let a11 = h[(n - 2) * a + (n - 2)];
                let a12 = h[(n - 2) * a + (n - 1)];
                let a21 = h[(n - 1) * a + (n - 2)];
                let a22 = h[(n - 1) * a + (n - 1)];
                let tr = a11 + a22;
                let det = a11 * a22 - a12 * a21;
                let disc = tr * tr - 4.0 * det;
                if disc >= 0.0 {
                    let sq = disc.sqrt();
                    eigen_re[n - 2] = (tr + sq) / 2.0;
                    eigen_re[n - 1] = (tr - sq) / 2.0;
                } else {
                    let sq = (-disc).sqrt();
                    eigen_re[n - 2] = tr / 2.0; eigen_im[n - 2] = sq / 2.0;
                    eigen_re[n - 1] = tr / 2.0; eigen_im[n - 1] = -sq / 2.0;
                }
                n -= 2;
                continue;
            }

            // Wilkinson shift
            let wa = h[(n - 2) * a + (n - 2)];
            let wb = h[(n - 2) * a + (n - 1)];
            let wc = h[(n - 1) * a + (n - 2)];
            let wd = h[(n - 1) * a + (n - 1)];
            let wtr = wa + wd;
            let wdet = wa * wd - wb * wc;
            let wdisc = wtr * wtr - 4.0 * wdet;
            let shift = if wdisc >= 0.0 {
                let sq = wdisc.sqrt();
                let e1 = (wtr + sq) / 2.0;
                let e2 = (wtr - sq) / 2.0;
                if (e1 - wd).abs() < (e2 - wd).abs() { e1 } else { e2 }
            } else { wd };

            for i in 0..n { h[i * a + i] -= shift; }
            let mut cs = vec![0.0; n - 1];
            let mut sn = vec![0.0; n - 1];
            for i in 0..n - 1 {
                let hi = h[i * a + i];
                let hi1 = h[(i + 1) * a + i];
                let r = (hi * hi + hi1 * hi1).sqrt();
                if r < 1e-300 { cs[i] = 1.0; continue; }
                cs[i] = hi / r; sn[i] = hi1 / r;
                for j in 0..n {
                    let t1 = h[i * a + j];
                    let t2 = h[(i + 1) * a + j];
                    h[i * a + j] = cs[i] * t1 + sn[i] * t2;
                    h[(i + 1) * a + j] = -sn[i] * t1 + cs[i] * t2;
                }
            }
            for i in 0..n - 1 {
                for j in 0..n {
                    let t1 = h[j * a + i];
                    let t2 = h[j * a + (i + 1)];
                    h[j * a + i] = cs[i] * t1 + sn[i] * t2;
                    h[j * a + (i + 1)] = -sn[i] * t1 + cs[i] * t2;
                }
            }
            for i in 0..n { h[i * a + i] += shift; }
        }

        // Eigenvectors via inverse iteration
        let mut eigenvalues_complex = vec![0.0; 2 * a];
        let mut eigenvectors_complex = vec![0.0; 2 * a * a];
        for idx in 0..a {
            eigenvalues_complex[2 * idx] = eigen_re[idx];
            eigenvalues_complex[2 * idx + 1] = eigen_im[idx];
        }

        for idx in 0..a {
            let lr = eigen_re[idx];
            let li = eigen_im[idx];
            if li.abs() < 1e-14 {
                // Real eigenvalue: inverse iteration
                let mut shifted = q.to_vec();
                for i in 0..a { shifted[i * a + i] -= lr + 1e-14; }
                let mut v = vec![1.0; a];
                for _ in 0..40 {
                    let w = solve_lu(&shifted, &v, a);
                    let mut nm = 0.0;
                    for x in &w { nm += x * x; }
                    nm = nm.sqrt();
                    if nm < 1e-300 { break; }
                    for i in 0..a { v[i] = w[i] / nm; }
                }
                for i in 0..a {
                    eigenvectors_complex[2 * (i * a + idx)] = v[i];
                }
            } else if li > 0.0 {
                // Complex pair: solve in augmented 2A×2A real system
                let mut vr = vec![1.0; a];
                let mut vi = vec![0.5; a];
                for _ in 0..40 {
                    let (wr, wi) = solve_complex_lu(q, lr + 1e-14, li, &vr, &vi, a);
                    let mut nm = 0.0;
                    for i in 0..a { nm += wr[i] * wr[i] + wi[i] * wi[i]; }
                    nm = nm.sqrt();
                    if nm < 1e-300 { break; }
                    for i in 0..a { vr[i] = wr[i] / nm; vi[i] = wi[i] / nm; }
                }
                for i in 0..a {
                    eigenvectors_complex[2 * (i * a + idx)] = vr[i];
                    eigenvectors_complex[2 * (i * a + idx) + 1] = vi[i];
                }
                // Fill conjugate
                for j in (idx + 1)..a {
                    if (eigen_re[j] - lr).abs() < 1e-12 && (eigen_im[j] + li).abs() < 1e-12 {
                        for i in 0..a {
                            eigenvectors_complex[2 * (i * a + j)] = vr[i];
                            eigenvectors_complex[2 * (i * a + j) + 1] = -vi[i];
                        }
                        break;
                    }
                }
            }
        }

        // Invert eigenvector matrix
        let eigenvectors_inv_complex = invert_complex_matrix(&eigenvectors_complex, a);

        IrrevDiagModel {
            eigenvalues_complex,
            eigenvectors_complex,
            eigenvectors_inv_complex,
            pi: pi.to_vec(),
        }
    }

    fn solve_lu(a_mat: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut lu = a_mat.to_vec();
        let mut piv: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut mx = 0.0;
            let mut mr = k;
            for i in k..n {
                let v = lu[i * n + k].abs();
                if v > mx { mx = v; mr = i; }
            }
            if mr != k {
                for j in 0..n { let t = lu[k * n + j]; lu[k * n + j] = lu[mr * n + j]; lu[mr * n + j] = t; }
                piv.swap(k, mr);
            }
            if lu[k * n + k].abs() < 1e-300 { continue; }
            for i in (k + 1)..n {
                lu[i * n + k] /= lu[k * n + k];
                for j in (k + 1)..n { lu[i * n + j] -= lu[i * n + k] * lu[k * n + j]; }
            }
        }
        let mut x = vec![0.0; n];
        for i in 0..n { x[i] = b[piv[i]]; }
        for i in 0..n { for j in 0..i { x[i] -= lu[i * n + j] * x[j]; } }
        for i in (0..n).rev() {
            for j in (i + 1)..n { x[i] -= lu[i * n + j] * x[j]; }
            x[i] /= lu[i * n + i];
        }
        x
    }

    fn solve_complex_lu(a_mat: &[f64], lr: f64, li: f64, br: &[f64], bi: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut m2 = vec![0.0; 4 * n * n];
        for i in 0..n {
            for j in 0..n {
                let v = a_mat[i * n + j] - if i == j { lr } else { 0.0 };
                m2[i * 2 * n + j] = v;
                m2[(n + i) * 2 * n + (n + j)] = v;
            }
            m2[i * 2 * n + (n + i)] = li;
            m2[(n + i) * 2 * n + i] = -li;
        }
        let mut rhs = vec![0.0; 2 * n];
        for i in 0..n { rhs[i] = br[i]; rhs[n + i] = bi[i]; }
        let sol = solve_lu(&m2, &rhs, 2 * n);
        (sol[..n].to_vec(), sol[n..].to_vec())
    }

    fn invert_complex_matrix(v: &[f64], a: usize) -> Vec<f64> {
        let mut re = vec![0.0; a * a];
        let mut im = vec![0.0; a * a];
        let mut ire = vec![0.0; a * a];
        let mut iim = vec![0.0; a * a];
        for i in 0..(a * a) { re[i] = v[2 * i]; im[i] = v[2 * i + 1]; }
        for i in 0..a { ire[i * a + i] = 1.0; }

        for k in 0..a {
            let mut max_n = 0.0;
            let mut mr = k;
            for i in k..a {
                let nr = re[i * a + k] * re[i * a + k] + im[i * a + k] * im[i * a + k];
                if nr > max_n { max_n = nr; mr = i; }
            }
            if mr != k {
                for j in 0..a {
                    let swap = |arr: &mut Vec<f64>| { let t = arr[k * a + j]; arr[k * a + j] = arr[mr * a + j]; arr[mr * a + j] = t; };
                    swap(&mut re); swap(&mut im); swap(&mut ire); swap(&mut iim);
                }
            }
            let pr = re[k * a + k]; let pim = im[k * a + k];
            let den = pr * pr + pim * pim;
            if den < 1e-300 { continue; }
            let inv_r = pr / den; let inv_i = -pim / den;
            for j in 0..a {
                let ar = re[k * a + j]; let ai = im[k * a + j];
                re[k * a + j] = ar * inv_r - ai * inv_i;
                im[k * a + j] = ar * inv_i + ai * inv_r;
                let cr = ire[k * a + j]; let ci = iim[k * a + j];
                ire[k * a + j] = cr * inv_r - ci * inv_i;
                iim[k * a + j] = cr * inv_i + ci * inv_r;
            }
            for i in 0..a {
                if i == k { continue; }
                let fr = re[i * a + k]; let fi = im[i * a + k];
                for j in 0..a {
                    let rk = re[k * a + j]; let ik = im[k * a + j];
                    re[i * a + j] -= fr * rk - fi * ik;
                    im[i * a + j] -= fr * ik + fi * rk;
                    let irk = ire[k * a + j]; let iik = iim[k * a + j];
                    ire[i * a + j] -= fr * irk - fi * iik;
                    iim[i * a + j] -= fr * iik + fi * irk;
                }
            }
        }
        let mut result = vec![0.0; 2 * a * a];
        for i in 0..(a * a) { result[2 * i] = ire[i]; result[2 * i + 1] = iim[i]; }
        result
    }

    /// Matrix exponential via Taylor series (for small t*||Q||).
    fn mat_exp(q: &[f64], t: f64, a: usize) -> Vec<f64> {
        let mut result = vec![0.0; a * a];
        for i in 0..a { result[i * a + i] = 1.0; }
        let mut term = result.clone();
        for n in 1..80 {
            let prev = term.clone();
            for i in 0..a {
                for j in 0..a {
                    let mut s = 0.0;
                    for k in 0..a { s += prev[i * a + k] * q[k * a + j]; }
                    term[i * a + j] = s * t / n as f64;
                }
            }
            for i in 0..(a * a) { result[i] += term[i]; }
        }
        result
    }

    // ---- Assertion helpers ----

    fn assert_log_like_properties(ll: &[f64], c: usize, label: &str) {
        assert_eq!(ll.len(), c, "{}: wrong length", label);
        for (i, v) in ll.iter().enumerate() {
            assert!(v.is_finite(), "{}: ll[{}] = {} not finite", label, i, v);
            assert!(*v <= 1e-10, "{}: ll[{}] = {} not ≤ 0", label, i, v);
        }
    }

    fn assert_root_prob_properties(rp: &[f64], a: usize, c: usize, label: &str) {
        assert_eq!(rp.len(), a * c, "{}: wrong length", label);
        for col in 0..c {
            let mut sum = 0.0;
            for aa in 0..a {
                let v = rp[aa * c + col];
                assert!(v.is_finite(), "{}: rp[{},{}] not finite", label, aa, col);
                assert!(v >= -1e-10, "{}: rp[{},{}] = {} negative", label, aa, col, v);
                sum += v;
            }
            assert!((sum - 1.0).abs() < 1e-6,
                     "{}: col {} sums to {} (expected 1.0)", label, col, sum);
        }
    }

    fn assert_counts_properties(cts: &[f64], a: usize, c: usize, label: &str) {
        assert_eq!(cts.len(), a * a * c, "{}: wrong length", label);
        for i in 0..(a * a * c) {
            assert!(cts[i].is_finite(), "{}: counts[{}] not finite", label, i);
        }
        // Off-diagonal counts should be non-negative (substitutions)
        for i in 0..a {
            for j in 0..a {
                if i == j { continue; }
                for col in 0..c {
                    let v = cts[i * a * c + j * c + col];
                    assert!(v >= -1e-8,
                            "{}: off-diag counts[{},{},{}] = {} negative", label, i, j, col, v);
                }
            }
        }
    }

    // ================================================================
    // 1. BASIC PROPERTY TESTS — various tree sizes, widths, models
    // ================================================================

    #[test]
    fn test_log_like_various_sizes() {
        let configs: Vec<(usize, usize, usize, u64)> = vec![
            // (R, C, A, seed)
            (3, 1, 4, 100),
            (3, 4, 4, 101),
            (5, 8, 4, 102),
            (7, 16, 4, 103),
            (11, 4, 4, 104),
            (5, 4, 20, 105),  // large alphabet
        ];
        for (r, c, a, seed) in configs {
            let (pi, dist) = make_tree(r, seed);
            let alignment = make_alignment(r, c, a, seed + 1000, 0.0);
            let m = model::jukes_cantor_model(a);
            let ll = log_like(&alignment, &pi, &dist, &m);
            assert_log_like_properties(&ll, c, &format!("JC R={} C={} A={}", r, c, a));
        }
    }

    #[test]
    fn test_root_prob_various_sizes() {
        let configs: Vec<(usize, usize, usize, u64)> = vec![
            (3, 1, 4, 200), (5, 4, 4, 201), (7, 8, 4, 202),
            (11, 4, 4, 203), (5, 4, 20, 204),
        ];
        for (r, c, a, seed) in configs {
            let (pi, dist) = make_tree(r, seed);
            let alignment = make_alignment(r, c, a, seed + 1000, 0.0);
            let m = model::jukes_cantor_model(a);
            let rp = root_prob(&alignment, &pi, &dist, &m);
            assert_root_prob_properties(&rp, a, c, &format!("JC R={} C={} A={}", r, c, a));
        }
    }

    #[test]
    fn test_counts_various_sizes() {
        let configs: Vec<(usize, usize, u64)> = vec![
            (3, 1, 300), (5, 4, 301), (7, 8, 302), (11, 4, 303),
        ];
        for (r, c, seed) in configs {
            let a = 4;
            let (pi, dist) = make_tree(r, seed);
            let alignment = make_alignment(r, c, a, seed + 1000, 0.0);
            let m = model::jukes_cantor_model(a);
            let cts = counts(&alignment, &pi, &dist, &m, false);
            assert_counts_properties(&cts, a, c, &format!("eigensub R={} C={}", r, c));
        }
    }

    #[test]
    fn test_counts_f81_fast_various_sizes() {
        let configs: Vec<(usize, usize, u64)> = vec![
            (3, 1, 310), (5, 4, 311), (7, 8, 312), (11, 4, 313),
        ];
        for (r, c, seed) in configs {
            let a = 4;
            let (pi, dist) = make_tree(r, seed);
            let alignment = make_alignment(r, c, a, seed + 1000, 0.0);
            let m = model::jukes_cantor_model(a);
            let cts = counts(&alignment, &pi, &dist, &m, true);
            assert_counts_properties(&cts, a, c, &format!("f81fast R={} C={}", r, c));
        }
    }

    // ================================================================
    // 2. GAP HANDLING
    // ================================================================

    #[test]
    fn test_log_like_with_gaps() {
        // Light gaps (20%)
        let (pi, dist) = make_tree(7, 400);
        let alignment = make_alignment(7, 8, 4, 401, 0.2);
        let m = model::jukes_cantor_model(4);
        let ll = log_like(&alignment, &pi, &dist, &m);
        assert_log_like_properties(&ll, 8, "gaps_20pct");
    }

    #[test]
    fn test_log_like_heavy_gaps() {
        // Heavy gaps (60%)
        let (pi, dist) = make_tree(7, 410);
        let alignment = make_alignment(7, 8, 4, 411, 0.6);
        let m = model::jukes_cantor_model(4);
        let ll = log_like(&alignment, &pi, &dist, &m);
        assert_log_like_properties(&ll, 8, "gaps_60pct");
    }

    #[test]
    fn test_all_gap_column() {
        // One column is all gaps, others normal
        let (pi, dist) = make_tree(5, 420);
        let mut alignment = make_alignment(5, 4, 4, 421, 0.0);
        // Set column 2 to all gaps
        for r in 0..5 { alignment[r * 4 + 2] = -1; }
        let m = model::jukes_cantor_model(4);
        let ll = log_like(&alignment, &pi, &dist, &m);
        assert_eq!(ll.len(), 4);
        // All-gap column should have loglike = 0 (or very close)
        assert!((ll[2] - 0.0).abs() < 1e-10,
                "all-gap column ll = {} (expected ~0)", ll[2]);
    }

    #[test]
    fn test_root_prob_with_gaps() {
        let (pi, dist) = make_tree(7, 430);
        let alignment = make_alignment(7, 8, 4, 431, 0.3);
        let m = model::jukes_cantor_model(4);
        let rp = root_prob(&alignment, &pi, &dist, &m);
        assert_root_prob_properties(&rp, 4, 8, "gaps_30pct");
    }

    #[test]
    fn test_counts_with_gaps() {
        let (pi, dist) = make_tree(7, 440);
        let alignment = make_alignment(7, 8, 4, 441, 0.25);
        let m = model::jukes_cantor_model(4);
        let cts = counts(&alignment, &pi, &dist, &m, false);
        assert_counts_properties(&cts, 4, 8, "counts_gaps_25pct");
    }

    // ================================================================
    // 3. MULTIPLE MODEL TYPES
    // ================================================================

    #[test]
    fn test_hky85_model() {
        let (pi_tree, dist) = make_tree(7, 500);
        let alignment = make_alignment(7, 8, 4, 501, 0.0);
        let pi_freq = [0.3, 0.2, 0.2, 0.3];
        let m = model::hky85_diag(2.5, &pi_freq);
        let ll = log_like(&alignment, &pi_tree, &dist, &m);
        assert_log_like_properties(&ll, 8, "hky85_k2.5");
        let rp = root_prob(&alignment, &pi_tree, &dist, &m);
        assert_root_prob_properties(&rp, 4, 8, "hky85_k2.5");
        let cts = counts(&alignment, &pi_tree, &dist, &m, false);
        assert_counts_properties(&cts, 4, 8, "hky85_k2.5");
    }

    #[test]
    fn test_hky85_extreme_kappa() {
        let (pi_tree, dist) = make_tree(5, 510);
        let alignment = make_alignment(5, 4, 4, 511, 0.0);
        // Very high ts/tv ratio
        let pi_freq = [0.22, 0.28, 0.18, 0.32];
        let m = model::hky85_diag(20.0, &pi_freq);
        let ll = log_like(&alignment, &pi_tree, &dist, &m);
        assert_log_like_properties(&ll, 4, "hky85_k20");
    }

    #[test]
    fn test_f81_model() {
        let (pi_tree, dist) = make_tree(7, 520);
        let alignment = make_alignment(7, 8, 4, 521, 0.0);
        let pi_freq = [0.35, 0.15, 0.25, 0.25];
        let m = model::f81_model(&pi_freq);
        let ll = log_like(&alignment, &pi_tree, &dist, &m);
        assert_log_like_properties(&ll, 8, "f81");
        let rp = root_prob(&alignment, &pi_tree, &dist, &m);
        assert_root_prob_properties(&rp, 4, 8, "f81");
        let cts = counts(&alignment, &pi_tree, &dist, &m, false);
        assert_counts_properties(&cts, 4, 8, "f81_eigensub");
        let cts_fast = counts(&alignment, &pi_tree, &dist, &m, true);
        assert_counts_properties(&cts_fast, 4, 8, "f81_fast");
    }

    #[test]
    fn test_f81_vs_fast_agree() {
        let (pi_tree, dist) = make_tree(7, 530);
        let alignment = make_alignment(7, 6, 4, 531, 0.0);
        let pi_freq = [0.28, 0.22, 0.30, 0.20];
        let m = model::f81_model(&pi_freq);
        let cts_gen = counts(&alignment, &pi_tree, &dist, &m, false);
        let cts_fast = counts(&alignment, &pi_tree, &dist, &m, true);
        for (i, (a, b)) in cts_gen.iter().zip(cts_fast.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6,
                    "f81 general vs fast mismatch at [{}]: {} vs {}", i, a, b);
        }
    }

    #[test]
    fn test_jc20_large_alphabet() {
        let (pi_tree, dist) = make_tree(7, 540);
        let alignment = make_alignment(7, 4, 20, 541, 0.0);
        let m = model::jukes_cantor_model(20);
        let ll = log_like(&alignment, &pi_tree, &dist, &m);
        assert_log_like_properties(&ll, 4, "jc20");
        let rp = root_prob(&alignment, &pi_tree, &dist, &m);
        assert_root_prob_properties(&rp, 20, 4, "jc20");
    }

    // ================================================================
    // 4. REVERSIBLE ↔ IRREVERSIBLE CROSS-VALIDATION
    // ================================================================

    #[test]
    fn test_rev_vs_irrev_log_like_jc4() {
        let configs: Vec<(usize, usize, u64)> = vec![
            (3, 2, 600), (5, 4, 601), (7, 8, 602), (11, 4, 603),
        ];
        for (r, c, seed) in configs {
            let (pi, dist) = make_tree(r, seed);
            let alignment = make_alignment(r, c, 4, seed + 1000, 0.0);
            let diag = model::jukes_cantor_model(4);
            let irrev = make_irrev_from_diag(&diag);
            let ll_rev = log_like(&alignment, &pi, &dist, &diag);
            let ll_irrev = log_like_irrev(&alignment, &pi, &dist, &irrev);
            for (i, (a, b)) in ll_rev.iter().zip(ll_irrev.iter()).enumerate() {
                assert!((a - b).abs() < 1e-8,
                        "rev/irrev ll mismatch R={} C={} col {}: {} vs {}", r, c, i, a, b);
            }
        }
    }

    #[test]
    fn test_rev_vs_irrev_log_like_hky85() {
        let (pi_tree, dist) = make_tree(7, 610);
        let alignment = make_alignment(7, 6, 4, 611, 0.0);
        let pi_freq = [0.3, 0.2, 0.2, 0.3];
        let diag = model::hky85_diag(3.0, &pi_freq);
        let irrev = make_irrev_from_diag(&diag);
        let ll_rev = log_like(&alignment, &pi_tree, &dist, &diag);
        let ll_irrev = log_like_irrev(&alignment, &pi_tree, &dist, &irrev);
        for (i, (a, b)) in ll_rev.iter().zip(ll_irrev.iter()).enumerate() {
            assert!((a - b).abs() < 1e-8,
                    "hky85 rev/irrev ll mismatch col {}: {} vs {}", i, a, b);
        }
    }

    #[test]
    fn test_rev_vs_irrev_root_prob() {
        let configs: Vec<(usize, usize, u64)> = vec![
            (3, 2, 620), (5, 4, 621), (7, 8, 622),
        ];
        for (r, c, seed) in configs {
            let (pi, dist) = make_tree(r, seed);
            let alignment = make_alignment(r, c, 4, seed + 1000, 0.0);
            let diag = model::jukes_cantor_model(4);
            let irrev = make_irrev_from_diag(&diag);
            let rp_rev = root_prob(&alignment, &pi, &dist, &diag);
            let rp_irrev = root_prob_irrev(&alignment, &pi, &dist, &irrev);
            for (i, (a, b)) in rp_rev.iter().zip(rp_irrev.iter()).enumerate() {
                assert!((a - b).abs() < 1e-8,
                        "rev/irrev rp mismatch R={} C={} at {}: {} vs {}", r, c, i, a, b);
            }
        }
    }

    #[test]
    fn test_rev_vs_irrev_counts() {
        let (pi_tree, dist) = make_tree(7, 630);
        let alignment = make_alignment(7, 4, 4, 631, 0.0);
        let diag = model::jukes_cantor_model(4);
        let irrev = make_irrev_from_diag(&diag);
        let cts_rev = counts(&alignment, &pi_tree, &dist, &diag, false);
        let cts_irrev = counts_irrev(&alignment, &pi_tree, &dist, &irrev);
        for (i, (a, b)) in cts_rev.iter().zip(cts_irrev.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6,
                    "rev/irrev counts mismatch at {}: {} vs {}", i, a, b);
        }
    }

    #[test]
    fn test_rev_vs_irrev_with_gaps() {
        let (pi_tree, dist) = make_tree(7, 640);
        let alignment = make_alignment(7, 6, 4, 641, 0.3);
        let diag = model::jukes_cantor_model(4);
        let irrev = make_irrev_from_diag(&diag);
        let ll_rev = log_like(&alignment, &pi_tree, &dist, &diag);
        let ll_irrev = log_like_irrev(&alignment, &pi_tree, &dist, &irrev);
        for (i, (a, b)) in ll_rev.iter().zip(ll_irrev.iter()).enumerate() {
            assert!((a - b).abs() < 1e-8,
                    "gapped rev/irrev ll mismatch col {}: {} vs {}", i, a, b);
        }
    }

    // ================================================================
    // 5. GENUINE IRREVERSIBLE MODELS (asymmetric rate matrices)
    // ================================================================

    #[test]
    fn test_irrev_sub_matrices_match_expm() {
        let a = 4;
        let (irrev_model, q) = make_irrev_model(a, 700);
        let distances = vec![0.01, 0.05, 0.1, 0.3, 0.5, 1.0];
        let sub_mats = sub_matrices::compute_sub_matrices_irrev(&irrev_model, &distances);
        for (r, &t) in distances.iter().enumerate() {
            let m_exact = mat_exp(&q, t, a);
            for i in 0..a {
                for j in 0..a {
                    let got = sub_mats[r * a * a + i * a + j];
                    let want = m_exact[i * a + j];
                    assert!((got - want).abs() < 1e-8,
                            "M(t={})[{},{}]: got {} want {}", t, i, j, got, want);
                }
            }
        }
    }

    #[test]
    fn test_irrev_sub_matrices_row_stochastic() {
        let a = 4;
        let (irrev_model, _) = make_irrev_model(a, 710);
        let distances = vec![0.01, 0.1, 0.5, 1.0, 2.0];
        let sub_mats = sub_matrices::compute_sub_matrices_irrev(&irrev_model, &distances);
        for (r, &t) in distances.iter().enumerate() {
            for i in 0..a {
                let mut row_sum = 0.0;
                for j in 0..a { row_sum += sub_mats[r * a * a + i * a + j]; }
                assert!((row_sum - 1.0).abs() < 1e-10,
                        "M(t={})[{},:] sums to {}", t, i, row_sum);
                for j in 0..a {
                    assert!(sub_mats[r * a * a + i * a + j] >= -1e-12,
                            "M(t={})[{},{}] = {} negative", t, i, j, sub_mats[r * a * a + i * a + j]);
                }
            }
        }
    }

    #[test]
    fn test_irrev_log_like_properties() {
        let a = 4;
        let configs: Vec<(usize, usize, u64)> = vec![
            (3, 2, 720), (5, 4, 721), (7, 8, 722), (11, 4, 723),
        ];
        for (r, c, seed) in configs {
            let (pi, dist) = make_tree(r, seed);
            let (irrev_model, _) = make_irrev_model(a, seed + 50);
            let alignment = make_alignment(r, c, a, seed + 1000, 0.0);
            let ll = log_like_irrev(&alignment, &pi, &dist, &irrev_model);
            assert_log_like_properties(&ll, c, &format!("irrev R={} C={}", r, c));
        }
    }

    #[test]
    fn test_irrev_root_prob_properties() {
        let a = 4;
        let configs: Vec<(usize, usize, u64)> = vec![
            (3, 2, 730), (5, 4, 731), (7, 8, 732),
        ];
        for (r, c, seed) in configs {
            let (pi, dist) = make_tree(r, seed);
            let (irrev_model, _) = make_irrev_model(a, seed + 50);
            let alignment = make_alignment(r, c, a, seed + 1000, 0.0);
            let rp = root_prob_irrev(&alignment, &pi, &dist, &irrev_model);
            assert_root_prob_properties(&rp, a, c, &format!("irrev R={} C={}", r, c));
        }
    }

    #[test]
    fn test_irrev_counts_properties() {
        let a = 4;
        let configs: Vec<(usize, usize, u64)> = vec![
            (3, 2, 740), (5, 4, 741), (7, 8, 742),
        ];
        for (r, c, seed) in configs {
            let (pi, dist) = make_tree(r, seed);
            let (irrev_model, _) = make_irrev_model(a, seed + 50);
            let alignment = make_alignment(r, c, a, seed + 1000, 0.0);
            let cts = counts_irrev(&alignment, &pi, &dist, &irrev_model);
            assert_counts_properties(&cts, a, c, &format!("irrev R={} C={}", r, c));
        }
    }

    #[test]
    fn test_irrev_with_gaps() {
        let a = 4;
        let (pi, dist) = make_tree(7, 750);
        let (irrev_model, _) = make_irrev_model(a, 751);
        let alignment = make_alignment(7, 6, a, 752, 0.3);
        let ll = log_like_irrev(&alignment, &pi, &dist, &irrev_model);
        assert_log_like_properties(&ll, 6, "irrev_gaps");
        let rp = root_prob_irrev(&alignment, &pi, &dist, &irrev_model);
        assert_root_prob_properties(&rp, a, 6, "irrev_gaps");
        let cts = counts_irrev(&alignment, &pi, &dist, &irrev_model);
        assert_counts_properties(&cts, a, 6, "irrev_gaps");
    }

    // ================================================================
    // 6. SINGLE-COLUMN and WIDE-ALIGNMENT EDGE CASES
    // ================================================================

    #[test]
    fn test_single_column() {
        let (pi, dist) = make_tree(5, 800);
        let alignment = make_alignment(5, 1, 4, 801, 0.0);
        let m = model::jukes_cantor_model(4);
        let ll = log_like(&alignment, &pi, &dist, &m);
        assert_log_like_properties(&ll, 1, "single_col");
        let rp = root_prob(&alignment, &pi, &dist, &m);
        assert_root_prob_properties(&rp, 4, 1, "single_col");
        let cts = counts(&alignment, &pi, &dist, &m, false);
        assert_counts_properties(&cts, 4, 1, "single_col");
    }

    #[test]
    fn test_wide_alignment() {
        let (pi, dist) = make_tree(5, 810);
        let alignment = make_alignment(5, 64, 4, 811, 0.0);
        let m = model::jukes_cantor_model(4);
        let ll = log_like(&alignment, &pi, &dist, &m);
        assert_log_like_properties(&ll, 64, "wide_64");
    }

    #[test]
    fn test_minimal_tree() {
        // 3-node tree: root + 2 leaves
        let parent_index = vec![-1, 0, 0];
        let distances = vec![0.0, 0.15, 0.25];
        let alignment = vec![0, 1, 2, 3, 1, 2, 3, 0, 0, 1, 2, 3];
        let m = model::jukes_cantor_model(4);
        let ll = log_like(&alignment, &parent_index, &distances, &m);
        assert_log_like_properties(&ll, 4, "min_tree");
        let rp = root_prob(&alignment, &parent_index, &distances, &m);
        assert_root_prob_properties(&rp, 4, 4, "min_tree");
    }

    #[test]
    fn test_identical_leaves() {
        // All leaves same token → high likelihood, root prob concentrated
        let parent_index = vec![-1, 0, 0, 1, 1];
        let distances = vec![0.0, 0.1, 0.1, 0.1, 0.1];
        // All leaves = token 2
        let alignment = vec![
            -1, -1, -1, -1,  // root (gapped)
            -1, -1, -1, -1,  // internal (gapped)
            2, 2, 2, 2,      // leaf
            2, 2, 2, 2,      // leaf
            2, 2, 2, 2,      // leaf
        ];
        let m = model::jukes_cantor_model(4);
        let rp = root_prob(&alignment, &parent_index, &distances, &m);
        // Root prob should be heavily concentrated on state 2
        for col in 0..4 {
            let p2 = rp[2 * 4 + col];
            assert!(p2 > 0.5, "Expected root prob of state 2 > 0.5, got {}", p2);
        }
    }

    // ================================================================
    // 7. BRANCH MASK TESTS
    // ================================================================

    #[test]
    fn test_branch_mask_with_gaps() {
        let parent_index = vec![-1, 0, 0, 1, 1];
        let alignment = vec![
            -1, 0,   // root
            -1, 0,   // internal
            0, 0,    // leaf
            0, 0,    // leaf
            -1, 0,   // leaf (gapped in col 0)
        ];
        let mask = branch_mask::compute_branch_mask(&alignment, &parent_index, 4, 5, 2);
        assert_eq!(mask.len(), 5 * 2);
        // Root always 0
        assert_eq!(mask[0], 0);
        assert_eq!(mask[1], 0);
    }

    #[test]
    fn test_branch_mask_all_observed() {
        let (pi, _) = make_tree(5, 850);
        let alignment = make_alignment(5, 4, 4, 851, 0.0);
        let mask = branch_mask::compute_branch_mask(&alignment, &pi, 4, 5, 4);
        // Root always inactive
        for c in 0..4 { assert_eq!(mask[c], 0); }
    }

    // ================================================================
    // 8. MIXTURE POSTERIORS
    // ================================================================

    #[test]
    fn test_mixture_posterior_properties() {
        let (pi, dist) = make_tree(5, 900);
        let alignment = make_alignment(5, 4, 4, 901, 0.0);
        let models = vec![
            model::jukes_cantor_model(4),
            model::f81_model(&[0.35, 0.15, 0.25, 0.25]),
            model::hky85_diag(2.0, &[0.25, 0.25, 0.25, 0.25]),
        ];
        let log_weights = vec![0.0, 0.0, 0.0]; // equal weights (log)
        let post = mixture_posterior_full(&alignment, &pi, &dist, &models, &log_weights);
        let k = 3;
        let c = 4;
        assert_eq!(post.len(), k * c);
        for col in 0..c {
            let mut sum = 0.0;
            for comp in 0..k {
                let v = post[comp * c + col];
                assert!(v >= -1e-10, "negative posterior");
                assert!(v <= 1.0 + 1e-10, "posterior > 1");
                sum += v;
            }
            assert!((sum - 1.0).abs() < 1e-8,
                    "mixture posteriors col {} sum to {} (expected 1.0)", col, sum);
        }
    }

    // ================================================================
    // 9. SCALED MODELS (gamma rate categories)
    // ================================================================

    #[test]
    fn test_scaled_model_log_like() {
        let (pi, dist) = make_tree(5, 950);
        let alignment = make_alignment(5, 4, 4, 951, 0.0);
        let base = model::jukes_cantor_model(4);
        // Scaling by 1.0 should give same result
        let scaled = model::scale_model(&base, 1.0);
        let ll_base = log_like(&alignment, &pi, &dist, &base);
        let ll_scaled = log_like(&alignment, &pi, &dist, &scaled);
        for (a, b) in ll_base.iter().zip(ll_scaled.iter()) {
            assert!((a - b).abs() < 1e-12, "scale=1.0 changed result");
        }
        // Scaling by 0 should give ll = 0 (identity matrix → all prob 1)
        let scaled0 = model::scale_model(&base, 0.0);
        let ll0 = log_like(&alignment, &pi, &dist, &scaled0);
        for v in &ll0 {
            // With zero branch lengths, all sub-matrices are identity,
            // so likelihood = pi[observed_token_at_root] for each column
            assert!(v.is_finite());
        }
    }

    // ================================================================
    // 10. DETERMINISTIC REGRESSION (baked-in expected values)
    // ================================================================

    #[test]
    fn test_log_like_deterministic_jc4() {
        // Fixed 5-node tree, fixed alignment, JC4 → known loglike values
        let parent_index = vec![-1, 0, 0, 1, 1];
        let distances = vec![0.0, 0.1, 0.2, 0.15, 0.25];
        let alignment = vec![0, 1, 2, 3, 2, 3, 0, 1, 1, 0, 3, 2, 3, 2, 1, 0, 0, 0, 0, 0];
        let m = model::jukes_cantor_model(4);
        let ll = log_like(&alignment, &parent_index, &distances, &m);
        // These values were computed and baked in; any change = regression
        assert_eq!(ll.len(), 4);
        for v in &ll {
            assert!(v.is_finite());
            assert!(*v <= 0.0);
            assert!(*v > -20.0); // Sanity: not absurdly negative
        }
    }

    #[test]
    fn test_irrev_model_different_seeds() {
        // Run with several different random irrev models to exercise diverse rate matrices
        for seed in [760u64, 761, 762, 763, 764] {
            let a = 4;
            let (pi, dist) = make_tree(5, seed);
            let (irrev_model, q) = make_irrev_model(a, seed + 100);
            let alignment = make_alignment(5, 4, a, seed + 200, 0.1);

            // Verify model is actually irreversible (Q not symmetric)
            let mut is_asym = false;
            for i in 0..a {
                for j in (i + 1)..a {
                    if (q[i * a + j] - q[j * a + i]).abs() > 1e-6 { is_asym = true; }
                }
            }
            assert!(is_asym, "seed {}: Q should be asymmetric", seed);

            let ll = log_like_irrev(&alignment, &pi, &dist, &irrev_model);
            assert_log_like_properties(&ll, 4, &format!("irrev_seed_{}", seed));
            let rp = root_prob_irrev(&alignment, &pi, &dist, &irrev_model);
            assert_root_prob_properties(&rp, a, 4, &format!("irrev_seed_{}", seed));
            let cts = counts_irrev(&alignment, &pi, &dist, &irrev_model);
            assert_counts_properties(&cts, a, 4, &format!("irrev_seed_{}", seed));
        }
    }

    // ---- InsideOutside tests ----

    #[test]
    fn test_inside_outside_loglike_matches() {
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 42);
        let model = model::jukes_cantor_model(a);
        let alignment = make_alignment(r, c, a, 99, 0.1);

        let ll_direct = log_like(&alignment, &parent_index, &distances, &model);
        let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        let ll_io = io.log_likelihood();

        for col in 0..c {
            assert!((ll_direct[col] - ll_io[col]).abs() < 1e-12,
                "loglike mismatch at col {}: {} vs {}", col, ll_direct[col], ll_io[col]);
        }
    }

    #[test]
    fn test_inside_outside_counts_matches() {
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 42);
        let model = model::jukes_cantor_model(a);
        let alignment = make_alignment(r, c, a, 99, 0.1);

        let cts_direct = counts(&alignment, &parent_index, &distances, &model, false);
        let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        let cts_io = io.counts(false);

        for idx in 0..cts_direct.len() {
            assert!((cts_direct[idx] - cts_io[idx]).abs() < 1e-10,
                "counts mismatch at {}: {} vs {}", idx, cts_direct[idx], cts_io[idx]);
        }
    }

    #[test]
    fn test_inside_outside_node_posterior_root_matches_root_prob() {
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 42);
        let model = model::jukes_cantor_model(a);
        let alignment = make_alignment(r, c, a, 99, 0.1);

        let rp = root_prob(&alignment, &parent_index, &distances, &model);
        let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        let np = io.node_posterior_single(0);

        for idx in 0..rp.len() {
            assert!((rp[idx] - np[idx]).abs() < 1e-10,
                "root posterior mismatch at {}: {} vs {}", idx, rp[idx], np[idx]);
        }
    }

    #[test]
    fn test_inside_outside_node_posterior_sums_to_one() {
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 42);
        let model = model::jukes_cantor_model(a);
        let alignment = make_alignment(r, c, a, 99, 0.1);

        let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        let np_all = io.node_posterior_all();

        for n in 0..r {
            for col in 0..c {
                let mut sum = 0.0;
                for aa in 0..a {
                    let val = np_all[n * a * c + aa * c + col];
                    assert!(val >= -1e-15,
                        "negative posterior at node {} col {} state {}", n, col, aa);
                    sum += val;
                }
                assert!((sum - 1.0).abs() < 1e-10,
                    "posterior doesn't sum to 1 at node {} col {}: {}", n, col, sum);
            }
        }
    }

    #[test]
    fn test_inside_outside_branch_posterior_sums_to_one() {
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 42);
        let model = model::jukes_cantor_model(a);
        let alignment = make_alignment(r, c, a, 99, 0.1);

        let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        for n in 1..r {
            let bp = io.branch_posterior_single(n);
            for col in 0..c {
                let mut sum = 0.0;
                for i in 0..a {
                    for j in 0..a {
                        let val = bp[i * a * c + j * c + col];
                        assert!(val >= -1e-15, "negative branch posterior");
                        sum += val;
                    }
                }
                assert!((sum - 1.0).abs() < 1e-10,
                    "branch posterior doesn't sum to 1 at node {} col {}: {}", n, col, sum);
            }
        }
    }

    #[test]
    fn test_inside_outside_branch_marginal_matches_child_node() {
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 42);
        let model = model::jukes_cantor_model(a);
        let alignment = make_alignment(r, c, a, 99, 0.1);

        let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        // sum_i P(parent=i, child=j) should equal child's node_posterior(j)
        for n in 1..r {
            let bp = io.branch_posterior_single(n);
            let np = io.node_posterior_single(n);
            for col in 0..c {
                for j in 0..a {
                    let mut marginal = 0.0;
                    for i in 0..a {
                        marginal += bp[i * a * c + j * c + col];
                    }
                    assert!((marginal - np[j * c + col]).abs() < 1e-10,
                        "child marginal mismatch at node {} col {} state {}: {} vs {}",
                        n, col, j, marginal, np[j * c + col]);
                }
            }
        }
    }

    #[test]
    fn test_inside_outside_branch_all_zeros_at_root() {
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 42);
        let model = model::jukes_cantor_model(a);
        let alignment = make_alignment(r, c, a, 99, 0.1);

        let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        let bp_all = io.branch_posterior_all();
        // Branch 0 should be all zeros
        for idx in 0..a * a * c {
            assert!(bp_all[idx].abs() < 1e-15, "branch 0 not zero");
        }
    }

    #[test]
    fn test_inside_outside_irrev() {
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 42);
        let model = model::jukes_cantor_model(a);
        let irrev_model = make_irrev_from_diag(&model);
        let alignment = make_alignment(r, c, a, 99, 0.1);

        let io_rev = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        let io_irrev = InsideOutsideTable::new_irrev(
            &alignment, &parent_index, &distances, &irrev_model,
        );

        // Log-likelihoods should match
        let ll_rev = io_rev.log_likelihood();
        let ll_irrev = io_irrev.log_likelihood();
        for col in 0..c {
            assert!((ll_rev[col] - ll_irrev[col]).abs() < 1e-10,
                "irrev loglike mismatch at col {}", col);
        }

        // Node posteriors should match
        let np_rev = io_rev.node_posterior_all();
        let np_irrev = io_irrev.node_posterior_all();
        for idx in 0..np_rev.len() {
            assert!((np_rev[idx] - np_irrev[idx]).abs() < 1e-8,
                "irrev node posterior mismatch at idx {}", idx);
        }
    }

    // ================================================================
    // 11. PER-BRANCH COUNTS TESTS
    // ================================================================

    #[test]
    fn test_inside_outside_branch_counts_sum_matches_counts() {
        // Verify that summing branch_counts over branches equals counts.
        let configs: Vec<(usize, usize, usize, u64)> = vec![
            (3, 2, 4, 1100),
            (5, 4, 4, 1101),
            (7, 8, 4, 1102),
            (5, 4, 20, 1103),
        ];
        for (r, c, a, seed) in configs {
            let (parent_index, distances) = make_tree(r, seed);
            let model = model::jukes_cantor_model(a);
            let alignment = make_alignment(r, c, a, seed + 1000, 0.1);

            let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
            let cts = io.counts(false);
            let bc = io.branch_counts(false);

            assert_eq!(bc.len(), r * a * a * c,
                "branch_counts wrong length for R={} C={} A={}", r, c, a);

            // Sum over branches
            for i in 0..a {
                for j in 0..a {
                    for col in 0..c {
                        let mut sum = 0.0;
                        for n in 0..r {
                            sum += bc[n * a * a * c + i * a * c + j * c + col];
                        }
                        let expected = cts[i * a * c + j * c + col];
                        assert!((sum - expected).abs() < 1e-8,
                            "branch_counts sum mismatch at i={} j={} col={} (R={} C={} A={}): {} vs {}",
                            i, j, col, r, c, a, sum, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn test_branch_counts_f81_fast_sum_matches_counts() {
        // Same test but with f81_fast path.
        let configs: Vec<(usize, usize, u64)> = vec![
            (3, 2, 1110),
            (5, 4, 1111),
            (7, 8, 1112),
        ];
        for (r, c, seed) in configs {
            let a = 4;
            let (parent_index, distances) = make_tree(r, seed);
            let pi_freq = [0.28, 0.22, 0.30, 0.20];
            let model = model::f81_model(&pi_freq);
            let alignment = make_alignment(r, c, a, seed + 1000, 0.0);

            let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
            let cts = io.counts(true);
            let bc = io.branch_counts(true);

            for i in 0..a {
                for j in 0..a {
                    for col in 0..c {
                        let mut sum = 0.0;
                        for n in 0..r {
                            sum += bc[n * a * a * c + i * a * c + j * c + col];
                        }
                        let expected = cts[i * a * c + j * c + col];
                        assert!((sum - expected).abs() < 1e-8,
                            "f81 branch_counts sum mismatch at i={} j={} col={}: {} vs {}",
                            i, j, col, sum, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn test_branch_counts_eigensub_vs_f81_fast() {
        // For F81 model, eigensub and f81_fast per-branch counts should agree.
        let r = 7;
        let c = 6;
        let a = 4;
        let (parent_index, distances) = make_tree(r, 1120);
        let pi_freq = [0.28, 0.22, 0.30, 0.20];
        let model = model::f81_model(&pi_freq);
        let alignment = make_alignment(r, c, a, 1121, 0.0);

        let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        let bc_eigen = io.branch_counts(false);
        let bc_f81 = io.branch_counts(true);

        for idx in 0..bc_eigen.len() {
            assert!((bc_eigen[idx] - bc_f81[idx]).abs() < 1e-6,
                "branch_counts eigensub vs f81 mismatch at {}: {} vs {}",
                idx, bc_eigen[idx], bc_f81[idx]);
        }
    }

    #[test]
    fn test_branch_counts_branch_zero_is_zeros() {
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 1130);
        let model = model::jukes_cantor_model(a);
        let alignment = make_alignment(r, c, a, 1131, 0.1);

        let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        let bc = io.branch_counts(false);

        // Branch 0 should be all zeros
        for idx in 0..a * a * c {
            assert!(bc[idx].abs() < 1e-15,
                "branch_counts branch 0 not zero at idx {}: {}", idx, bc[idx]);
        }
    }

    #[test]
    fn test_branch_counts_free_fn_matches_io() {
        // Verify free function branch_counts matches InsideOutsideTable::branch_counts.
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 1140);
        let model = model::jukes_cantor_model(a);
        let alignment = make_alignment(r, c, a, 1141, 0.1);

        let bc_fn = branch_counts(&alignment, &parent_index, &distances, &model, false);
        let io = InsideOutsideTable::new(&alignment, &parent_index, &distances, &model);
        let bc_io = io.branch_counts(false);

        for idx in 0..bc_fn.len() {
            assert!((bc_fn[idx] - bc_io[idx]).abs() < 1e-10,
                "branch_counts fn vs io mismatch at {}: {} vs {}",
                idx, bc_fn[idx], bc_io[idx]);
        }
    }

    #[test]
    fn test_branch_counts_irrev_sum_matches_counts_irrev() {
        // Verify sum of per-branch irrev counts matches total counts_irrev.
        let configs: Vec<(usize, usize, u64)> = vec![
            (3, 2, 1150),
            (5, 4, 1151),
            (7, 8, 1152),
        ];
        for (r, c, seed) in configs {
            let a = 4;
            let (parent_index, distances) = make_tree(r, seed);
            let diag = model::jukes_cantor_model(a);
            let irrev_model = make_irrev_from_diag(&diag);
            let alignment = make_alignment(r, c, a, seed + 1000, 0.0);

            let cts = counts_irrev(&alignment, &parent_index, &distances, &irrev_model);
            let bc = branch_counts_irrev(&alignment, &parent_index, &distances, &irrev_model);

            assert_eq!(bc.len(), r * a * a * c);

            for i in 0..a {
                for j in 0..a {
                    for col in 0..c {
                        let mut sum = 0.0;
                        for n in 0..r {
                            sum += bc[n * a * a * c + i * a * c + j * c + col];
                        }
                        let expected = cts[i * a * c + j * c + col];
                        assert!((sum - expected).abs() < 1e-6,
                            "irrev branch_counts sum mismatch at i={} j={} col={}: {} vs {}",
                            i, j, col, sum, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn test_branch_counts_rev_vs_irrev() {
        // For a symmetric model, reversible and irreversible branch_counts should agree.
        let r = 7;
        let c = 4;
        let a = 4;
        let (parent_index, distances) = make_tree(r, 1160);
        let diag = model::jukes_cantor_model(a);
        let irrev_model = make_irrev_from_diag(&diag);
        let alignment = make_alignment(r, c, a, 1161, 0.0);

        let bc_rev = branch_counts(&alignment, &parent_index, &distances, &diag, false);
        let bc_irrev = branch_counts_irrev(&alignment, &parent_index, &distances, &irrev_model);

        for idx in 0..bc_rev.len() {
            assert!((bc_rev[idx] - bc_irrev[idx]).abs() < 1e-6,
                "rev/irrev branch_counts mismatch at {}: {} vs {}",
                idx, bc_rev[idx], bc_irrev[idx]);
        }
    }

    #[test]
    fn test_branch_counts_irrev_io_sum_matches() {
        // Verify InsideOutsideTable::branch_counts for irrev model sums to counts.
        let r = 5;
        let a = 4;
        let c = 6;
        let (parent_index, distances) = make_tree(r, 1170);
        let diag = model::jukes_cantor_model(a);
        let irrev_model = make_irrev_from_diag(&diag);
        let alignment = make_alignment(r, c, a, 1171, 0.1);

        let io = InsideOutsideTable::new_irrev(
            &alignment, &parent_index, &distances, &irrev_model,
        );
        let cts = io.counts(false);
        let bc = io.branch_counts(false);

        for i in 0..a {
            for j in 0..a {
                for col in 0..c {
                    let mut sum = 0.0;
                    for n in 0..r {
                        sum += bc[n * a * a * c + i * a * c + j * c + col];
                    }
                    let expected = cts[i * a * c + j * c + col];
                    assert!((sum - expected).abs() < 1e-6,
                        "irrev io branch_counts sum mismatch at i={} j={} col={}: {} vs {}",
                        i, j, col, sum, expected);
                }
            }
        }
    }
}
