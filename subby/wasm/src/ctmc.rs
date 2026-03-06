/// Standalone CTMC branch integral functions.
///
/// Computes expected substitution counts and dwell times for a single
/// continuous-time Markov chain (CTMC) branch, independent of any alignment
/// or tree.
///
/// Returns flat (A*A*A*A) with layout result[a*A*A*A + b*A*A + i*A + j].

use crate::complex::*;

/// Expected counts from pre-computed eigendecomposition (reversible).
///
/// Returns flat (A^4) array: result[a*A^3 + b*A^2 + i*A + j].
pub fn expected_counts_eigen(
    eigenvalues: &[f64],
    eigenvectors: &[f64],
    _pi: &[f64],
    t: f64,
    a: usize,
) -> Vec<f64> {
    let mu = eigenvalues;
    let v = eigenvectors; // row-major: v[x*a + k]

    // J matrix (A*A)
    let mut j_mat = vec![0.0; a * a];
    for k in 0..a {
        let exp_k = (mu[k] * t).exp();
        for l in 0..a {
            let diff = mu[k] - mu[l];
            if diff.abs() < 1e-8 {
                j_mat[k * a + l] = t * exp_k;
            } else {
                j_mat[k * a + l] = (exp_k - (mu[l] * t).exp()) / diff;
            }
        }
    }

    // S_t[x, y] = sum_k V[x,k] exp(mu_k*t) V[y,k]
    let mut s_t = vec![0.0; a * a];
    for x in 0..a {
        for y in 0..a {
            let mut s = 0.0;
            for k in 0..a {
                s += v[x * a + k] * (mu[k] * t).exp() * v[y * a + k];
            }
            s_t[x * a + y] = s;
        }
    }

    // S_rate[i,j] = sum_k V[i,k] mu_k V[j,k]
    let mut s_rate = vec![0.0; a * a];
    for i in 0..a {
        for j in 0..a {
            let mut s = 0.0;
            for k in 0..a {
                s += v[i * a + k] * mu[k] * v[j * a + k];
            }
            s_rate[i * a + j] = s;
        }
    }

    // W[aa,bb,ii,jj] = sum_{kl} V[aa,k]*V[ii,k] * J[k,l] * V[bb,l]*V[jj,l]
    // Then result = dwell or subs scaled by M
    let a4 = a * a * a * a;
    let mut result = vec![0.0; a4];

    for aa in 0..a {
        for bb in 0..a {
            let st_ab = s_t[aa * a + bb];
            if st_ab.abs() < 1e-300 {
                continue;
            }
            for ii in 0..a {
                for jj in 0..a {
                    let mut w = 0.0;
                    for k in 0..a {
                        for l in 0..a {
                            w += v[aa * a + k] * v[ii * a + k]
                                * j_mat[k * a + l]
                                * v[bb * a + l] * v[jj * a + l];
                        }
                    }
                    let idx = aa * a * a * a + bb * a * a + ii * a + jj;
                    if ii == jj {
                        result[idx] = w / st_ab;
                    } else {
                        result[idx] = s_rate[ii * a + jj] * w / st_ab;
                    }
                }
            }
        }
    }

    result
}

/// Expected counts from pre-computed eigendecomposition (irreversible).
///
/// Returns flat (A^4) real array: result[a*A^3 + b*A^2 + i*A + j].
pub fn expected_counts_eigen_irrev(
    eigenvalues: &[f64],       // interleaved (2*A)
    eigenvectors: &[f64],      // interleaved (2*A*A), row-major
    eigenvectors_inv: &[f64],  // interleaved (2*A*A), row-major
    pi: &[f64],
    t: f64,
    a: usize,
) -> Vec<f64> {
    // J matrix (A*A) complex
    let mut j_mat = vec![0.0; 2 * a * a];
    for k in 0..a {
        let mu_k = cload(eigenvalues, k);
        let exp_k = cexp(cscale(mu_k, t));
        for l in 0..a {
            let mu_l = cload(eigenvalues, l);
            let diff = csub(mu_k, mu_l);
            let idx = k * a + l;
            if cabs(diff) < 1e-8 {
                cstore(&mut j_mat, idx, cscale(exp_k, t));
            } else {
                let exp_l = cexp(cscale(mu_l, t));
                cstore(&mut j_mat, idx, cdiv(csub(exp_k, exp_l), diff));
            }
        }
    }

    // M[aa,bb] = Re(sum_k V[aa,k] exp(mu_k*t) V_inv[k,bb])
    let mut m_mat = vec![0.0; a * a];
    for aa in 0..a {
        for bb in 0..a {
            let mut s = CZERO;
            for k in 0..a {
                let v_ak = cload(eigenvectors, aa * a + k);
                let mu_k = cload(eigenvalues, k);
                let exp_k = cexp(cscale(mu_k, t));
                let vinv_kb = cload(eigenvectors_inv, k * a + bb);
                s = cadd(s, cmul(cmul(v_ak, exp_k), vinv_kb));
            }
            m_mat[aa * a + bb] = s.0; // Re
        }
    }

    // Q[ii,jj] = sum_k V[ii,k] mu_k V_inv[k,jj]
    let mut q_mat = vec![0.0; 2 * a * a];
    for ii in 0..a {
        for jj in 0..a {
            let mut s = CZERO;
            for k in 0..a {
                let v_ik = cload(eigenvectors, ii * a + k);
                let mu_k = cload(eigenvalues, k);
                let vinv_kj = cload(eigenvectors_inv, k * a + jj);
                s = cadd(s, cmul(cmul(v_ik, mu_k), vinv_kj));
            }
            cstore(&mut q_mat, ii * a + jj, s);
        }
    }

    let a4 = a * a * a * a;
    let mut result = vec![0.0; a4];

    for aa in 0..a {
        for bb in 0..a {
            let m_ab = m_mat[aa * a + bb];
            if m_ab.abs() < 1e-300 {
                continue;
            }
            for ii in 0..a {
                for jj in 0..a {
                    // W = sum_{kl} V[aa,k]*V_inv[k,ii] * J[k,l] * V[jj,l]*V_inv[l,bb]
                    let mut w = CZERO;
                    for k in 0..a {
                        let v_ak = cload(eigenvectors, aa * a + k);
                        let vinv_ki = cload(eigenvectors_inv, k * a + ii);
                        for l in 0..a {
                            let j_kl = cload(&j_mat, k * a + l);
                            let v_jl = cload(eigenvectors, jj * a + l);
                            let vinv_lb = cload(eigenvectors_inv, l * a + bb);
                            w = cadd(w, cmul(cmul(cmul(v_ak, vinv_ki), j_kl),
                                             cmul(v_jl, vinv_lb)));
                        }
                    }
                    let idx = aa * a * a * a + bb * a * a + ii * a + jj;
                    if ii == jj {
                        result[idx] = w.0 / m_ab; // Re(W) / M
                    } else {
                        let q_ij = cload(&q_mat, ii * a + jj);
                        result[idx] = cmul(q_ij, w).0 / m_ab; // Re(Q*W) / M
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::jukes_cantor_model;

    #[test]
    fn test_dwell_sum_to_t() {
        let a = 4;
        let model = jukes_cantor_model(a);
        let t = 0.3;
        let result = expected_counts_eigen(
            &model.eigenvalues, &model.eigenvectors, &model.pi, t, a,
        );

        // For each (aa,bb) with M[aa,bb] > 0, sum_i result[aa,bb,i,i] = t
        // Compute M to check reachability
        for aa in 0..a {
            for bb in 0..a {
                let mut m_ab = 0.0;
                for k in 0..a {
                    m_ab += model.eigenvectors[aa * a + k]
                        * (model.eigenvalues[k] * t).exp()
                        * model.eigenvectors[bb * a + k];
                }
                // unsymmetrize
                let sqrt_pi_b = model.pi[bb].sqrt();
                let inv_sqrt_pi_a = 1.0 / model.pi[aa].sqrt();
                m_ab *= inv_sqrt_pi_a * sqrt_pi_b;

                if m_ab.abs() < 1e-10 {
                    continue;
                }

                let mut dwell_sum = 0.0;
                for ii in 0..a {
                    dwell_sum += result[aa * a * a * a + bb * a * a + ii * a + ii];
                }
                assert!(
                    (dwell_sum - t).abs() < 1e-10,
                    "dwell sum {} != t {} for ({}, {})",
                    dwell_sum, t, aa, bb
                );
            }
        }
    }

    #[test]
    fn test_nonnegative() {
        let a = 4;
        let model = jukes_cantor_model(a);
        let t = 0.5;
        let result = expected_counts_eigen(
            &model.eigenvalues, &model.eigenvectors, &model.pi, t, a,
        );
        for val in &result {
            assert!(*val >= -1e-10, "negative value: {}", val);
        }
    }
}
