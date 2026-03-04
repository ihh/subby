/// Model parameterization: eigendecomposed substitution models.

use std::f64::consts::PI as STD_PI;

/// Diagonalized substitution model.
#[derive(Clone, Debug)]
pub struct DiagModel {
    pub eigenvalues: Vec<f64>,   // (A,)
    pub eigenvectors: Vec<f64>,  // (A*A,) row-major: v[a*A+k]
    pub pi: Vec<f64>,            // (A,)
}

/// Irreversible diagonalized substitution model.
/// Complex arrays stored as interleaved (re, im) pairs.
#[derive(Clone, Debug)]
pub struct IrrevDiagModel {
    pub eigenvalues_complex: Vec<f64>,       // (2*A,) interleaved re/im
    pub eigenvectors_complex: Vec<f64>,      // (2*A*A,) interleaved re/im, row-major
    pub eigenvectors_inv_complex: Vec<f64>,  // (2*A*A,) interleaved re/im, row-major
    pub pi: Vec<f64>,                        // (A,)
}

/// HKY85 model with closed-form eigendecomposition.
pub fn hky85_diag(kappa: f64, pi: &[f64]) -> DiagModel {
    let (pi_a, pi_c, pi_g, pi_t) = (pi[0], pi[1], pi[2], pi[3]);
    let pi_r = pi_a + pi_g;
    let pi_y = pi_c + pi_t;

    let beta = 1.0 / (2.0 * pi_r * pi_y + 2.0 * kappa * (pi_a * pi_g + pi_c * pi_t));

    let eigenvalues = vec![
        0.0,
        -beta,
        -beta * (pi_r + kappa * pi_y),
        -beta * (pi_y + kappa * pi_r),
    ];

    let sqrt_pi: Vec<f64> = pi.iter().map(|&p| p.sqrt()).collect();

    // w0 = sqrt(pi)
    let w0 = sqrt_pi.clone();

    // w1: purine-pyrimidine
    let mut w1 = vec![
        sqrt_pi[0] * pi_y,
        -sqrt_pi[1] * pi_r,
        sqrt_pi[2] * pi_y,
        -sqrt_pi[3] * pi_r,
    ];
    let norm1 = (pi_r * pi_y).sqrt();
    for v in &mut w1 { *v /= norm1; }

    // w2: within-pyrimidine
    let mut w2 = vec![
        0.0,
        sqrt_pi[1] * pi_t,
        0.0,
        -sqrt_pi[3] * pi_c,
    ];
    let norm2 = (pi_c * pi_t * pi_y).sqrt();
    for v in &mut w2 { *v /= norm2; }

    // w3: within-purine
    let mut w3 = vec![
        sqrt_pi[0] * pi_g,
        0.0,
        -sqrt_pi[2] * pi_a,
        0.0,
    ];
    let norm3 = (pi_a * pi_g * pi_r).sqrt();
    for v in &mut w3 { *v /= norm3; }

    // eigenvectors: (4,4) row-major, eigenvectors[a*4+k]
    let mut eigenvectors = vec![0.0; 16];
    for a in 0..4 {
        eigenvectors[a * 4 + 0] = w0[a];
        eigenvectors[a * 4 + 1] = w1[a];
        eigenvectors[a * 4 + 2] = w2[a];
        eigenvectors[a * 4 + 3] = w3[a];
    }

    DiagModel {
        eigenvalues,
        eigenvectors,
        pi: pi.to_vec(),
    }
}

/// Jukes-Cantor model for an A-state alphabet.
pub fn jukes_cantor_model(a: usize) -> DiagModel {
    let pi = vec![1.0 / a as f64; a];
    let mu = a as f64 / (a as f64 - 1.0);
    let mut eigenvalues = vec![0.0; a];
    for i in 1..a {
        eigenvalues[i] = -mu;
    }

    // Eigenvectors via QR of [v0 | I]
    let v0: Vec<f64> = vec![1.0 / (a as f64).sqrt(); a];
    let eigenvectors = qr_orthonormal_basis(&v0, a);

    DiagModel { eigenvalues, eigenvectors, pi }
}

/// F81 model.
pub fn f81_model(pi: &[f64]) -> DiagModel {
    let a = pi.len();
    let pi_sq_sum: f64 = pi.iter().map(|&p| p * p).sum();
    let mu = 1.0 / (1.0 - pi_sq_sum);
    let mut eigenvalues = vec![0.0; a];
    for i in 1..a {
        eigenvalues[i] = -mu;
    }

    let sqrt_pi: Vec<f64> = pi.iter().map(|&p| p.sqrt()).collect();
    let eigenvectors = qr_orthonormal_basis(&sqrt_pi, a);

    DiagModel { eigenvalues, eigenvectors, pi: pi.to_vec() }
}

/// Build orthonormal basis with v0 as first column via modified Gram-Schmidt.
fn qr_orthonormal_basis(v0: &[f64], a: usize) -> Vec<f64> {
    // Augmented matrix: [v0 | I] of shape (A, A+1)
    // QR decomposition to get A orthonormal columns
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(a + 1);
    cols.push(v0.to_vec());
    for i in 0..a {
        let mut col = vec![0.0; a];
        col[i] = 1.0;
        cols.push(col);
    }

    // Modified Gram-Schmidt
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(a);
    for col in &cols {
        if basis.len() >= a { break; }
        let mut v = col.clone();
        // Orthogonalize against existing basis
        for b in &basis {
            let dot: f64 = v.iter().zip(b).map(|(a, b)| a * b).sum();
            for (vi, bi) in v.iter_mut().zip(b) {
                *vi -= dot * bi;
            }
        }
        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for vi in &mut v { *vi /= norm; }
            basis.push(v);
        }
    }

    // Flatten to row-major (A*A)
    let mut result = vec![0.0; a * a];
    for row in 0..a {
        for col in 0..a {
            result[row * a + col] = basis[col][row];
        }
    }
    result
}

/// Discretized gamma rate categories (Yang 1994).
pub fn gamma_rate_categories(alpha: f64, k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut rates = vec![0.0; k];
    for i in 0..k {
        let midpoint = (i as f64 + 0.5) / k as f64;
        rates[i] = gamma_quantile(alpha, midpoint);
    }
    let sum: f64 = rates.iter().sum();
    for r in &mut rates {
        *r *= k as f64 / sum;
    }
    let weights = vec![1.0 / k as f64; k];
    (rates, weights)
}

fn gamma_quantile(alpha: f64, p: f64) -> f64 {
    // Bisection: find x such that gammainc(alpha, x*alpha) = p
    let mut lo = 0.0_f64;
    let mut hi = f64::max(50.0, 10.0 / alpha);
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        let cdf = regularized_gamma_p(alpha, mid * alpha);
        if cdf < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Regularized lower incomplete gamma function P(a, x) = gamma(a,x)/Gamma(a).
/// Simple series expansion.
fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if x == 0.0 { return 0.0; }

    // Use series expansion for x < a+1
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut term = 1.0 / a;
        for n in 1..200 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-15 * sum.abs() { break; }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Use continued fraction for x >= a+1
        1.0 - regularized_gamma_q_cf(a, x)
    }
}

fn regularized_gamma_q_cf(a: f64, x: f64) -> f64 {
    // Lentz continued fraction
    let mut c = 1e-30_f64;
    let mut d = 1.0 / (x + 1.0 - a);
    let mut f = d;
    for n in 1..200 {
        let an = -(n as f64) * (n as f64 - a);
        let bn = x + 2.0 * n as f64 + 1.0 - a;
        d = 1.0 / (bn + an * d);
        c = bn + an / c;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-15 { break; }
    }
    f * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Log-gamma function (Stirling approximation).
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let mut sum = c[0];
    for i in 1..9 {
        sum += c[i] / (x + i as f64);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * STD_PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Scale eigenvalues by a rate multiplier.
pub fn scale_model(model: &DiagModel, rate_multiplier: f64) -> DiagModel {
    DiagModel {
        eigenvalues: model.eigenvalues.iter().map(|&e| e * rate_multiplier).collect(),
        eigenvectors: model.eigenvectors.clone(),
        pi: model.pi.clone(),
    }
}
