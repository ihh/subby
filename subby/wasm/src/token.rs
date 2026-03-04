/// Convert integer token alignment to likelihood vectors.

/// Token encoding:
///   0..A-1 : observed (one-hot)
///   A      : ungapped-unobserved (all ones)
///   A+1    : gapped (all ones)
///   -1     : gap (legacy, all ones)
///
/// Returns flat (R*C*A) f64 array, row-major.
pub fn token_to_likelihood(alignment: &[i32], r: usize, c: usize, a: usize) -> Vec<f64> {
    let mut l = vec![0.0; r * c * a];
    for row in 0..r {
        for col in 0..c {
            let tok = alignment[row * c + col];
            let base = (row * c + col) * a;
            if tok >= 0 && (tok as usize) < a {
                l[base + tok as usize] = 1.0;
            } else {
                for s in 0..a {
                    l[base + s] = 1.0;
                }
            }
        }
    }
    l
}
