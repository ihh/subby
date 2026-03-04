/// Complex arithmetic helpers operating on (re, im) tuples.
///
/// Used by the irreversible model path where eigenvalues/eigenvectors
/// are complex-valued but stored as interleaved f64 pairs.

pub type C64 = (f64, f64);

pub const CZERO: C64 = (0.0, 0.0);

#[inline]
pub fn cmul(a: C64, b: C64) -> C64 {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
pub fn cadd(a: C64, b: C64) -> C64 {
    (a.0 + b.0, a.1 + b.1)
}

#[inline]
pub fn csub(a: C64, b: C64) -> C64 {
    (a.0 - b.0, a.1 - b.1)
}

#[inline]
pub fn cdiv(a: C64, b: C64) -> C64 {
    let denom = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / denom, (a.1 * b.0 - a.0 * b.1) / denom)
}

#[inline]
pub fn cexp(a: C64) -> C64 {
    let r = a.0.exp();
    (r * a.1.cos(), r * a.1.sin())
}

#[inline]
pub fn cabs(a: C64) -> f64 {
    (a.0 * a.0 + a.1 * a.1).sqrt()
}

#[inline]
pub fn cscale(a: C64, s: f64) -> C64 {
    (a.0 * s, a.1 * s)
}

/// Load complex value from interleaved buffer at index i.
#[inline]
pub fn cload(buf: &[f64], i: usize) -> C64 {
    (buf[2 * i], buf[2 * i + 1])
}

/// Store complex value to interleaved buffer at index i.
#[inline]
pub fn cstore(buf: &mut [f64], i: usize, v: C64) {
    buf[2 * i] = v.0;
    buf[2 * i + 1] = v.1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmul() {
        // (1+2i) * (3+4i) = -5+10i
        let r = cmul((1.0, 2.0), (3.0, 4.0));
        assert!((r.0 - (-5.0)).abs() < 1e-12);
        assert!((r.1 - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_cexp() {
        // exp(0) = 1
        let r = cexp((0.0, 0.0));
        assert!((r.0 - 1.0).abs() < 1e-12);
        assert!(r.1.abs() < 1e-12);
        // exp(i*pi) = -1
        let r2 = cexp((0.0, std::f64::consts::PI));
        assert!((r2.0 - (-1.0)).abs() < 1e-12);
        assert!(r2.1.abs() < 1e-12);
    }

    #[test]
    fn test_cdiv() {
        // (1+2i) / (1+0i) = (1+2i)
        let r = cdiv((1.0, 2.0), (1.0, 0.0));
        assert!((r.0 - 1.0).abs() < 1e-12);
        assert!((r.1 - 2.0).abs() < 1e-12);
    }
}
