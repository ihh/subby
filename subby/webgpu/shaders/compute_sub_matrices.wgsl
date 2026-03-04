// Compute substitution probability matrices M(t) from eigendecomposition.
// M_ij(t) = sqrt(pi_j/pi_i) * sum_k v_ik * exp(mu_k * t) * v_jk
// Dispatch: ceil(R / 64) — each thread computes one A×A matrix for one branch

struct Params {
    R: u32,
    A: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> eigenvalues: array<f32>;   // (A)
@group(0) @binding(2) var<storage, read> eigenvectors: array<f32>;  // (A*A) row-major: v[a*A+k]
@group(0) @binding(3) var<storage, read> pi: array<f32>;            // (A)
@group(0) @binding(4) var<storage, read> distances: array<f32>;     // (R)
@group(0) @binding(5) var<storage, read_write> M: array<f32>;       // (R*A*A)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let r = gid.x;
    if (r >= params.R) { return; }

    let A = params.A;
    let t = distances[r];

    for (var i = 0u; i < A; i = i + 1u) {
        let inv_sqrt_pi_i = 1.0 / sqrt(pi[i]);
        for (var j = 0u; j < A; j = j + 1u) {
            let sqrt_pi_j = sqrt(pi[j]);
            var s: f32 = 0.0;
            for (var k = 0u; k < A; k = k + 1u) {
                s += eigenvectors[i * A + k] * exp(eigenvalues[k] * t) * eigenvectors[j * A + k];
            }
            M[r * A * A + i * A + j] = inv_sqrt_pi_i * sqrt_pi_j * s;
        }
    }
}
