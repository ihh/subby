// Compute substitution probability matrices for irreversible model.
// M_ij(t) = Re(sum_k V_ik * exp(mu_k * t) * V_inv_kj)
// Dispatch: ceil(R / 64)

struct Params {
    R: u32,
    A: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> eigenvalues: array<f32>;       // (2*A) interleaved
@group(0) @binding(2) var<storage, read> eigenvectors: array<f32>;      // (2*A*A) interleaved
@group(0) @binding(3) var<storage, read> eigenvectors_inv: array<f32>;  // (2*A*A) interleaved
@group(0) @binding(4) var<storage, read> distances: array<f32>;          // (R)
@group(0) @binding(5) var<storage, read_write> M: array<f32>;           // (R*A*A) real output

fn cmul_local(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cexp_local(a: vec2<f32>) -> vec2<f32> {
    let r = exp(a.x);
    return vec2(r * cos(a.y), r * sin(a.y));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let r = gid.x;
    if (r >= params.R) { return; }

    let A = params.A;
    let t = distances[r];

    for (var i = 0u; i < A; i = i + 1u) {
        for (var j = 0u; j < A; j = j + 1u) {
            var s = vec2<f32>(0.0, 0.0);
            for (var k = 0u; k < A; k = k + 1u) {
                let v_ik = vec2(eigenvectors[2u * (i * A + k)], eigenvectors[2u * (i * A + k) + 1u]);
                let mu_k = vec2(eigenvalues[2u * k], eigenvalues[2u * k + 1u]);
                let exp_k = cexp_local(vec2(mu_k.x * t, mu_k.y * t));
                let vinv_kj = vec2(eigenvectors_inv[2u * (k * A + j)], eigenvectors_inv[2u * (k * A + j) + 1u]);
                s += cmul_local(cmul_local(v_ik, exp_k), vinv_kj);
            }
            M[r * A * A + i * A + j] = s.x; // Real part
        }
    }
}
