// Project inside/outside vectors into eigenbasis for irreversible model.
// U_tilde_k = sum_b U_b * V_bk          (no pi weighting)
// D_tilde_k = sum_a D_a * V_inv_ka      (using V^{-1})
// Output is complex (interleaved), 2*R*C*A f32.
// Dispatch: ceil(R * C / 64)

struct Params {
    R: u32,
    C: u32,
    A: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> eigenvectors: array<f32>;       // (2*A*A) interleaved
@group(0) @binding(2) var<storage, read> eigenvectors_inv: array<f32>;   // (2*A*A) interleaved
@group(0) @binding(3) var<storage, read> U: array<f32>;                   // (R*C*A) real
@group(0) @binding(4) var<storage, read> D: array<f32>;                   // (R*C*A) real
@group(0) @binding(5) var<storage, read_write> U_tilde: array<f32>;      // (2*R*C*A) interleaved
@group(0) @binding(6) var<storage, read_write> D_tilde: array<f32>;      // (2*R*C*A) interleaved

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.R * params.C;
    if (idx >= total) { return; }

    let r = idx / params.C;
    let c = idx % params.C;
    let A = params.A;
    let base_real = (r * params.C + c) * A;
    let base_complex = (r * params.C + c) * A;

    for (var k = 0u; k < A; k = k + 1u) {
        var su = vec2<f32>(0.0, 0.0);
        var sd = vec2<f32>(0.0, 0.0);
        for (var b = 0u; b < A; b = b + 1u) {
            let u_b = U[base_real + b];
            let d_b = D[base_real + b];
            // V[b,k]
            let v_bk = vec2(eigenvectors[2u * (b * A + k)], eigenvectors[2u * (b * A + k) + 1u]);
            su += vec2(u_b * v_bk.x, u_b * v_bk.y);
            // V_inv[k,b]
            let vinv_kb = vec2(eigenvectors_inv[2u * (k * A + b)], eigenvectors_inv[2u * (k * A + b) + 1u]);
            sd += vec2(d_b * vinv_kb.x, d_b * vinv_kb.y);
        }
        U_tilde[2u * (base_complex + k)] = su.x;
        U_tilde[2u * (base_complex + k) + 1u] = su.y;
        D_tilde[2u * (base_complex + k)] = sd.x;
        D_tilde[2u * (base_complex + k) + 1u] = sd.y;
    }
}
