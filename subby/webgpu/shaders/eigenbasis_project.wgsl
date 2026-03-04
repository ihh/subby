// Project inside/outside vectors into eigenbasis.
// U_tilde_l = sum_b U_b * v_{bl} * sqrt(pi_b)
// D_tilde_k = sum_a D_a * v_{ak} / sqrt(pi_a)
// Dispatch: ceil(R * C / 64) — each thread handles (r, c) pair

struct Params {
    R: u32,
    C: u32,
    A: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> eigenvectors: array<f32>;    // (A*A)
@group(0) @binding(2) var<storage, read> pi: array<f32>;              // (A)
@group(0) @binding(3) var<storage, read> U: array<f32>;               // (R*C*A)
@group(0) @binding(4) var<storage, read> D: array<f32>;               // (R*C*A)
@group(0) @binding(5) var<storage, read_write> U_tilde: array<f32>;   // (R*C*A)
@group(0) @binding(6) var<storage, read_write> D_tilde: array<f32>;   // (R*C*A)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.R * params.C;
    if (idx >= total) { return; }

    let r = idx / params.C;
    let c = idx % params.C;
    let A = params.A;
    let base = (r * params.C + c) * A;

    for (var k = 0u; k < A; k = k + 1u) {
        var su: f32 = 0.0;
        var sd: f32 = 0.0;
        for (var b = 0u; b < A; b = b + 1u) {
            let v = eigenvectors[b * A + k];
            su += U[base + b] * v * sqrt(pi[b]);
            sd += D[base + b] * v / sqrt(pi[b]);
        }
        U_tilde[base + k] = su;
        D_tilde[base + k] = sd;
    }
}
