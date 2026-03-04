// Accumulate eigenbasis counts C_{kl} for irreversible model (complex).
// C_{kl,c} = sum_{n>0} D_tilde_k^(n) * J^{kl}(t_n) * U_tilde_l^(n) * scale[n,c]
// All complex except scale. Output: 2*A*A*C interleaved f32.
// Dispatch: ceil(A * A * C / 64)

struct Params {
    R: u32,
    C: u32,
    A: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> D_tilde: array<f32>;    // (2*R*C*A) interleaved
@group(0) @binding(2) var<storage, read> U_tilde: array<f32>;    // (2*R*C*A) interleaved
@group(0) @binding(3) var<storage, read> J: array<f32>;          // (2*R*A*A) interleaved
@group(0) @binding(4) var<storage, read> logNormU: array<f32>;   // (R*C)
@group(0) @binding(5) var<storage, read> logNormD: array<f32>;   // (R*C)
@group(0) @binding(6) var<storage, read> logLike: array<f32>;    // (C)
@group(0) @binding(7) var<storage, read_write> C_out: array<f32>; // (2*A*A*C) interleaved

fn cmul_local(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let A = params.A;
    let total = A * A * params.C;
    if (idx >= total) { return; }

    let k = idx / (A * params.C);
    let rem = idx % (A * params.C);
    let l = rem / params.C;
    let c = rem % params.C;

    var acc = vec2<f32>(0.0, 0.0);

    for (var n = 1u; n < params.R; n = n + 1u) {
        let log_s = logNormD[n * params.C + c] + logNormU[n * params.C + c] - logLike[c];
        let scale = exp(log_s);

        let dt_idx = (n * params.C + c) * A + k;
        let dt_k = vec2(D_tilde[2u * dt_idx], D_tilde[2u * dt_idx + 1u]);

        let ut_idx = (n * params.C + c) * A + l;
        let ut_l = vec2(U_tilde[2u * ut_idx], U_tilde[2u * ut_idx + 1u]);

        let j_idx = n * A * A + k * A + l;
        let j_kl = vec2(J[2u * j_idx], J[2u * j_idx + 1u]);

        let contrib = cmul_local(cmul_local(dt_k, j_kl), ut_l);
        acc += vec2(contrib.x * scale, contrib.y * scale);
    }

    let out_idx = k * A * params.C + l * params.C + c;
    C_out[2u * out_idx] = acc.x;
    C_out[2u * out_idx + 1u] = acc.y;
}
