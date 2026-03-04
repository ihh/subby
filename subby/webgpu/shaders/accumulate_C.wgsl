// Accumulate eigenbasis counts C_{kl} over branches.
// C_{kl,c} = sum_{n>0} D_tilde_k^(n) * J^{kl}(t_n) * U_tilde_l^(n) * scale[n,c]
// Dispatch: ceil(A * A * C / 64)

struct Params {
    R: u32,
    C: u32,
    A: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> D_tilde: array<f32>;    // (R*C*A)
@group(0) @binding(2) var<storage, read> U_tilde: array<f32>;    // (R*C*A)
@group(0) @binding(3) var<storage, read> J: array<f32>;          // (R*A*A)
@group(0) @binding(4) var<storage, read> logNormU: array<f32>;   // (R*C)
@group(0) @binding(5) var<storage, read> logNormD: array<f32>;   // (R*C)
@group(0) @binding(6) var<storage, read> logLike: array<f32>;    // (C)
@group(0) @binding(7) var<storage, read_write> C_out: array<f32>; // (A*A*C)

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

    var acc: f32 = 0.0;

    // Sum over branches n = 1..R-1
    for (var n = 1u; n < params.R; n = n + 1u) {
        let log_s = logNormD[n * params.C + c] + logNormU[n * params.C + c] - logLike[c];
        let scale = exp(log_s);
        let dt_k = D_tilde[(n * params.C + c) * A + k];
        let ut_l = U_tilde[(n * params.C + c) * A + l];
        let j_kl = J[n * A * A + k * A + l];
        acc += dt_k * j_kl * ut_l * scale;
    }

    C_out[k * A * params.C + l * params.C + c] = acc;
}
