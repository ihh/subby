// Compute mixture posterior: softmax over K components per column.
// Dispatch: ceil(C / 64)

struct Params {
    K: u32,
    C: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logLikes: array<f32>;      // (K*C)
@group(0) @binding(2) var<storage, read> logWeights: array<f32>;    // (K)
@group(0) @binding(3) var<storage, read_write> posteriors: array<f32>; // (K*C)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let c = gid.x;
    if (c >= params.C) { return; }

    let K = params.K;

    // Find max for numerical stability
    var max_val: f32 = -1e30;
    for (var k = 0u; k < K; k = k + 1u) {
        let lj = logLikes[k * params.C + c] + logWeights[k];
        max_val = max(max_val, lj);
    }

    // Compute exp and sum
    var denom: f32 = 0.0;
    for (var k = 0u; k < K; k = k + 1u) {
        let val = exp(logLikes[k * params.C + c] + logWeights[k] - max_val);
        posteriors[k * params.C + c] = val;
        denom += val;
    }

    // Normalize
    for (var k = 0u; k < K; k = k + 1u) {
        posteriors[k * params.C + c] /= denom;
    }
}
