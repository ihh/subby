// Transform eigenbasis counts to natural basis.
// VCV_{ij,c} = sum_{kl} V_{ik} C_{kl,c} V_{jl}
// counts[i,j,c] = VCV[i,j,c] if i==j (dwell), S[i,j]*VCV[i,j,c] if i!=j (subs)
// Dispatch: ceil(A * A * C / 64)

struct Params {
    C: u32,
    A: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> eigenvectors: array<f32>;  // (A*A)
@group(0) @binding(2) var<storage, read> eigenvalues: array<f32>;   // (A)
@group(0) @binding(3) var<storage, read> C_in: array<f32>;          // (A*A*C)
@group(0) @binding(4) var<storage, read_write> counts: array<f32>;  // (A*A*C)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let A = params.A;
    let total = A * A * params.C;
    if (idx >= total) { return; }

    let i = idx / (A * params.C);
    let rem = idx % (A * params.C);
    let j = rem / params.C;
    let c = rem % params.C;

    // VCV_{ij,c} = sum_{kl} V[i,k] * C[k,l,c] * V[j,l]
    var vcv: f32 = 0.0;
    for (var k = 0u; k < A; k = k + 1u) {
        let v_ik = eigenvectors[i * A + k];
        for (var l = 0u; l < A; l = l + 1u) {
            vcv += v_ik * C_in[k * A * params.C + l * params.C + c] * eigenvectors[j * A + l];
        }
    }

    if (i == j) {
        counts[i * A * params.C + j * params.C + c] = vcv;
    } else {
        // S_{ij} = sum_k V[i,k] * mu[k] * V[j,k]
        var s_ij: f32 = 0.0;
        for (var k = 0u; k < A; k = k + 1u) {
            s_ij += eigenvectors[i * A + k] * eigenvalues[k] * eigenvectors[j * A + k];
        }
        counts[i * A * params.C + j * params.C + c] = s_ij * vcv;
    }
}
