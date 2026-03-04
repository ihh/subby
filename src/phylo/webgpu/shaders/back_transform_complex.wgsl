// Transform eigenbasis counts to natural basis for irreversible model.
// VCV_{ij,c} = sum_{kl} V_{ik} C_{kl,c} V_inv_{lj}
// R_{ij} = sum_k V_{ik} mu_k V_inv_{kj}
// counts[i,j,c] = Re(VCV[i,j,c]) if i==j, Re(R[i,j]*VCV[i,j,c]) if i!=j
// Dispatch: ceil(A * A * C / 64)

struct Params {
    C: u32,
    A: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> eigenvectors: array<f32>;      // (2*A*A) interleaved
@group(0) @binding(2) var<storage, read> eigenvectors_inv: array<f32>;  // (2*A*A) interleaved
@group(0) @binding(3) var<storage, read> eigenvalues: array<f32>;       // (2*A) interleaved
@group(0) @binding(4) var<storage, read> C_in: array<f32>;              // (2*A*A*C) interleaved
@group(0) @binding(5) var<storage, read_write> counts: array<f32>;      // (A*A*C) real output

fn cmul_local(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

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

    // VCV_{ij,c} = sum_{kl} V_{ik} C_{kl,c} V_inv_{lj}
    var vcv = vec2<f32>(0.0, 0.0);
    for (var k = 0u; k < A; k = k + 1u) {
        let v_ik = vec2(eigenvectors[2u * (i * A + k)], eigenvectors[2u * (i * A + k) + 1u]);
        for (var l = 0u; l < A; l = l + 1u) {
            let c_idx = k * A * params.C + l * params.C + c;
            let c_kl = vec2(C_in[2u * c_idx], C_in[2u * c_idx + 1u]);
            let vinv_lj = vec2(eigenvectors_inv[2u * (l * A + j)], eigenvectors_inv[2u * (l * A + j) + 1u]);
            vcv += cmul_local(cmul_local(v_ik, c_kl), vinv_lj);
        }
    }

    if (i == j) {
        counts[i * A * params.C + j * params.C + c] = vcv.x; // Re
    } else {
        // R_{ij} = sum_k V_{ik} mu_k V_inv_{kj}
        var r_ij = vec2<f32>(0.0, 0.0);
        for (var k = 0u; k < A; k = k + 1u) {
            let v_ik = vec2(eigenvectors[2u * (i * A + k)], eigenvectors[2u * (i * A + k) + 1u]);
            let mu_k = vec2(eigenvalues[2u * k], eigenvalues[2u * k + 1u]);
            let vinv_kj = vec2(eigenvectors_inv[2u * (k * A + j)], eigenvectors_inv[2u * (k * A + j) + 1u]);
            r_ij += cmul_local(cmul_local(v_ik, mu_k), vinv_kj);
        }
        counts[i * A * params.C + j * params.C + c] = cmul_local(r_ij, vcv).x; // Re
    }
}
