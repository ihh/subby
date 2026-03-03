// Compute J^{kl}(T) interaction matrix between decay modes.
// Dispatch: ceil(R / 64) — each thread computes A×A J matrix for one branch

struct Params {
    R: u32,
    A: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> eigenvalues: array<f32>;  // (A)
@group(0) @binding(2) var<storage, read> distances: array<f32>;    // (R)
@group(0) @binding(3) var<storage, read_write> J: array<f32>;      // (R*A*A)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let r = gid.x;
    if (r >= params.R) { return; }

    let A = params.A;
    let t = distances[r];
    let base = r * A * A;

    for (var k = 0u; k < A; k = k + 1u) {
        let mu_k = eigenvalues[k];
        let exp_k = exp(mu_k * t);
        for (var l = 0u; l < A; l = l + 1u) {
            let mu_l = eigenvalues[l];
            let diff = mu_k - mu_l;
            if (abs(diff) < 1e-6) {
                // Degenerate case
                J[base + k * A + l] = t * exp_k;
            } else {
                J[base + k * A + l] = (exp_k - exp(mu_l * t)) / diff;
            }
        }
    }
}
