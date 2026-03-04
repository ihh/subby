// F81/JC fast path: O(CRA^2) direct computation.
// Dispatch: ceil(A * A * C / 64)
// Each thread computes counts[i,j,c] by summing over branches.

struct Params {
    R: u32,
    C: u32,
    A: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> U: array<f32>;          // (R*C*A)
@group(0) @binding(2) var<storage, read> D: array<f32>;          // (R*C*A)
@group(0) @binding(3) var<storage, read> logNormU: array<f32>;   // (R*C)
@group(0) @binding(4) var<storage, read> logNormD: array<f32>;   // (R*C)
@group(0) @binding(5) var<storage, read> logLike: array<f32>;    // (C)
@group(0) @binding(6) var<storage, read> distances: array<f32>;  // (R)
@group(0) @binding(7) var<storage, read> pi: array<f32>;         // (A)
@group(0) @binding(8) var<uniform> mu_val: f32;                   // mu scalar
@group(0) @binding(9) var<storage, read_write> counts: array<f32>; // (A*A*C)

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

    var acc: f32 = 0.0;

    for (var n = 1u; n < params.R; n = n + 1u) {
        let t = distances[n];
        let mu_t = mu_val * t;
        let e_t = exp(-mu_t);
        let p = 1.0 - e_t;

        let alpha_n = t * e_t;
        let beta_n = p / mu_val - t * e_t;
        let gamma_n = t * (1.0 + e_t) - 2.0 * p / mu_val;

        let log_s = logNormD[n * params.C + c] + logNormU[n * params.C + c] - logLike[c];
        let scale = exp(log_s);

        let U_base = (n * params.C + c) * A;
        let D_base = (n * params.C + c) * A;

        // piU = sum_b pi[b] * U[n,c,b]
        var piU: f32 = 0.0;
        for (var b = 0u; b < A; b = b + 1u) {
            piU += pi[b] * U[U_base + b];
        }

        // Dsum = sum_a D[n,c,a] * scale
        var Dsum: f32 = 0.0;
        for (var a = 0u; a < A; a = a + 1u) {
            Dsum += D[D_base + a] * scale;
        }

        let D_i_scaled = D[D_base + i] * scale;
        let U_j = U[U_base + j];

        let I_sum = alpha_n * D_i_scaled * U_j
                  + beta_n * (D_i_scaled * piU + pi[i] * Dsum * U_j)
                  + gamma_n * pi[i] * Dsum * piU;

        if (i == j) {
            acc += I_sum;
        } else {
            acc += mu_val * pi[j] * I_sum;
        }
    }

    counts[i * A * params.C + j * params.C + c] = acc;
}
