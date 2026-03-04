// Compute J^{kl}(T) interaction matrix for complex eigenvalues.
// Output is complex (interleaved), stored as 2*R*A*A f32.
// Dispatch: ceil(R / 64)

struct Params {
    R: u32,
    A: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> eigenvalues: array<f32>;   // (2*A) interleaved
@group(0) @binding(2) var<storage, read> distances: array<f32>;      // (R)
@group(0) @binding(3) var<storage, read_write> J: array<f32>;       // (2*R*A*A) interleaved

fn cmul_local(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cexp_local(a: vec2<f32>) -> vec2<f32> {
    let r = exp(a.x);
    return vec2(r * cos(a.y), r * sin(a.y));
}

fn cdiv_local(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let denom = b.x * b.x + b.y * b.y;
    return vec2((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}

fn cabs_local(a: vec2<f32>) -> f32 {
    return sqrt(a.x * a.x + a.y * a.y);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let r = gid.x;
    if (r >= params.R) { return; }

    let A = params.A;
    let t = distances[r];
    let base = r * A * A;

    for (var k = 0u; k < A; k = k + 1u) {
        let mu_k = vec2(eigenvalues[2u * k], eigenvalues[2u * k + 1u]);
        let exp_k = cexp_local(vec2(mu_k.x * t, mu_k.y * t));
        for (var l = 0u; l < A; l = l + 1u) {
            let mu_l = vec2(eigenvalues[2u * l], eigenvalues[2u * l + 1u]);
            let diff = mu_k - mu_l;
            let idx = base + k * A + l;
            if (cabs_local(diff) < 1e-6) {
                // Degenerate: J = t * exp(mu_k * t)
                J[2u * idx] = t * exp_k.x;
                J[2u * idx + 1u] = t * exp_k.y;
            } else {
                let exp_l = cexp_local(vec2(mu_l.x * t, mu_l.y * t));
                let result = cdiv_local(exp_k - exp_l, diff);
                J[2u * idx] = result.x;
                J[2u * idx + 1u] = result.y;
            }
        }
    }
}
