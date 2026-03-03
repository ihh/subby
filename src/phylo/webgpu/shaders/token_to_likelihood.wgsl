// Convert integer token alignment to likelihood vectors.
// Dispatch: ceil(R * C / 64)
// Token encoding: 0..A-1 = one-hot, else = all ones

struct Params {
    R: u32,
    C: u32,
    A: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> alignment: array<i32>;       // (R*C)
@group(0) @binding(2) var<storage, read_write> likelihood: array<f32>; // (R*C*A)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = params.R * params.C;
    if (idx >= total) { return; }

    let r = idx / params.C;
    let c = idx % params.C;
    let tok = alignment[r * params.C + c];
    let base = (r * params.C + c) * params.A;

    for (var a = 0u; a < params.A; a = a + 1u) {
        if (tok >= 0 && u32(tok) < params.A) {
            if (a == u32(tok)) {
                likelihood[base + a] = 1.0;
            } else {
                likelihood[base + a] = 0.0;
            }
        } else {
            likelihood[base + a] = 1.0;
        }
    }
}
