// One step of the upward (Felsenstein pruning) pass.
// Called R-1 times (once per child in postorder).
// Each dispatch: ceil(C / 64) — each thread handles one column.

struct Params {
    C: u32,
    A: u32,
    child: u32,    // index of child node
    parent: u32,   // index of parent node
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> subMatrix: array<f32>;        // (A*A) for this child's branch
@group(0) @binding(2) var<storage, read_write> likelihood: array<f32>; // (R*C*A)
@group(0) @binding(3) var<storage, read_write> logNormU: array<f32>;   // (R*C)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let c = gid.x;
    if (c >= params.C) { return; }

    let A = params.A;
    let child = params.child;
    let parent = params.parent;

    // child_contrib[b] = sum_j M[b,j] * likelihood[child,c,j]
    // Then parent[c,b] *= child_contrib[b]
    let child_base = child * params.C * A + c * A;
    let parent_base = parent * params.C * A + c * A;

    // Compute child contribution and multiply into parent
    for (var b = 0u; b < A; b = b + 1u) {
        var s: f32 = 0.0;
        for (var j = 0u; j < A; j = j + 1u) {
            s += subMatrix[b * A + j] * likelihood[child_base + j];
        }
        likelihood[parent_base + b] *= s;
    }

    // Rescale parent: divide by max
    var max_val: f32 = likelihood[parent_base];
    for (var b = 1u; b < A; b = b + 1u) {
        max_val = max(max_val, likelihood[parent_base + b]);
    }
    max_val = max(max_val, 1e-30);

    for (var b = 0u; b < A; b = b + 1u) {
        likelihood[parent_base + b] /= max_val;
    }

    // Update logNormU: parent += child + log(max_val)
    let child_ln = logNormU[child * params.C + c];
    logNormU[parent * params.C + c] += child_ln + log(max_val);
}
