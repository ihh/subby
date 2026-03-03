// One step of the downward (outside) pass.
// Called R-1 times (once per non-root node in preorder).
// Each dispatch: ceil(C / 64)

struct Params {
    C: u32,
    A: u32,
    node: u32,
    parent: u32,
    sibling: u32,
    parent_is_root: u32,  // 1 if parent == 0, else 0
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sibMatrix: array<f32>;     // (A*A) sibling's branch matrix
@group(0) @binding(2) var<storage, read> parentMatrix: array<f32>;  // (A*A) parent's branch matrix
@group(0) @binding(3) var<storage, read> U: array<f32>;             // (R*C*A) inside vectors
@group(0) @binding(4) var<storage, read> logNormU: array<f32>;      // (R*C)
@group(0) @binding(5) var<storage, read_write> D: array<f32>;       // (R*C*A) outside vectors
@group(0) @binding(6) var<storage, read_write> logNormD: array<f32>; // (R*C)
@group(0) @binding(7) var<storage, read> rootProb: array<f32>;      // (A)
@group(0) @binding(8) var<storage, read> obsLike: array<f32>;       // (R*C*A)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let c = gid.x;
    if (c >= params.C) { return; }

    let A = params.A;
    let node = params.node;
    let parent = params.parent;
    let sib = params.sibling;

    let sib_U_base = sib * params.C * A + c * A;
    let parent_D_base = parent * params.C * A + c * A;
    let parent_obs_base = parent * params.C * A + c * A;

    // Sibling contribution: sum_j sibMatrix[a,j] * U[sib,c,j]
    // Parent contribution: rootProb[a] if parent_is_root, else sum_i D[parent,c,i] * parentMatrix[i,a]
    // D_raw[a] = sib_contrib[a] * parent_contrib[a] * obs_like[parent,c,a]

    var max_val: f32 = 0.0;

    // First compute D_raw in temporary storage (reuse D output slot)
    let node_D_base = node * params.C * A + c * A;

    for (var a = 0u; a < A; a = a + 1u) {
        // Sibling contribution
        var sib_contrib: f32 = 0.0;
        for (var j = 0u; j < A; j = j + 1u) {
            sib_contrib += sibMatrix[a * A + j] * U[sib_U_base + j];
        }

        // Parent contribution
        var parent_contrib: f32;
        if (params.parent_is_root == 1u) {
            parent_contrib = rootProb[a];
        } else {
            parent_contrib = 0.0;
            for (var i = 0u; i < A; i = i + 1u) {
                parent_contrib += D[parent_D_base + i] * parentMatrix[i * A + a];
            }
        }

        // Include parent observation likelihood
        parent_contrib *= obsLike[parent_obs_base + a];

        let d_raw = sib_contrib * parent_contrib;
        D[node_D_base + a] = d_raw;
        max_val = max(max_val, d_raw);
    }

    // Rescale
    max_val = max(max_val, 1e-30);
    for (var a = 0u; a < A; a = a + 1u) {
        D[node_D_base + a] /= max_val;
    }

    // logNormD
    let log_norm_sib = logNormU[sib * params.C + c];
    var log_norm_prior: f32;
    if (params.parent_is_root == 1u) {
        log_norm_prior = 0.0;
    } else {
        log_norm_prior = logNormD[parent * params.C + c];
    }
    logNormD[node * params.C + c] = log_norm_sib + log_norm_prior + log(max_val);
}
