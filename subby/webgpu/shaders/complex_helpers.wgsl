// Complex arithmetic helpers for irreversible model shaders.
// Complex numbers are represented as vec2<f32> where .x = real, .y = imag.

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cexp(a: vec2<f32>) -> vec2<f32> {
    let r = exp(a.x);
    return vec2(r * cos(a.y), r * sin(a.y));
}

fn cadd(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

fn csub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a - b;
}

fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let denom = b.x * b.x + b.y * b.y;
    return vec2((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}

fn cabs(a: vec2<f32>) -> f32 {
    return sqrt(a.x * a.x + a.y * a.y);
}

fn cscale(a: vec2<f32>, s: f32) -> vec2<f32> {
    return vec2(a.x * s, a.y * s);
}

// Load complex from interleaved f32 array at index i: (buf[2*i], buf[2*i+1])
fn cload(buf: ptr<storage, array<f32>, read>, i: u32) -> vec2<f32> {
    return vec2((*buf)[2u * i], (*buf)[2u * i + 1u]);
}
