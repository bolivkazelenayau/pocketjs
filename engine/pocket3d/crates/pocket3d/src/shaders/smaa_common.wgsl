struct Metrics {
    texel_size: vec2f,
    alpha_edge: f32,
    _pad: f32,
}

struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let x = f32(i32(i % 2u) * 4 - 1);
    let y = f32(i32(i / 2u) * 4 - 1);
    var out: VsOut;
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}
