struct VertexOutput {
    @builtin(position) position: vec4f,
};

@group(0) @binding(0) var logical_output: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.position = vec4f(positions[vertex_index], 0.0, 1.0);
    return output;
}

fn linear_to_srgb(value: vec3f) -> vec3f {
    let low = 12.92 * value;
    let high = 1.055 * pow(value, vec3f(1.0 / 2.4)) - vec3f(0.055);
    return select(high, low, value <= vec3f(0.0031308));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    // The logical target and physical surface have identical dimensions, so
    // read the exact destination texel rather than filtering between texels.
    // Loading the sRGB texture returns decoded linear RGB. Those components
    // are premultiplied by linear alpha at this point.
    let texel = vec2i(trunc(input.position.xy));
    let sampled = textureLoad(logical_output, texel, 0);
    let alpha = clamp(sampled.a, 0.0, 1.0);
    if (alpha <= 0.00001) {
        return vec4f(0.0);
    }

    let straight_linear = clamp(sampled.rgb / alpha, vec3f(0.0), vec3f(1.0));
    let encoded_straight = linear_to_srgb(straight_linear);
    let packed_rgb = clamp(encoded_straight * alpha, vec3f(0.0), vec3f(alpha));
    return vec4f(packed_rgb, alpha);
}
