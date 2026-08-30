// Adapted from SMAA / iryoku/smaa (MIT). See ../../../../../../THIRD_PARTY_NOTICES.md.

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_sampler: sampler;
@group(0) @binding(2) var<uniform> metrics: Metrics;

fn scene_signal(uv: vec2f) -> f32 {
    let c = textureSampleLevel(scene_tex, scene_sampler, uv, 0.0);
    let luma = dot(c.rgb, vec3f(0.2126, 0.7152, 0.0722));
    // The edge bind group deliberately uses the non-sRGB view of the sRGB
    // scene texture. Canonical SMAA luma detection expects gamma-encoded
    // values; alpha adds a silhouette signal without unpremultiplying RGB.
    return luma + metrics.alpha_edge * c.a * 0.25;
}

@fragment
fn fs_edge(in: VsOut) -> @location(0) vec4f {
    let t = metrics.texel_size;
    let center = scene_signal(in.uv);
    let left = scene_signal(in.uv - vec2f(t.x, 0.0));
    let top = scene_signal(in.uv - vec2f(0.0, t.y));
    let right = scene_signal(in.uv + vec2f(t.x, 0.0));
    let bottom = scene_signal(in.uv + vec2f(0.0, t.y));
    var delta = vec4f(
        abs(center - left),
        abs(center - top),
        abs(center - right),
        abs(center - bottom),
    );
    var edges = step(vec2f(0.1), delta.xy);
    if (dot(edges, vec2f(1.0)) == 0.0) {
        return vec4f(0.0);
    }

    let max_direct_delta = max(delta.xy, delta.zw);
    let left_left = scene_signal(in.uv - vec2f(2.0 * t.x, 0.0));
    let top_top = scene_signal(in.uv - vec2f(0.0, 2.0 * t.y));
    let left_left_delta = abs(left - left_left);
    let top_top_delta = abs(top - top_top);
    delta.z = left_left_delta;
    delta.w = top_top_delta;
    let max_delta = max(max_direct_delta, delta.zw);
    let final_delta = max(max_delta.x, max_delta.y);

    // Canonical SMAA local-contrast adaptation: discard an edge when a
    // neighboring contrast is more than twice the current contrast.
    edges *= step(vec2f(final_delta), 2.0 * delta.xy);
    return vec4f(edges, 0.0, 0.0);
}
