// Adapted from SMAA / iryoku/smaa (MIT). See ../../../../../../THIRD_PARTY_NOTICES.md.

@group(0) @binding(0) var neighborhood_scene: texture_2d<f32>;
@group(0) @binding(1) var neighborhood_scene_sampler: sampler;
@group(0) @binding(2) var weights_tex: texture_2d<f32>;
@group(0) @binding(3) var weights_sampler: sampler;
@group(0) @binding(4) var<uniform> neighborhood_metrics: Metrics;

fn sanitize_premultiplied(c: vec4f) -> vec4f {
    let alpha = clamp(c.a, 0.0, 1.0);
    let rgb = select(max(c.rgb, vec3f(0.0)), vec3f(0.0), alpha <= 0.00001);
    return vec4f(rgb, alpha);
}

@fragment
fn fs_neighborhood(in: VsOut) -> @location(0) vec4f {
    let t = neighborhood_metrics.texel_size;
    let source = textureSampleLevel(neighborhood_scene, neighborhood_scene_sampler, in.uv, 0.0);
    let current_weights = textureSampleLevel(weights_tex, weights_sampler, in.uv, 0.0);
    // Match the canonical SMAA channel layout: right, top, left, bottom.
    let a = vec4f(
        textureSampleLevel(
            weights_tex,
            weights_sampler,
            in.uv + vec2f(t.x, 0.0),
            0.0,
        )
        .w,
        textureSampleLevel(
            weights_tex,
            weights_sampler,
            in.uv + vec2f(0.0, t.y),
            0.0,
        )
        .y,
        current_weights.z,
        current_weights.x,
    );

    if (dot(a, vec4f(1.0)) < 0.00001) {
        return sanitize_premultiplied(source);
    }

    // Choose the dominant axis and retain the two canonical weights for it.
    let horizontal = max(a.x, a.z) > max(a.y, a.w);
    var blending_offset = vec4f(0.0, a.y, 0.0, a.w);
    var blending_weight = a.yw;
    if (horizontal) {
        blending_offset = vec4f(a.x, 0.0, a.z, 0.0);
        blending_weight = a.xz;
    }
    blending_weight /= dot(blending_weight, vec2f(1.0));

    let blending_coord = in.uv.xyxy + blending_offset * vec4f(t.x, t.y, -t.x, -t.y);
    let color = blending_weight.x * textureSampleLevel(
        neighborhood_scene,
        neighborhood_scene_sampler,
        blending_coord.xy,
        0.0,
    ) + blending_weight.y * textureSampleLevel(
        neighborhood_scene,
        neighborhood_scene_sampler,
        blending_coord.zw,
        0.0,
    );
    return sanitize_premultiplied(color);
}
