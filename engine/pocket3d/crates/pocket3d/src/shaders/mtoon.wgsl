// Native VRMC_materials_mtoon 1.0 Stage B-F surface and outline.
struct Globals {
    view_proj: mat4x4f,
    inverse_view_proj: mat4x4f,
    cam_pos: vec4f,
    sky_zenith: vec4f,
    sky_horizon: vec4f,
    sky_sun_dir: vec4f,
    sky_sun_color: vec4f,
    model_sun_dir: vec4f,
    model_sun_color: vec4f,
    model_ambient: vec4f,
    toon: vec4f,
    rim_color: vec4f,
    rim_params: vec4f,
    fog_color: vec4f,
    fog_params: vec4f,
    // xy: viewport dimensions, zw: reciprocal dimensions
    viewport: vec4f,
}
struct Instance {
    model: mat4x4f,
    normal_model: mat4x4f,
    tint: vec4f,
    params: vec4f,
}
struct UvDesc {
    // xy: scale, zw: cos(rotation), sin(rotation)
    scale_rotation: vec4f,
    // xy: offset, z: effective TEXCOORD set
    offset_set: vec4f,
}
struct MtoonMaterial {
    base_color_factor: vec4f,
    shade_color_factor: vec4f,
    emissive_factor: vec4f,
    matcap_factor: vec4f,
    rim_color_factor: vec4f,
    // x: Fresnel power, y: lift, z: lighting mix
    rim_params: vec4f,
    // x: shift factor, y: toony, z: GI equalization, w: normal scale
    surface: vec4f,
    // x: MASK, y: authored cutoff, z: double sided, w: authored BLEND
    alpha: vec4f,
    outline_color_factor: vec4f,
    // x: width mode (0 none, 1 world, 2 screen), y: width factor,
    // z: lighting mix
    outline: vec4f,
    base_uv: UvDesc,
    normal_uv: UvDesc,
    shade_uv: UvDesc,
    shift_uv: UvDesc,
    emissive_uv: UvDesc,
    rim_uv: UvDesc,
    outline_uv: UvDesc,
    uv_animation_mask_uv: UvDesc,
    // x: shading shift texture scale
    texture_scales: vec4f,
    // xy: scroll UV/second, z: rotation radians/second
    uv_animation: vec4f,
}
@group(0) @binding(0) var<uniform> globals: Globals;
@group(1) @binding(0) var t_base: texture_2d<f32>;
@group(1) @binding(1) var s_base: sampler;
@group(1) @binding(2) var t_normal: texture_2d<f32>;
@group(1) @binding(3) var s_normal: sampler;
@group(1) @binding(4) var t_shade: texture_2d<f32>;
@group(1) @binding(5) var s_shade: sampler;
@group(1) @binding(6) var t_shift: texture_2d<f32>;
@group(1) @binding(7) var s_shift: sampler;
@group(1) @binding(8) var t_emissive: texture_2d<f32>;
@group(1) @binding(9) var s_emissive: sampler;
@group(1) @binding(10) var t_matcap: texture_2d<f32>;
@group(1) @binding(11) var s_matcap: sampler;
@group(1) @binding(12) var t_rim: texture_2d<f32>;
@group(1) @binding(13) var s_rim: sampler;
@group(1) @binding(14) var<uniform> material: MtoonMaterial;
@group(1) @binding(15) var t_outline_width: texture_2d<f32>;
@group(1) @binding(16) var s_outline_width: sampler;
@group(1) @binding(17) var t_uv_animation_mask: texture_2d<f32>;
@group(1) @binding(18) var s_uv_animation_mask: sampler;
@group(2) @binding(0) var<uniform> instance: Instance;
@group(2) @binding(1) var<storage, read> joints: array<mat4x4f>;

struct VsIn {
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) joints: vec4u,
    @location(4) weights: vec4f,
    @location(5) uv1: vec2f,
}
struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) uv0: vec2f,
    @location(1) normal: vec3f,
    @location(2) world_pos: vec3f,
    @location(3) uv1: vec2f,
}
fn safe_normalize(v: vec3f) -> vec3f {
    let length_squared = dot(v, v);
    if length_squared > 1e-10 {
        return v * inverseSqrt(length_squared);
    }
    return vec3f(0.0, 0.0, 1.0);
}
struct DeformedVertex {
    world_pos: vec3f,
    world_normal: vec3f,
}
fn deform_vertex(in: VsIn) -> DeformedVertex {
    var skin = mat4x4f(
        vec4f(1.0, 0.0, 0.0, 0.0),
        vec4f(0.0, 1.0, 0.0, 0.0),
        vec4f(0.0, 0.0, 1.0, 0.0),
        vec4f(0.0, 0.0, 0.0, 1.0),
    );
    let wsum = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
    if wsum > 0.001 {
        skin = in.weights.x * joints[in.joints.x]
            + in.weights.y * joints[in.joints.y]
            + in.weights.z * joints[in.joints.z]
            + in.weights.w * joints[in.joints.w];
    }
    let skinned_pos = skin * vec4f(in.pos, 1.0);
    let skinned_normal = (skin * vec4f(in.normal, 0.0)).xyz;
    let wp = instance.model * skinned_pos;
    var out: DeformedVertex;
    out.world_pos = wp.xyz;
    out.world_normal = safe_normalize(
        (instance.normal_model * vec4f(skinned_normal, 0.0)).xyz,
    );
    return out;
}
fn vertex_varyings(in: VsIn, deformed: DeformedVertex) -> VsOut {
    var out: VsOut;
    out.clip = globals.view_proj * vec4f(deformed.world_pos, 1.0);
    out.uv0 = in.uv;
    out.uv1 = in.uv1;
    out.normal = deformed.world_normal;
    out.world_pos = deformed.world_pos;
    return out;
}
fn selected_uv(in: VsOut, desc: UvDesc) -> vec2f {
    return select(in.uv0, in.uv1, desc.offset_set.z > 0.5);
}
fn apply_khr_texture_transform(uv: vec2f, desc: UvDesc) -> vec2f {
    let scaled = uv * desc.scale_rotation.xy;
    let c = desc.scale_rotation.z;
    let s = desc.scale_rotation.w;
    return vec2f(c * scaled.x - s * scaled.y, s * scaled.x + c * scaled.y)
        + desc.offset_set.xy;
}
fn static_transformed_uv(in: VsOut, desc: UvDesc) -> vec2f {
    return apply_khr_texture_transform(selected_uv(in, desc), desc);
}
fn animated_uv(uv: vec2f, mask: f32) -> vec2f {
    let time = globals.cam_pos.w;
    let angle = material.uv_animation.z * time * mask;
    let c = cos(angle);
    let s = sin(angle);
    let centered = uv - vec2f(0.5);
    let rotated = vec2f(
        c * centered.x - s * centered.y,
        s * centered.x + c * centered.y,
    ) + vec2f(0.5);
    return rotated + material.uv_animation.xy * time * mask;
}
fn animated_transformed_uv(in: VsOut, desc: UvDesc, mask: f32) -> vec2f {
    // MToon animation is applied to the target's selected source UV before
    // that target's independent authored KHR_texture_transform.
    return apply_khr_texture_transform(animated_uv(selected_uv(in, desc), mask), desc);
}
fn fragment_uv_animation_mask(in: VsOut) -> f32 {
    // The mask has its own static TextureInfo and is never animated itself.
    return textureSample(
        t_uv_animation_mask,
        s_uv_animation_mask,
        static_transformed_uv(in, material.uv_animation_mask_uv),
    ).b;
}
@vertex
fn vs_main(in: VsIn) -> VsOut {
    return vertex_varyings(in, deform_vertex(in));
}
@vertex
fn vs_outline(in: VsIn) -> VsOut {
    let deformed = deform_vertex(in);
    var out = vertex_varyings(in, deformed);
    let uv_animation_mask = textureSampleLevel(
        t_uv_animation_mask,
        s_uv_animation_mask,
        static_transformed_uv(out, material.uv_animation_mask_uv),
        0.0,
    ).b;
    let width_mask = textureSampleLevel(
        t_outline_width,
        s_outline_width,
        animated_transformed_uv(out, material.outline_uv, uv_animation_mask),
        0.0,
    ).g;
    let width = material.outline.y * width_mask;
    if material.outline.x < 1.5 {
        // World-coordinate width is a normalized world-normal displacement,
        // after morphing, skinning, and the complete instance transform.
        let extruded = deformed.world_pos + deformed.world_normal * width;
        out.clip = globals.view_proj * vec4f(extruded, 1.0);
    } else {
        // Project the world-normal differential, normalize it in units of
        // viewport height, then convert the requested height ratio back to NDC.
        // This derivative form retains shifted/off-axis projection terms.
        let original_clip = out.clip;
        let normal_clip = globals.view_proj * vec4f(deformed.world_normal, 0.0);
        let w_squared = max(original_clip.w * original_clip.w, 1e-12);
        let ndc_derivative = (
            normal_clip.xy * original_clip.w - original_clip.xy * normal_clip.w
        ) / w_squared;
        let aspect = globals.viewport.x / max(globals.viewport.y, 1.0);
        let height_direction = vec2f(ndc_derivative.x * aspect, ndc_derivative.y);
        let direction_length_squared = dot(height_direction, height_direction);
        if direction_length_squared > 1e-12 {
            let direction = height_direction * inverseSqrt(direction_length_squared);
            let ndc_offset = 2.0 * width * vec2f(direction.x / aspect, direction.y);
            out.clip = vec4f(out.clip.xy + ndc_offset * original_clip.w, out.clip.zw);
        }
    }
    // Match the mature MToon implementations' minimal normalized-depth bias:
    // move the hull away by one millionth of clip.w to suppress coplanar leaks.
    out.clip = vec4f(out.clip.xy, out.clip.z + 1e-6 * out.clip.w, out.clip.w);
    return out;
}
fn shading_normal(in: VsOut, geometric: vec3f, uv: vec2f, normal_sample: vec3f) -> vec3f {
    // Derivatives are evaluated before the determinant branch (uniformity).
    let position_dx = dpdx(in.world_pos);
    let position_dy = dpdy(in.world_pos);
    let duvdx = dpdx(uv);
    let duvdy = dpdy(uv);
    let det = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
    if abs(det) < 1e-8 {
        return geometric;
    }
    let tangent_raw = (position_dx * duvdy.y - position_dy * duvdx.y) / det;
    let tangent_ortho = tangent_raw - geometric * dot(geometric, tangent_raw);
    if dot(tangent_ortho, tangent_ortho) < 1e-10 {
        return geometric;
    }
    let tangent = safe_normalize(tangent_ortho);
    let bitangent_raw = (position_dy * duvdx.x - position_dx * duvdy.x) / det;
    let handedness = select(-1.0, 1.0, dot(cross(geometric, tangent), bitangent_raw) >= 0.0);
    let bitangent = cross(geometric, tangent) * handedness;
    let map = normal_sample * 2.0 - vec3f(1.0);
    let scaled = safe_normalize(vec3f(map.xy * material.surface.w, map.z));
    return safe_normalize(tangent * scaled.x + bitangent * scaled.y + geometric * scaled.z);
}
fn raw_gi(n: vec3f) -> vec3f {
    return mix(
        globals.model_ambient.rgb * 0.72,
        globals.model_ambient.rgb * 0.58 + globals.sky_zenith.rgb * 0.56,
        clamp(n.y * 0.5 + 0.5, 0.0, 1.0),
    );
}
fn toon_factor(shading: f32, toony: f32) -> f32 {
    let clamped = clamp(toony, 0.0, 1.0);
    if clamped >= 1.0 {
        return select(0.0, 1.0, shading >= 0.0);
    }
    let edge0 = -1.0 + clamped;
    let edge1 = 1.0 - clamped;
    return clamp((shading - edge0) / (edge1 - edge0), 0.0, 1.0);
}
fn matcap_uv(n: vec3f, view: vec3f) -> vec2f {
    // The specified world-view basis is undefined for a vertical view vector.
    // Use a perpendicular world X axis in that degenerate case.
    let horizontal = vec3f(view.z, 0.0, -view.x);
    var view_x = vec3f(1.0, 0.0, 0.0);
    if dot(horizontal, horizontal) > 1e-10 {
        view_x = normalize(horizontal);
    }
    let view_y = cross(view, view_x);
    return vec2f(dot(view_x, n), dot(view_y, n)) * 0.495 + vec2f(0.5);
}
fn sample_base(in: VsOut, uv_animation_mask: f32) -> vec4f {
    return textureSample(
        t_base,
        s_base,
        animated_transformed_uv(in, material.base_uv, uv_animation_mask),
    )
        * material.base_color_factor;
}
fn surface_color(
    in: VsOut,
    geometric: vec3f,
    base: vec4f,
    uv_animation_mask: f32,
) -> vec3f {
    let normal_uv = animated_transformed_uv(in, material.normal_uv, uv_animation_mask);
    let n = shading_normal(
        in, geometric, normal_uv, textureSample(t_normal, s_normal, normal_uv).xyz,
    );
    let shift = material.surface.x
        + textureSample(
            t_shift,
            s_shift,
            animated_transformed_uv(in, material.shift_uv, uv_animation_mask),
        ).r
            * material.texture_scales.x;
    let light_dir = safe_normalize(globals.model_sun_dir.xyz);
    let toon = toon_factor(dot(n, light_dir) + shift, material.surface.y);
    let shade = material.shade_color_factor.rgb
        * textureSample(
            t_shade,
            s_shade,
            animated_transformed_uv(in, material.shade_uv, uv_animation_mask),
        ).rgb;
    let direct = mix(shade, base.rgb, toon) * globals.model_sun_color.rgb;
    let uniform_gi = (raw_gi(vec3f(0.0, 1.0, 0.0))
        + raw_gi(vec3f(0.0, -1.0, 0.0))) * 0.5;
    let gi = mix(raw_gi(n), uniform_gi, clamp(material.surface.z, 0.0, 1.0));
    // V is surface-to-camera in world space, consistent with the world normal.
    let view = safe_normalize(globals.cam_pos.xyz - in.world_pos);
    let emission = material.emissive_factor.rgb
        * textureSample(
            t_emissive,
            s_emissive,
            animated_transformed_uv(in, material.emissive_uv, uv_animation_mask),
        ).rgb;
    let matcap = material.matcap_factor.rgb
        * textureSample(t_matcap, s_matcap, matcap_uv(n, view)).rgb;
    let rim_shape = pow(
        clamp(1.0 - dot(n, view) + material.rim_params.y, 0.0, 1.0),
        max(material.rim_params.x, 0.00001),
    );
    var rim = matcap + rim_shape * material.rim_color_factor.rgb;
    rim *= textureSample(
        t_rim,
        s_rim,
        animated_transformed_uv(in, material.rim_uv, uv_animation_mask),
    ).rgb;
    // Pocket3D has one unattenuated direct sun and equalized hemisphere GI.
    // Their RGB lighting sum influences rim independently of base/shade/toon.
    let lighting = globals.model_sun_color.rgb + gi;
    rim *= mix(vec3f(1.0), lighting, material.rim_params.z);
    return direct + gi * base.rgb + emission + rim;
}
fn output_alpha(base: vec4f) -> f32 {
    let material_alpha = select(1.0, base.a, material.alpha.w > 0.5);
    return material_alpha * instance.tint.a;
}
@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4f {
    let uv_animation_mask = fragment_uv_animation_mask(in);
    let base = sample_base(in, uv_animation_mask);
    if material.alpha.x > 0.5 && base.a < material.alpha.y {
        discard;
    }
    var geometric = safe_normalize(in.normal);
    if material.alpha.z > 0.5 && !front_facing {
        geometric = -geometric;
    }
    let color = surface_color(in, geometric, base, uv_animation_mask) * instance.tint.rgb;
    return vec4f(color, output_alpha(base));
}
@fragment
fn fs_outline(in: VsOut) -> @location(0) vec4f {
    let uv_animation_mask = fragment_uv_animation_mask(in);
    let base = sample_base(in, uv_animation_mask);
    if material.alpha.x > 0.5 && base.a < material.alpha.y {
        discard;
    }
    // The lit endpoint is the same complete MToon surface result used above:
    // direct lighting + GI + emission + rim/MatCap. Normal maps may influence
    // its color, but never the geometric extrusion performed in vs_outline.
    let lit = surface_color(in, safe_normalize(in.normal), base, uv_animation_mask);
    let lighting_mix = mix(
        vec3f(1.0),
        lit,
        clamp(material.outline.z, 0.0, 1.0),
    );
    let color = material.outline_color_factor.rgb * lighting_mix * instance.tint.rgb;
    return vec4f(color, output_alpha(base));
}
