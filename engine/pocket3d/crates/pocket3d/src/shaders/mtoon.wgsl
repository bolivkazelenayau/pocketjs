// Native VRMC_materials_mtoon 1.0 opaque/masked surface. Later passes are absent.
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
    // x: shift factor, y: toony, z: GI equalization, w: normal scale
    surface: vec4f,
    // x: MASK, y: authored cutoff, z: double sided
    alpha: vec4f,
    base_uv: UvDesc,
    normal_uv: UvDesc,
    shade_uv: UvDesc,
    shift_uv: UvDesc,
    // x: shading shift texture scale
    texture_scales: vec4f,
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
@group(1) @binding(8) var<uniform> material: MtoonMaterial;
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
@vertex
fn vs_main(in: VsIn) -> VsOut {
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
    var out: VsOut;
    out.clip = globals.view_proj * wp;
    out.uv0 = in.uv;
    out.uv1 = in.uv1;
    out.normal = safe_normalize(
        (instance.normal_model * vec4f(skinned_normal, 0.0)).xyz,
    );
    out.world_pos = wp.xyz;
    return out;
}
fn transformed_uv(in: VsOut, desc: UvDesc) -> vec2f {
    let uv = select(in.uv0, in.uv1, desc.offset_set.z > 0.5);
    let scaled = uv * desc.scale_rotation.xy;
    let c = desc.scale_rotation.z;
    let s = desc.scale_rotation.w;
    return vec2f(c * scaled.x - s * scaled.y, s * scaled.x + c * scaled.y)
        + desc.offset_set.xy;
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
@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4f {
    let base = textureSample(t_base, s_base, transformed_uv(in, material.base_uv))
        * material.base_color_factor;
    if material.alpha.x > 0.5 && base.a < material.alpha.y {
        discard;
    }
    var geometric = safe_normalize(in.normal);
    if material.alpha.z > 0.5 && !front_facing {
        geometric = -geometric;
    }
    let normal_uv = transformed_uv(in, material.normal_uv);
    let n = shading_normal(
        in, geometric, normal_uv, textureSample(t_normal, s_normal, normal_uv).xyz,
    );
    let shift = material.surface.x
        + textureSample(t_shift, s_shift, transformed_uv(in, material.shift_uv)).r
            * material.texture_scales.x;
    let light_dir = safe_normalize(globals.model_sun_dir.xyz);
    let toon = toon_factor(dot(n, light_dir) + shift, material.surface.y);
    let shade = material.shade_color_factor.rgb
        * textureSample(t_shade, s_shade, transformed_uv(in, material.shade_uv)).rgb;
    let direct = mix(shade, base.rgb, toon) * globals.model_sun_color.rgb;
    let uniform_gi = (raw_gi(vec3f(0.0, 1.0, 0.0))
        + raw_gi(vec3f(0.0, -1.0, 0.0))) * 0.5;
    let gi = mix(raw_gi(n), uniform_gi, clamp(material.surface.z, 0.0, 1.0));
    let color = direct + gi * base.rgb;
    return vec4f(color, 1.0);
}
