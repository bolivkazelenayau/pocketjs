//! Complete typed `VRMC_materials_mtoon` 1.0 semantics.
//!
//! This module owns VRM JSON parsing, normative defaults, validation, and the
//! one-way conversion to Pocket3D's render-facing descriptor. Pocket3D does
//! not inspect VRM extension JSON.

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};

use pocket3d::material as render;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Vrm1MtoonAlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Vrm1MtoonOutlineWidthMode {
    None,
    WorldCoordinates,
    ScreenCoordinates,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Vrm1MagFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Vrm1MinFilter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Vrm1WrapMode {
    ClampToEdge,
    MirroredRepeat,
    Repeat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Vrm1Sampler {
    pub index: Option<usize>,
    pub mag_filter: Option<Vrm1MagFilter>,
    pub min_filter: Option<Vrm1MinFilter>,
    pub wrap_s: Vrm1WrapMode,
    pub wrap_t: Vrm1WrapMode,
}

impl Default for Vrm1Sampler {
    fn default() -> Self {
        Self {
            index: None,
            mag_filter: None,
            min_filter: None,
            wrap_s: Vrm1WrapMode::Repeat,
            wrap_t: Vrm1WrapMode::Repeat,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vrm1TextureTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub tex_coord_override: Option<u32>,
}

impl Default for Vrm1TextureTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            rotation: 0.0,
            scale: [1.0, 1.0],
            tex_coord_override: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1TextureInfo {
    pub texture_index: usize,
    pub image_index: usize,
    pub sampler: Vrm1Sampler,
    pub tex_coord: u32,
    pub transform: Vrm1TextureTransform,
}

impl Vrm1TextureInfo {
    pub fn effective_tex_coord(&self) -> u32 {
        self.transform.tex_coord_override.unwrap_or(self.tex_coord)
    }

    fn to_render(&self, role: render::TextureRole) -> render::TextureInfo {
        render::TextureInfo {
            texture_index: self.texture_index,
            image_index: self.image_index,
            sampler: self.sampler.into(),
            tex_coord: self.tex_coord,
            transform: render::TextureTransform {
                offset: self.transform.offset,
                rotation: self.transform.rotation,
                scale: self.transform.scale,
                tex_coord_override: self.transform.tex_coord_override,
            },
            role,
            color_space: role.color_space(),
        }
    }
}

impl From<Vrm1Sampler> for render::GltfSampler {
    fn from(value: Vrm1Sampler) -> Self {
        Self {
            index: value.index,
            mag_filter: value.mag_filter.map(|filter| match filter {
                Vrm1MagFilter::Nearest => render::GltfMagFilter::Nearest,
                Vrm1MagFilter::Linear => render::GltfMagFilter::Linear,
            }),
            min_filter: value.min_filter.map(|filter| match filter {
                Vrm1MinFilter::Nearest => render::GltfMinFilter::Nearest,
                Vrm1MinFilter::Linear => render::GltfMinFilter::Linear,
                Vrm1MinFilter::NearestMipmapNearest => render::GltfMinFilter::NearestMipmapNearest,
                Vrm1MinFilter::LinearMipmapNearest => render::GltfMinFilter::LinearMipmapNearest,
                Vrm1MinFilter::NearestMipmapLinear => render::GltfMinFilter::NearestMipmapLinear,
                Vrm1MinFilter::LinearMipmapLinear => render::GltfMinFilter::LinearMipmapLinear,
            }),
            wrap_s: match value.wrap_s {
                Vrm1WrapMode::ClampToEdge => render::GltfWrapMode::ClampToEdge,
                Vrm1WrapMode::MirroredRepeat => render::GltfWrapMode::MirroredRepeat,
                Vrm1WrapMode::Repeat => render::GltfWrapMode::Repeat,
            },
            wrap_t: match value.wrap_t {
                Vrm1WrapMode::ClampToEdge => render::GltfWrapMode::ClampToEdge,
                Vrm1WrapMode::MirroredRepeat => render::GltfWrapMode::MirroredRepeat,
                Vrm1WrapMode::Repeat => render::GltfWrapMode::Repeat,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1ScaledTextureInfo {
    pub texture: Vrm1TextureInfo,
    pub scale: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1MtoonMaterial {
    pub material_index: usize,

    // Inherited glTF material inputs used by MToon.
    pub base_color_factor: [f32; 4],
    pub base_color_texture: Option<Vrm1TextureInfo>,
    pub normal_texture: Option<Vrm1ScaledTextureInfo>,
    pub emissive_factor: [f32; 3],
    pub emissive_texture: Option<Vrm1TextureInfo>,
    pub alpha_mode: Vrm1MtoonAlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,

    // VRMC_materials_mtoon 1.0.
    pub spec_version: String,
    pub transparent_with_z_write: bool,
    pub render_queue_offset_number: i32,
    pub shade_color_factor: [f32; 3],
    pub shade_multiply_texture: Option<Vrm1TextureInfo>,
    pub shading_shift_factor: f32,
    pub shading_shift_texture: Option<Vrm1ScaledTextureInfo>,
    pub shading_toony_factor: f32,
    pub gi_equalization_factor: f32,
    pub matcap_factor: [f32; 3],
    pub matcap_texture: Option<Vrm1TextureInfo>,
    pub parametric_rim_color_factor: [f32; 3],
    pub parametric_rim_fresnel_power_factor: f32,
    pub parametric_rim_lift_factor: f32,
    pub rim_multiply_texture: Option<Vrm1TextureInfo>,
    pub rim_lighting_mix_factor: f32,
    pub outline_width_mode: Vrm1MtoonOutlineWidthMode,
    pub outline_width_factor: f32,
    pub outline_width_multiply_texture: Option<Vrm1TextureInfo>,
    pub outline_color_factor: [f32; 3],
    pub outline_lighting_mix_factor: f32,
    pub uv_animation_mask_texture: Option<Vrm1TextureInfo>,
    pub uv_animation_scroll_x_speed_factor: f32,
    pub uv_animation_scroll_y_speed_factor: f32,
    pub uv_animation_rotation_speed_factor: f32,
}

impl Vrm1MtoonMaterial {
    pub fn to_pocket3d_descriptor(&self) -> render::MtoonMaterialDescriptor {
        let alpha_mode = match self.alpha_mode {
            Vrm1MtoonAlphaMode::Opaque => render::MaterialAlphaMode::Opaque,
            Vrm1MtoonAlphaMode::Mask => render::MaterialAlphaMode::Mask,
            Vrm1MtoonAlphaMode::Blend => render::MaterialAlphaMode::Blend,
        };
        render::MtoonMaterialDescriptor {
            material_index: self.material_index,
            inputs: render::MaterialInputs {
                base_color_factor: self.base_color_factor,
                base_color_texture: self
                    .base_color_texture
                    .as_ref()
                    .map(|texture| texture.to_render(render::TextureRole::BaseColor)),
                normal_texture: self.normal_texture.as_ref().map(|texture| {
                    render::ScaledTextureInfo {
                        texture: texture.texture.to_render(render::TextureRole::Normal),
                        scale: texture.scale,
                    }
                }),
                emissive_factor: self.emissive_factor,
                emissive_texture: self
                    .emissive_texture
                    .as_ref()
                    .map(|texture| texture.to_render(render::TextureRole::Emissive)),
                alpha_mode,
                alpha_cutoff: self.alpha_cutoff,
                double_sided: self.double_sided,
            },
            mtoon: render::MtoonMaterial {
                spec_version: self.spec_version.clone(),
                transparent_with_z_write: self.transparent_with_z_write,
                render_queue_offset_number: self.render_queue_offset_number,
                shade_color_factor: self.shade_color_factor,
                shade_multiply_texture: self
                    .shade_multiply_texture
                    .as_ref()
                    .map(|texture| texture.to_render(render::TextureRole::ShadeMultiply)),
                shading_shift_factor: self.shading_shift_factor,
                shading_shift_texture: self.shading_shift_texture.as_ref().map(|texture| {
                    render::ScaledTextureInfo {
                        texture: texture.texture.to_render(render::TextureRole::ShadingShift),
                        scale: texture.scale,
                    }
                }),
                shading_toony_factor: self.shading_toony_factor,
                gi_equalization_factor: self.gi_equalization_factor,
                matcap_factor: self.matcap_factor,
                matcap_texture: self
                    .matcap_texture
                    .as_ref()
                    .map(|texture| texture.to_render(render::TextureRole::Matcap)),
                parametric_rim_color_factor: self.parametric_rim_color_factor,
                parametric_rim_fresnel_power_factor: self.parametric_rim_fresnel_power_factor,
                parametric_rim_lift_factor: self.parametric_rim_lift_factor,
                rim_multiply_texture: self
                    .rim_multiply_texture
                    .as_ref()
                    .map(|texture| texture.to_render(render::TextureRole::RimMultiply)),
                rim_lighting_mix_factor: self.rim_lighting_mix_factor,
                outline_width_mode: match self.outline_width_mode {
                    Vrm1MtoonOutlineWidthMode::None => render::MtoonOutlineWidthMode::None,
                    Vrm1MtoonOutlineWidthMode::WorldCoordinates => {
                        render::MtoonOutlineWidthMode::WorldCoordinates
                    }
                    Vrm1MtoonOutlineWidthMode::ScreenCoordinates => {
                        render::MtoonOutlineWidthMode::ScreenCoordinates
                    }
                },
                outline_width_factor: self.outline_width_factor,
                outline_width_multiply_texture: self
                    .outline_width_multiply_texture
                    .as_ref()
                    .map(|texture| texture.to_render(render::TextureRole::OutlineWidth)),
                outline_color_factor: self.outline_color_factor,
                outline_lighting_mix_factor: self.outline_lighting_mix_factor,
                uv_animation_mask_texture: self
                    .uv_animation_mask_texture
                    .as_ref()
                    .map(|texture| texture.to_render(render::TextureRole::UvAnimationMask)),
                uv_animation_scroll_x_speed_factor: self.uv_animation_scroll_x_speed_factor,
                uv_animation_scroll_y_speed_factor: self.uv_animation_scroll_y_speed_factor,
                uv_animation_rotation_speed_factor: self.uv_animation_rotation_speed_factor,
            },
        }
    }
}

pub(crate) fn parse_mtoon_materials(root: &Value) -> Result<Vec<Vrm1MtoonMaterial>> {
    let Some(materials_value) = root.get("materials") else {
        return Ok(Vec::new());
    };
    let materials = materials_value
        .as_array()
        .context("glTF materials must be an array")?;
    let mut parsed = Vec::new();
    for (material_index, material_value) in materials.iter().enumerate() {
        let material = material_value
            .as_object()
            .with_context(|| format!("glTF material {material_index} must be an object"))?;
        let Some(extensions_value) = material.get("extensions") else {
            continue;
        };
        let extensions = extensions_value.as_object().with_context(|| {
            format!("glTF material {material_index}.extensions must be an object")
        })?;
        let Some(mtoon_value) = extensions.get("VRMC_materials_mtoon") else {
            continue;
        };
        parsed.push(parse_material(root, material_index, material, mtoon_value)?);
    }
    Ok(parsed)
}

fn parse_material(
    root: &Value,
    material_index: usize,
    material: &Map<String, Value>,
    mtoon_value: &Value,
) -> Result<Vrm1MtoonMaterial> {
    let context = format!("material {material_index}");
    let mtoon = mtoon_value
        .as_object()
        .with_context(|| format!("{context}.extensions.VRMC_materials_mtoon must be an object"))?;
    let mtoon_context = format!("{context}.extensions.VRMC_materials_mtoon");
    let spec_version = required_string(mtoon, "specVersion", &mtoon_context)?;
    ensure!(
        spec_version == "1.0",
        "{mtoon_context}.specVersion must be \"1.0\", got {spec_version:?}"
    );

    let pbr = optional_object(material, "pbrMetallicRoughness", &context)?;
    let alpha_mode = match optional_string(material, "alphaMode", &context)?.as_deref() {
        None | Some("OPAQUE") => Vrm1MtoonAlphaMode::Opaque,
        Some("MASK") => Vrm1MtoonAlphaMode::Mask,
        Some("BLEND") => Vrm1MtoonAlphaMode::Blend,
        Some(value) => bail!("{context}.alphaMode has unsupported value {value:?}"),
    };
    let transparent_with_z_write =
        optional_bool(mtoon, "transparentWithZWrite", &mtoon_context)?.unwrap_or(false);
    let render_queue_offset_number =
        optional_i32(mtoon, "renderQueueOffsetNumber", &mtoon_context)?.unwrap_or(0);
    validate_render_queue(
        material_index,
        alpha_mode,
        transparent_with_z_write,
        render_queue_offset_number,
    )?;

    Ok(Vrm1MtoonMaterial {
        material_index,
        base_color_factor: pbr
            .map(|pbr| {
                bounded_vec4(
                    pbr,
                    "baseColorFactor",
                    &format!("{context}.pbrMetallicRoughness"),
                    [1.0; 4],
                )
            })
            .transpose()?
            .unwrap_or([1.0; 4]),
        base_color_texture: pbr
            .and_then(|pbr| pbr.get("baseColorTexture"))
            .map(|value| {
                parse_texture_info(
                    root,
                    value,
                    &format!("{context}.pbrMetallicRoughness.baseColorTexture"),
                )
            })
            .transpose()?,
        normal_texture: material
            .get("normalTexture")
            .map(|value| {
                parse_scaled_texture_info(root, value, &format!("{context}.normalTexture"), 1.0)
            })
            .transpose()?,
        emissive_factor: bounded_vec3(material, "emissiveFactor", &context, [0.0; 3])?,
        emissive_texture: material
            .get("emissiveTexture")
            .map(|value| parse_texture_info(root, value, &format!("{context}.emissiveTexture")))
            .transpose()?,
        alpha_mode,
        alpha_cutoff: nonnegative_field(material, "alphaCutoff", &context, 0.5)?,
        double_sided: optional_bool(material, "doubleSided", &context)?.unwrap_or(false),
        spec_version,
        transparent_with_z_write,
        render_queue_offset_number,
        // The normative 1.0 prose/reference default is black. The published
        // schema's [1,1,1] annotation is intentionally not used here.
        shade_color_factor: bounded_vec3(mtoon, "shadeColorFactor", &mtoon_context, [0.0; 3])?,
        shade_multiply_texture: optional_texture_info(
            root,
            mtoon,
            "shadeMultiplyTexture",
            &mtoon_context,
        )?,
        shading_shift_factor: optional_f32(mtoon, "shadingShiftFactor", &mtoon_context)?
            .unwrap_or(0.0),
        shading_shift_texture: optional_scaled_texture_info(
            root,
            mtoon,
            "shadingShiftTexture",
            &mtoon_context,
            1.0,
        )?,
        shading_toony_factor: bounded_field(
            mtoon,
            "shadingToonyFactor",
            &mtoon_context,
            0.9,
            0.0,
            1.0,
        )?,
        gi_equalization_factor: bounded_field(
            mtoon,
            "giEqualizationFactor",
            &mtoon_context,
            0.9,
            0.0,
            1.0,
        )?,
        matcap_factor: bounded_vec3(mtoon, "matcapFactor", &mtoon_context, [1.0; 3])?,
        matcap_texture: optional_texture_info(root, mtoon, "matcapTexture", &mtoon_context)?,
        parametric_rim_color_factor: bounded_vec3(
            mtoon,
            "parametricRimColorFactor",
            &mtoon_context,
            [0.0; 3],
        )?,
        parametric_rim_fresnel_power_factor: nonnegative_field(
            mtoon,
            "parametricRimFresnelPowerFactor",
            &mtoon_context,
            5.0,
        )?,
        parametric_rim_lift_factor: optional_f32(mtoon, "parametricRimLiftFactor", &mtoon_context)?
            .unwrap_or(0.0),
        rim_multiply_texture: optional_texture_info(
            root,
            mtoon,
            "rimMultiplyTexture",
            &mtoon_context,
        )?,
        rim_lighting_mix_factor: bounded_field(
            mtoon,
            "rimLightingMixFactor",
            &mtoon_context,
            1.0,
            0.0,
            1.0,
        )?,
        outline_width_mode: match optional_string(mtoon, "outlineWidthMode", &mtoon_context)?
            .as_deref()
        {
            None | Some("none") => Vrm1MtoonOutlineWidthMode::None,
            Some("worldCoordinates") => Vrm1MtoonOutlineWidthMode::WorldCoordinates,
            Some("screenCoordinates") => Vrm1MtoonOutlineWidthMode::ScreenCoordinates,
            Some(value) => bail!("{mtoon_context}.outlineWidthMode has invalid value {value:?}"),
        },
        outline_width_factor: nonnegative_field(mtoon, "outlineWidthFactor", &mtoon_context, 0.0)?,
        outline_width_multiply_texture: optional_texture_info(
            root,
            mtoon,
            "outlineWidthMultiplyTexture",
            &mtoon_context,
        )?,
        outline_color_factor: bounded_vec3(mtoon, "outlineColorFactor", &mtoon_context, [0.0; 3])?,
        outline_lighting_mix_factor: bounded_field(
            mtoon,
            "outlineLightingMixFactor",
            &mtoon_context,
            1.0,
            0.0,
            1.0,
        )?,
        uv_animation_mask_texture: optional_texture_info(
            root,
            mtoon,
            "uvAnimationMaskTexture",
            &mtoon_context,
        )?,
        uv_animation_scroll_x_speed_factor: optional_f32(
            mtoon,
            "uvAnimationScrollXSpeedFactor",
            &mtoon_context,
        )?
        .unwrap_or(0.0),
        uv_animation_scroll_y_speed_factor: optional_f32(
            mtoon,
            "uvAnimationScrollYSpeedFactor",
            &mtoon_context,
        )?
        .unwrap_or(0.0),
        uv_animation_rotation_speed_factor: optional_f32(
            mtoon,
            "uvAnimationRotationSpeedFactor",
            &mtoon_context,
        )?
        .unwrap_or(0.0),
    })
}

fn validate_render_queue(
    material_index: usize,
    alpha_mode: Vrm1MtoonAlphaMode,
    transparent_with_z_write: bool,
    value: i32,
) -> Result<()> {
    let valid = match (alpha_mode, transparent_with_z_write) {
        (Vrm1MtoonAlphaMode::Opaque | Vrm1MtoonAlphaMode::Mask, _) => value == 0,
        (Vrm1MtoonAlphaMode::Blend, true) => (0..=9).contains(&value),
        (Vrm1MtoonAlphaMode::Blend, false) => (-9..=0).contains(&value),
    };
    ensure!(
        valid,
        "material {material_index}.extensions.VRMC_materials_mtoon.renderQueueOffsetNumber {value} is invalid for alphaMode {alpha_mode:?} and transparentWithZWrite {transparent_with_z_write}"
    );
    Ok(())
}

fn optional_texture_info(
    root: &Value,
    object: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<Option<Vrm1TextureInfo>> {
    object
        .get(key)
        .map(|value| parse_texture_info(root, value, &format!("{context}.{key}")))
        .transpose()
}

fn optional_scaled_texture_info(
    root: &Value,
    object: &Map<String, Value>,
    key: &str,
    context: &str,
    default_scale: f32,
) -> Result<Option<Vrm1ScaledTextureInfo>> {
    object
        .get(key)
        .map(|value| {
            parse_scaled_texture_info(root, value, &format!("{context}.{key}"), default_scale)
        })
        .transpose()
}

fn parse_scaled_texture_info(
    root: &Value,
    value: &Value,
    context: &str,
    default_scale: f32,
) -> Result<Vrm1ScaledTextureInfo> {
    let object = value
        .as_object()
        .with_context(|| format!("{context} must be an object"))?;
    Ok(Vrm1ScaledTextureInfo {
        texture: parse_texture_info(root, value, context)?,
        scale: optional_f32(object, "scale", context)?.unwrap_or(default_scale),
    })
}

fn parse_texture_info(root: &Value, value: &Value, context: &str) -> Result<Vrm1TextureInfo> {
    let object = value
        .as_object()
        .with_context(|| format!("{context} must be an object"))?;
    let texture_index = required_index(object, "index", context)?;
    let textures = root
        .get("textures")
        .and_then(Value::as_array)
        .context("glTF textures must be an array when a material references a texture")?;
    let texture = textures
        .get(texture_index)
        .with_context(|| {
            format!(
                "{context}.index {texture_index} is out of range for {} textures",
                textures.len()
            )
        })?
        .as_object()
        .with_context(|| format!("glTF texture {texture_index} must be an object"))?;
    let image_index = required_index(texture, "source", &format!("texture {texture_index}"))?;
    let image_count = optional_array(root, "images")?.map_or(0, <[_]>::len);
    ensure!(
        image_index < image_count,
        "{context}.index {texture_index} references image {image_index}, but glTF has {image_count} images"
    );
    let sampler_index = optional_index(texture, "sampler", &format!("texture {texture_index}"))?;
    let sampler = parse_sampler(root, sampler_index, context)?;
    let tex_coord = optional_u32(object, "texCoord", context)?.unwrap_or(0);
    let transform = parse_texture_transform(object, context)?;
    Ok(Vrm1TextureInfo {
        texture_index,
        image_index,
        sampler,
        tex_coord,
        transform,
    })
}

fn parse_texture_transform(
    texture_info: &Map<String, Value>,
    context: &str,
) -> Result<Vrm1TextureTransform> {
    let Some(extensions_value) = texture_info.get("extensions") else {
        return Ok(Vrm1TextureTransform::default());
    };
    let extensions = extensions_value
        .as_object()
        .with_context(|| format!("{context}.extensions must be an object"))?;
    let Some(transform_value) = extensions.get("KHR_texture_transform") else {
        return Ok(Vrm1TextureTransform::default());
    };
    let transform = transform_value
        .as_object()
        .with_context(|| format!("{context}.extensions.KHR_texture_transform must be an object"))?;
    let transform_context = format!("{context}.extensions.KHR_texture_transform");
    Ok(Vrm1TextureTransform {
        offset: vec2_field(transform, "offset", &transform_context, [0.0, 0.0])?,
        rotation: optional_f32(transform, "rotation", &transform_context)?.unwrap_or(0.0),
        scale: vec2_field(transform, "scale", &transform_context, [1.0, 1.0])?,
        tex_coord_override: optional_u32(transform, "texCoord", &transform_context)?,
    })
}

fn parse_sampler(root: &Value, index: Option<usize>, context: &str) -> Result<Vrm1Sampler> {
    let Some(index) = index else {
        return Ok(Vrm1Sampler::default());
    };
    let samplers = root
        .get("samplers")
        .and_then(Value::as_array)
        .context("glTF samplers must be an array when a texture references a sampler")?;
    let sampler = samplers
        .get(index)
        .with_context(|| {
            format!(
                "{context} references sampler {index}, but glTF has {} samplers",
                samplers.len()
            )
        })?
        .as_object()
        .with_context(|| format!("glTF sampler {index} must be an object"))?;
    Ok(Vrm1Sampler {
        index: Some(index),
        mag_filter: sampler
            .get("magFilter")
            .map(
                |value| match integer(value, &format!("sampler {index}.magFilter"))? {
                    9728 => Ok(Vrm1MagFilter::Nearest),
                    9729 => Ok(Vrm1MagFilter::Linear),
                    value => bail!("sampler {index}.magFilter has invalid glTF value {value}"),
                },
            )
            .transpose()?,
        min_filter: sampler
            .get("minFilter")
            .map(
                |value| match integer(value, &format!("sampler {index}.minFilter"))? {
                    9728 => Ok(Vrm1MinFilter::Nearest),
                    9729 => Ok(Vrm1MinFilter::Linear),
                    9984 => Ok(Vrm1MinFilter::NearestMipmapNearest),
                    9985 => Ok(Vrm1MinFilter::LinearMipmapNearest),
                    9986 => Ok(Vrm1MinFilter::NearestMipmapLinear),
                    9987 => Ok(Vrm1MinFilter::LinearMipmapLinear),
                    value => bail!("sampler {index}.minFilter has invalid glTF value {value}"),
                },
            )
            .transpose()?,
        wrap_s: wrap_mode(sampler.get("wrapS"), &format!("sampler {index}.wrapS"))?,
        wrap_t: wrap_mode(sampler.get("wrapT"), &format!("sampler {index}.wrapT"))?,
    })
}

fn wrap_mode(value: Option<&Value>, context: &str) -> Result<Vrm1WrapMode> {
    match value {
        None => Ok(Vrm1WrapMode::Repeat),
        Some(value) => match integer(value, context)? {
            33071 => Ok(Vrm1WrapMode::ClampToEdge),
            33648 => Ok(Vrm1WrapMode::MirroredRepeat),
            10497 => Ok(Vrm1WrapMode::Repeat),
            value => bail!("{context} has invalid glTF value {value}"),
        },
    }
}

fn optional_object<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<Option<&'a Map<String, Value>>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_object()
                .with_context(|| format!("{context}.{key} must be an object"))
        })
        .transpose()
}

fn optional_array<'a>(root: &'a Value, key: &str) -> Result<Option<&'a [Value]>> {
    root.get(key)
        .map(|value| {
            value
                .as_array()
                .map(Vec::as_slice)
                .with_context(|| format!("glTF {key} must be an array"))
        })
        .transpose()
}

fn required_string(object: &Map<String, Value>, key: &str, context: &str) -> Result<String> {
    let value = object
        .get(key)
        .with_context(|| format!("{context} is missing required {key}"))?
        .as_str()
        .with_context(|| format!("{context}.{key} must be a string"))?;
    ensure!(!value.is_empty(), "{context}.{key} must not be empty");
    Ok(value.to_owned())
}

fn optional_string(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<Option<String>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("{context}.{key} must be a string"))
        })
        .transpose()
}

fn optional_bool(object: &Map<String, Value>, key: &str, context: &str) -> Result<Option<bool>> {
    object
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .with_context(|| format!("{context}.{key} must be a boolean"))
        })
        .transpose()
}

fn integer(value: &Value, context: &str) -> Result<i64> {
    value
        .as_i64()
        .with_context(|| format!("{context} must be an integer"))
}

fn required_index(object: &Map<String, Value>, key: &str, context: &str) -> Result<usize> {
    let value = object
        .get(key)
        .with_context(|| format!("{context} is missing required {key}"))?;
    index(value, &format!("{context}.{key}"))
}

fn index(value: &Value, context: &str) -> Result<usize> {
    let value = value
        .as_u64()
        .with_context(|| format!("{context} must be a non-negative integer"))?;
    usize::try_from(value).with_context(|| format!("{context} does not fit in usize"))
}

fn optional_index(object: &Map<String, Value>, key: &str, context: &str) -> Result<Option<usize>> {
    object
        .get(key)
        .map(|value| index(value, &format!("{context}.{key}")))
        .transpose()
}

fn optional_i32(object: &Map<String, Value>, key: &str, context: &str) -> Result<Option<i32>> {
    object
        .get(key)
        .map(|value| {
            let value = integer(value, &format!("{context}.{key}"))?;
            i32::try_from(value).with_context(|| format!("{context}.{key} does not fit in i32"))
        })
        .transpose()
}

fn optional_u32(object: &Map<String, Value>, key: &str, context: &str) -> Result<Option<u32>> {
    object
        .get(key)
        .map(|value| {
            let value = index(value, &format!("{context}.{key}"))?;
            u32::try_from(value).with_context(|| format!("{context}.{key} does not fit in u32"))
        })
        .transpose()
}

fn number(value: &Value, context: &str) -> Result<f32> {
    let value = value
        .as_f64()
        .with_context(|| format!("{context} must be a number"))? as f32;
    ensure!(
        value.is_finite(),
        "{context} must remain finite after f32 conversion"
    );
    Ok(value)
}

fn optional_f32(object: &Map<String, Value>, key: &str, context: &str) -> Result<Option<f32>> {
    object
        .get(key)
        .map(|value| number(value, &format!("{context}.{key}")))
        .transpose()
}

fn bounded_field(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
    default: f32,
    min: f32,
    max: f32,
) -> Result<f32> {
    let value = optional_f32(object, key, context)?.unwrap_or(default);
    ensure!(
        (min..=max).contains(&value),
        "{context}.{key} must be in [{min}, {max}], got {value}"
    );
    Ok(value)
}

fn nonnegative_field(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
    default: f32,
) -> Result<f32> {
    let value = optional_f32(object, key, context)?.unwrap_or(default);
    ensure!(
        value >= 0.0,
        "{context}.{key} must be non-negative, got {value}"
    );
    Ok(value)
}

fn vec2_field(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
    default: [f32; 2],
) -> Result<[f32; 2]> {
    let Some(value) = object.get(key) else {
        return Ok(default);
    };
    let values = value
        .as_array()
        .with_context(|| format!("{context}.{key} must be an array"))?;
    ensure!(
        values.len() == 2,
        "{context}.{key} must contain exactly two numbers"
    );
    Ok([
        number(&values[0], &format!("{context}.{key}[0]"))?,
        number(&values[1], &format!("{context}.{key}[1]"))?,
    ])
}

fn bounded_vec3(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
    default: [f32; 3],
) -> Result<[f32; 3]> {
    let Some(value) = object.get(key) else {
        return Ok(default);
    };
    let values = value
        .as_array()
        .with_context(|| format!("{context}.{key} must be an array"))?;
    ensure!(
        values.len() == 3,
        "{context}.{key} must contain exactly three numbers"
    );
    let mut out = [0.0; 3];
    for (index, value) in values.iter().enumerate() {
        out[index] = number(value, &format!("{context}.{key}[{index}]"))?;
        ensure!(
            (0.0..=1.0).contains(&out[index]),
            "{context}.{key}[{index}] must be in [0, 1], got {}",
            out[index]
        );
    }
    Ok(out)
}

fn bounded_vec4(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
    default: [f32; 4],
) -> Result<[f32; 4]> {
    let Some(value) = object.get(key) else {
        return Ok(default);
    };
    let values = value
        .as_array()
        .with_context(|| format!("{context}.{key} must be an array"))?;
    ensure!(
        values.len() == 4,
        "{context}.{key} must contain exactly four numbers"
    );
    let mut out = [0.0; 4];
    for (index, value) in values.iter().enumerate() {
        out[index] = number(value, &format!("{context}.{key}[{index}]"))?;
        ensure!(
            (0.0..=1.0).contains(&out[index]),
            "{context}.{key}[{index}] must be in [0, 1], got {}",
            out[index]
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn root(extension: Value) -> Value {
        json!({
            "materials": [{"extensions":{"VRMC_materials_mtoon":extension}}],
            "textures": [{"source":0}],
            "images": [{}]
        })
    }

    fn parse(extension: Value) -> Result<Vrm1MtoonMaterial> {
        Ok(parse_mtoon_materials(&root(extension))?.remove(0))
    }

    #[test]
    fn minimal_valid_mtoon_has_every_normative_default() {
        let material = parse(json!({"specVersion":"1.0"})).unwrap();
        assert_eq!(material.material_index, 0);
        assert_eq!(material.base_color_factor, [1.0; 4]);
        assert_eq!(material.base_color_texture, None);
        assert_eq!(material.normal_texture, None);
        assert_eq!(material.emissive_factor, [0.0; 3]);
        assert_eq!(material.emissive_texture, None);
        assert_eq!(material.alpha_mode, Vrm1MtoonAlphaMode::Opaque);
        assert_eq!(material.alpha_cutoff, 0.5);
        assert!(!material.double_sided);
        assert_eq!(material.spec_version, "1.0");
        assert!(!material.transparent_with_z_write);
        assert_eq!(material.render_queue_offset_number, 0);
        assert_eq!(material.shade_color_factor, [0.0; 3]);
        assert_eq!(material.shade_multiply_texture, None);
        assert_eq!(material.shading_shift_factor, 0.0);
        assert_eq!(material.shading_shift_texture, None);
        assert_eq!(material.shading_toony_factor, 0.9);
        assert_eq!(material.gi_equalization_factor, 0.9);
        assert_eq!(material.matcap_factor, [1.0; 3]);
        assert_eq!(material.matcap_texture, None);
        assert_eq!(material.parametric_rim_color_factor, [0.0; 3]);
        assert_eq!(material.parametric_rim_fresnel_power_factor, 5.0);
        assert_eq!(material.parametric_rim_lift_factor, 0.0);
        assert_eq!(material.rim_multiply_texture, None);
        assert_eq!(material.rim_lighting_mix_factor, 1.0);
        assert_eq!(material.outline_width_mode, Vrm1MtoonOutlineWidthMode::None);
        assert_eq!(material.outline_width_factor, 0.0);
        assert_eq!(material.outline_width_multiply_texture, None);
        assert_eq!(material.outline_color_factor, [0.0; 3]);
        assert_eq!(material.outline_lighting_mix_factor, 1.0);
        assert_eq!(material.uv_animation_mask_texture, None);
        assert_eq!(material.uv_animation_scroll_x_speed_factor, 0.0);
        assert_eq!(material.uv_animation_scroll_y_speed_factor, 0.0);
        assert_eq!(material.uv_animation_rotation_speed_factor, 0.0);
    }

    #[test]
    fn all_authored_fields_and_texture_transform_are_retained() {
        let mut value = root(json!({
            "specVersion":"1.0",
            "transparentWithZWrite":true,
            "renderQueueOffsetNumber":4,
            "shadeColorFactor":[0.1,0.2,0.3],
            "shadeMultiplyTexture":{"index":0,"texCoord":0,"extensions":{"KHR_texture_transform":{"offset":[0.2,0.3],"rotation":0.4,"scale":[0.5,0.6],"texCoord":1}}},
            "shadingShiftFactor":-0.2,
            "shadingShiftTexture":{"index":0,"scale":0.7},
            "shadingToonyFactor":0.8,
            "giEqualizationFactor":0.7,
            "matcapFactor":[0.2,0.3,0.4],
            "matcapTexture":{"index":0},
            "parametricRimColorFactor":[0.3,0.4,0.5],
            "parametricRimFresnelPowerFactor":2.0,
            "parametricRimLiftFactor":-0.1,
            "rimMultiplyTexture":{"index":0},
            "rimLightingMixFactor":0.6,
            "outlineWidthMode":"worldCoordinates",
            "outlineWidthFactor":0.01,
            "outlineWidthMultiplyTexture":{"index":0},
            "outlineColorFactor":[0.4,0.5,0.6],
            "outlineLightingMixFactor":0.5,
            "uvAnimationMaskTexture":{"index":0},
            "uvAnimationScrollXSpeedFactor":1.0,
            "uvAnimationScrollYSpeedFactor":-2.0,
            "uvAnimationRotationSpeedFactor":3.0
        }));
        value["materials"][0]["alphaMode"] = json!("BLEND");
        value["materials"][0]["alphaCutoff"] = json!(0.25);
        value["materials"][0]["doubleSided"] = json!(true);
        value["materials"][0]["pbrMetallicRoughness"] = json!({
            "baseColorFactor":[0.6,0.7,0.8,0.9],
            "baseColorTexture":{"index":0}
        });
        value["materials"][0]["normalTexture"] = json!({"index":0,"scale":0.75});
        value["materials"][0]["emissiveFactor"] = json!([0.1, 0.2, 0.3]);
        value["materials"][0]["emissiveTexture"] = json!({"index":0});
        value["samplers"] = json!([{
            "magFilter":9728,"minFilter":9987,"wrapS":33071,"wrapT":33648
        }]);
        value["textures"][0]["sampler"] = json!(0);

        let material = parse_mtoon_materials(&value).unwrap().remove(0);
        assert!(material.transparent_with_z_write);
        assert_eq!(material.render_queue_offset_number, 4);
        assert_eq!(material.base_color_factor, [0.6, 0.7, 0.8, 0.9]);
        assert_eq!(material.normal_texture.as_ref().unwrap().scale, 0.75);
        assert_eq!(material.emissive_factor, [0.1, 0.2, 0.3]);
        assert!(material.double_sided);
        assert_eq!(material.shade_color_factor, [0.1, 0.2, 0.3]);
        let shade = material.shade_multiply_texture.as_ref().unwrap();
        assert_eq!(shade.effective_tex_coord(), 1);
        assert_eq!(shade.transform.offset, [0.2, 0.3]);
        assert_eq!(shade.transform.rotation, 0.4);
        assert_eq!(shade.transform.scale, [0.5, 0.6]);
        assert_eq!(shade.sampler.mag_filter, Some(Vrm1MagFilter::Nearest));
        assert_eq!(
            shade.sampler.min_filter,
            Some(Vrm1MinFilter::LinearMipmapLinear)
        );
        assert_eq!(shade.sampler.wrap_s, Vrm1WrapMode::ClampToEdge);
        assert_eq!(shade.sampler.wrap_t, Vrm1WrapMode::MirroredRepeat);
        assert_eq!(material.shading_shift_texture.as_ref().unwrap().scale, 0.7);
        assert_eq!(
            material.outline_width_mode,
            Vrm1MtoonOutlineWidthMode::WorldCoordinates
        );
        assert_eq!(material.uv_animation_scroll_y_speed_factor, -2.0);

        let descriptor = material.to_pocket3d_descriptor();
        assert_eq!(
            descriptor.mtoon.shade_multiply_texture.unwrap().color_space,
            render::TextureColorSpace::Srgb
        );
        assert_eq!(
            descriptor
                .mtoon
                .uv_animation_mask_texture
                .unwrap()
                .color_space,
            render::TextureColorSpace::Linear
        );
    }

    #[test]
    fn outline_modes_are_strict() {
        for (value, expected) in [
            ("none", Vrm1MtoonOutlineWidthMode::None),
            (
                "worldCoordinates",
                Vrm1MtoonOutlineWidthMode::WorldCoordinates,
            ),
            (
                "screenCoordinates",
                Vrm1MtoonOutlineWidthMode::ScreenCoordinates,
            ),
        ] {
            assert_eq!(
                parse(json!({"specVersion":"1.0","outlineWidthMode":value}))
                    .unwrap()
                    .outline_width_mode,
                expected
            );
        }
        assert!(parse(json!({"specVersion":"1.0","outlineWidthMode":"world"})).is_err());
    }

    #[test]
    fn scalar_and_color_ranges_are_validated() {
        for field in [
            "shadingToonyFactor",
            "giEqualizationFactor",
            "rimLightingMixFactor",
            "outlineLightingMixFactor",
        ] {
            for invalid in [-0.01, 1.01, 3.5e38_f64] {
                let mut extension = json!({"specVersion":"1.0"});
                extension[field] = json!(invalid);
                assert!(parse(extension).is_err(), "{field} accepted {invalid}");
            }
        }
        for field in ["parametricRimFresnelPowerFactor", "outlineWidthFactor"] {
            let mut extension = json!({"specVersion":"1.0"});
            extension[field] = json!(-0.01);
            assert!(
                parse(extension).is_err(),
                "{field} accepted a negative value"
            );
        }
        for field in [
            "shadeColorFactor",
            "matcapFactor",
            "parametricRimColorFactor",
            "outlineColorFactor",
        ] {
            let mut extension = json!({"specVersion":"1.0"});
            extension[field] = json!([0.0, 1.01, 0.0]);
            assert!(
                parse(extension).is_err(),
                "{field} accepted an out-of-range channel"
            );
        }
        for field in [
            "shadingShiftFactor",
            "parametricRimLiftFactor",
            "uvAnimationScrollXSpeedFactor",
            "uvAnimationScrollYSpeedFactor",
            "uvAnimationRotationSpeedFactor",
        ] {
            let mut extension = json!({"specVersion":"1.0"});
            extension[field] = json!(3.5e38_f64);
            assert!(parse(extension).is_err(), "{field} accepted f32 infinity");
        }
    }

    #[test]
    fn render_queue_validation_is_contextual() {
        for (alpha, z_write, good, bad) in [
            ("OPAQUE", false, 0, 1),
            ("MASK", false, 0, -1),
            ("BLEND", true, 9, -1),
            ("BLEND", false, -9, 1),
        ] {
            let mut value = root(json!({
                "specVersion":"1.0",
                "transparentWithZWrite":z_write,
                "renderQueueOffsetNumber":good
            }));
            value["materials"][0]["alphaMode"] = json!(alpha);
            assert!(parse_mtoon_materials(&value).is_ok());
            value["materials"][0]["extensions"]["VRMC_materials_mtoon"]["renderQueueOffsetNumber"] =
                json!(bad);
            let error = parse_mtoon_materials(&value).unwrap_err().to_string();
            assert!(
                error.contains("material 0") && error.contains("renderQueueOffsetNumber"),
                "{error}"
            );
        }
    }

    #[test]
    fn malformed_version_booleans_textures_samplers_and_transforms_fail() {
        for extension in [
            json!({}),
            json!({"specVersion":"0.9"}),
            json!({"specVersion":1.0}),
            json!({"specVersion":"1.0","transparentWithZWrite":"false"}),
            json!({"specVersion":"1.0","shadeMultiplyTexture":null}),
            json!({"specVersion":"1.0","shadeMultiplyTexture":{}}),
            json!({"specVersion":"1.0","shadeMultiplyTexture":{"index":1}}),
            json!({"specVersion":"1.0","shadeMultiplyTexture":{"index":0,"extensions":{"KHR_texture_transform":{"offset":[0.0],"rotation":"bad"}}}}),
        ] {
            assert!(parse(extension.clone()).is_err(), "accepted {extension}");
        }

        let mut bad_image = root(json!({"specVersion":"1.0","matcapTexture":{"index":0}}));
        bad_image["textures"][0]["source"] = json!(2);
        assert!(parse_mtoon_materials(&bad_image).is_err());

        let mut bad_sampler = root(json!({"specVersion":"1.0","matcapTexture":{"index":0}}));
        bad_sampler["textures"][0]["sampler"] = json!(0);
        bad_sampler["samplers"] = json!([{"minFilter":1234}]);
        assert!(parse_mtoon_materials(&bad_sampler).is_err());

        let mut bad_cutoff = root(json!({"specVersion":"1.0"}));
        bad_cutoff["materials"][0]["alphaCutoff"] = json!(-0.1);
        let error = parse_mtoon_materials(&bad_cutoff).unwrap_err().to_string();
        assert!(error.contains("material 0") && error.contains("alphaCutoff"));
    }

    #[test]
    fn texcoord_selection_is_preserved_for_pocket3d_validation() {
        let material = parse(json!({
            "specVersion":"1.0",
            "rimMultiplyTexture":{"index":0,"texCoord":1,"extensions":{"KHR_texture_transform":{"texCoord":2}}}
        }))
        .unwrap();
        let texture = material.rim_multiply_texture.unwrap();
        assert_eq!(texture.tex_coord, 1);
        assert_eq!(texture.effective_tex_coord(), 2);
    }
}
