//! Authored material semantics and per-instance material state.
//!
//! Authored MToon materials use the native surface pipeline when every active
//! semantic is supported. PocketLit and Unlit remain the conservative
//! fallback for later-stage MToon features.

use std::cmp::Ordering;
use std::collections::HashMap;

/// The glTF alpha policy retained on each authored material and primitive.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaterialAlphaMode {
    Opaque,
    Mask,
    Blend,
}

impl MaterialAlphaMode {
    pub(crate) fn from_gltf(mode: gltf::material::AlphaMode) -> Self {
        match mode {
            gltf::material::AlphaMode::Opaque => Self::Opaque,
            gltf::material::AlphaMode::Mask => Self::Mask,
            gltf::material::AlphaMode::Blend => Self::Blend,
        }
    }
}

/// Stable, explicit authored shading-model identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaterialKind {
    PocketLit,
    Unlit,
    Mtoon,
}

/// Texture sampling role, independent of image identity or filename.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TextureRole {
    BaseColor,
    Normal,
    Emissive,
    ShadeMultiply,
    ShadingShift,
    Matcap,
    RimMultiply,
    OutlineWidth,
    UvAnimationMask,
}

impl TextureRole {
    pub const fn color_space(self) -> TextureColorSpace {
        match self {
            Self::BaseColor
            | Self::Emissive
            | Self::ShadeMultiply
            | Self::Matcap
            | Self::RimMultiply => TextureColorSpace::Srgb,
            Self::Normal | Self::ShadingShift | Self::OutlineWidth | Self::UvAnimationMask => {
                TextureColorSpace::Linear
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TextureColorSpace {
    Srgb,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub tex_coord_override: Option<u32>,
}

impl Default for TextureTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            rotation: 0.0,
            scale: [1.0, 1.0],
            tex_coord_override: None,
        }
    }
}

impl TextureTransform {
    /// KHR_texture_transform: offset + rotation * (scale * selected UV).
    pub fn apply_uv(self, uv: [f32; 2]) -> [f32; 2] {
        let (s, c) = self.rotation.sin_cos();
        let x = uv[0] * self.scale[0];
        let y = uv[1] * self.scale[1];
        [
            self.offset[0] + c * x - s * y,
            self.offset[1] + s * x + c * y,
        ]
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GltfMagFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GltfMinFilter {
    Nearest,
    Linear,
    NearestMipmapNearest,
    LinearMipmapNearest,
    NearestMipmapLinear,
    LinearMipmapLinear,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GltfWrapMode {
    ClampToEdge,
    MirroredRepeat,
    Repeat,
}

/// Authored glTF sampler values. Missing filters stay distinguishable from
/// authored values while deterministic conversion applies the glTF defaults.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GltfSampler {
    pub index: Option<usize>,
    pub mag_filter: Option<GltfMagFilter>,
    pub min_filter: Option<GltfMinFilter>,
    pub wrap_s: GltfWrapMode,
    pub wrap_t: GltfWrapMode,
}

impl Default for GltfSampler {
    fn default() -> Self {
        Self {
            index: None,
            mag_filter: None,
            min_filter: None,
            wrap_s: GltfWrapMode::Repeat,
            wrap_t: GltfWrapMode::Repeat,
        }
    }
}

impl GltfSampler {
    pub fn to_wgpu_descriptor(self) -> wgpu::SamplerDescriptor<'static> {
        let uses_mipmaps = !matches!(
            self.min_filter,
            None | Some(GltfMinFilter::Nearest | GltfMinFilter::Linear)
        );
        let (min_filter, mipmap_filter) = match self.min_filter {
            None | Some(GltfMinFilter::Linear) => {
                (wgpu::FilterMode::Linear, wgpu::FilterMode::Nearest)
            }
            Some(GltfMinFilter::Nearest) | Some(GltfMinFilter::NearestMipmapNearest) => {
                (wgpu::FilterMode::Nearest, wgpu::FilterMode::Nearest)
            }
            Some(GltfMinFilter::LinearMipmapNearest) => {
                (wgpu::FilterMode::Linear, wgpu::FilterMode::Nearest)
            }
            Some(GltfMinFilter::NearestMipmapLinear) => {
                (wgpu::FilterMode::Nearest, wgpu::FilterMode::Linear)
            }
            Some(GltfMinFilter::LinearMipmapLinear) => {
                (wgpu::FilterMode::Linear, wgpu::FilterMode::Linear)
            }
        };
        wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wrap_to_wgpu(self.wrap_s),
            address_mode_v: wrap_to_wgpu(self.wrap_t),
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: match self.mag_filter.unwrap_or(GltfMagFilter::Linear) {
                GltfMagFilter::Nearest => wgpu::FilterMode::Nearest,
                GltfMagFilter::Linear => wgpu::FilterMode::Linear,
            },
            min_filter,
            mipmap_filter,
            lod_max_clamp: if uses_mipmaps { 32.0 } else { 0.0 },
            ..Default::default()
        }
    }
}

const fn wrap_to_wgpu(mode: GltfWrapMode) -> wgpu::AddressMode {
    match mode {
        GltfWrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        GltfWrapMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        GltfWrapMode::Repeat => wgpu::AddressMode::Repeat,
    }
}

/// Reuses equivalent authored sampler state even when glTF sampler indices
/// differ. The authored index is metadata and is excluded from equivalence.
#[derive(Default)]
pub struct GltfSamplerCache {
    entries: HashMap<GltfSamplerKey, wgpu::Sampler>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GltfSamplerKey {
    mag_filter: wgpu::FilterMode,
    min_filter: wgpu::FilterMode,
    mipmap_filter: wgpu::FilterMode,
    wrap_s: wgpu::AddressMode,
    wrap_t: wgpu::AddressMode,
    uses_mipmaps: bool,
}

impl From<GltfSampler> for GltfSamplerKey {
    fn from(value: GltfSampler) -> Self {
        let descriptor = value.to_wgpu_descriptor();
        Self {
            mag_filter: descriptor.mag_filter,
            min_filter: descriptor.min_filter,
            mipmap_filter: descriptor.mipmap_filter,
            wrap_s: descriptor.address_mode_u,
            wrap_t: descriptor.address_mode_v,
            uses_mipmaps: descriptor.lod_max_clamp > 0.0,
        }
    }
}

impl GltfSamplerCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get_or_create(&mut self, device: &wgpu::Device, sampler: GltfSampler) -> &wgpu::Sampler {
        self.entries
            .entry(sampler.into())
            .or_insert_with(|| device.create_sampler(&sampler.to_wgpu_descriptor()))
    }
}

/// Render-facing texture reference. Cache/resource identity includes both the
/// source image and `color_space`; one image may legally serve both roles.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureInfo {
    pub texture_index: usize,
    pub image_index: usize,
    pub sampler: GltfSampler,
    pub tex_coord: u32,
    pub transform: TextureTransform,
    pub role: TextureRole,
    pub color_space: TextureColorSpace,
}

impl TextureInfo {
    pub fn effective_tex_coord(&self) -> u32 {
        self.transform.tex_coord_override.unwrap_or(self.tex_coord)
    }

    /// The current PocketLit/Unlit surface shader samples only TEXCOORD_0.
    /// MToon descriptors may retain TEXCOORD_1 for the future native path,
    /// but a base-color fallback must not silently sample the wrong set.
    pub fn current_base_color_fallback_uv_supported(&self) -> bool {
        self.effective_tex_coord() == 0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScaledTextureInfo {
    pub texture: TextureInfo,
    pub scale: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialInputs {
    pub base_color_factor: [f32; 4],
    pub base_color_texture: Option<TextureInfo>,
    pub normal_texture: Option<ScaledTextureInfo>,
    pub emissive_factor: [f32; 3],
    pub emissive_texture: Option<TextureInfo>,
    pub alpha_mode: MaterialAlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

impl Default for MaterialInputs {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0; 4],
            base_color_texture: None,
            normal_texture: None,
            emissive_factor: [0.0; 3],
            emissive_texture: None,
            alpha_mode: MaterialAlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtoonOutlineWidthMode {
    None,
    WorldCoordinates,
    ScreenCoordinates,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MtoonMaterial {
    pub spec_version: String,
    pub transparent_with_z_write: bool,
    pub render_queue_offset_number: i32,
    pub shade_color_factor: [f32; 3],
    pub shade_multiply_texture: Option<TextureInfo>,
    pub shading_shift_factor: f32,
    pub shading_shift_texture: Option<ScaledTextureInfo>,
    pub shading_toony_factor: f32,
    pub gi_equalization_factor: f32,
    pub matcap_factor: [f32; 3],
    pub matcap_texture: Option<TextureInfo>,
    pub parametric_rim_color_factor: [f32; 3],
    pub parametric_rim_fresnel_power_factor: f32,
    pub parametric_rim_lift_factor: f32,
    pub rim_multiply_texture: Option<TextureInfo>,
    pub rim_lighting_mix_factor: f32,
    pub outline_width_mode: MtoonOutlineWidthMode,
    pub outline_width_factor: f32,
    pub outline_width_multiply_texture: Option<TextureInfo>,
    pub outline_color_factor: [f32; 3],
    pub outline_lighting_mix_factor: f32,
    pub uv_animation_mask_texture: Option<TextureInfo>,
    pub uv_animation_scroll_x_speed_factor: f32,
    pub uv_animation_scroll_y_speed_factor: f32,
    pub uv_animation_rotation_speed_factor: f32,
}

/// One typed semantic handoff from PocketVRM, addressed by authored glTF
/// material index. Pocket3D never parses VRM extension JSON.
#[derive(Clone, Debug, PartialEq)]
pub struct MtoonMaterialDescriptor {
    pub material_index: usize,
    pub inputs: MaterialInputs,
    pub mtoon: MtoonMaterial,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PocketLitMaterial;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UnlitMaterial;

#[derive(Clone, Debug, PartialEq)]
pub enum MaterialModel {
    PocketLit(PocketLitMaterial),
    Unlit(UnlitMaterial),
    Mtoon(Box<MtoonMaterial>),
}

impl MaterialModel {
    pub const fn kind(&self) -> MaterialKind {
        match self {
            Self::PocketLit(_) => MaterialKind::PocketLit,
            Self::Unlit(_) => MaterialKind::Unlit,
            Self::Mtoon(_) => MaterialKind::Mtoon,
        }
    }
}

/// Immutable CPU-side authored material stored by `ModelAsset`.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialAsset {
    pub gltf_material_index: usize,
    pub name: Option<String>,
    pub inputs: MaterialInputs,
    pub model: MaterialModel,
}

impl MaterialAsset {
    pub const fn kind(&self) -> MaterialKind {
        self.model.kind()
    }

    pub fn pocket_lit(gltf_material_index: usize, name: Option<String>) -> Self {
        Self {
            gltf_material_index,
            name,
            inputs: MaterialInputs::default(),
            model: MaterialModel::PocketLit(PocketLitMaterial),
        }
    }

    pub fn authored_state(&self) -> MaterialState {
        let mtoon = match &self.model {
            MaterialModel::Mtoon(material) => Some(MtoonMaterialState {
                shade_color_factor: material.shade_color_factor,
                matcap_factor: material.matcap_factor,
                parametric_rim_color_factor: material.parametric_rim_color_factor,
                outline_color_factor: material.outline_color_factor,
            }),
            MaterialModel::PocketLit(_) | MaterialModel::Unlit(_) => None,
        };
        let mut texture_transforms = HashMap::new();
        for texture in self.textures() {
            texture_transforms.insert(texture.role, texture.transform);
        }
        MaterialState {
            base_color_factor: self.inputs.base_color_factor,
            emissive_factor: self.inputs.emissive_factor,
            mtoon,
            texture_transforms,
        }
    }

    pub fn textures(&self) -> Vec<&TextureInfo> {
        let mut textures = Vec::new();
        if let Some(texture) = self.inputs.base_color_texture.as_ref() {
            textures.push(texture);
        }
        if let Some(texture) = self.inputs.normal_texture.as_ref() {
            textures.push(&texture.texture);
        }
        if let Some(texture) = self.inputs.emissive_texture.as_ref() {
            textures.push(texture);
        }
        if let MaterialModel::Mtoon(mtoon) = &self.model {
            textures.extend(
                [
                    mtoon.shade_multiply_texture.as_ref(),
                    mtoon
                        .shading_shift_texture
                        .as_ref()
                        .map(|value| &value.texture),
                    mtoon.matcap_texture.as_ref(),
                    mtoon.rim_multiply_texture.as_ref(),
                    mtoon.outline_width_multiply_texture.as_ref(),
                    mtoon.uv_animation_mask_texture.as_ref(),
                ]
                .into_iter()
                .flatten(),
            );
        }
        textures
    }
}

/// Mutable values owned by one model instance. No value here aliases or
/// mutates the shared `MaterialAsset`.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialState {
    pub base_color_factor: [f32; 4],
    pub emissive_factor: [f32; 3],
    pub mtoon: Option<MtoonMaterialState>,
    pub texture_transforms: HashMap<TextureRole, TextureTransform>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MtoonMaterialState {
    pub shade_color_factor: [f32; 3],
    pub matcap_factor: [f32; 3],
    pub parametric_rim_color_factor: [f32; 3],
    pub outline_color_factor: [f32; 3],
}

/// Per-instance state indexed by authored glTF material index.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialStateSet {
    states: Vec<MaterialState>,
}

impl MaterialStateSet {
    pub fn from_assets(materials: &[MaterialAsset]) -> Self {
        debug_assert!(
            materials
                .iter()
                .enumerate()
                .all(|(index, material)| index == material.gltf_material_index)
        );
        Self {
            states: materials
                .iter()
                .map(MaterialAsset::authored_state)
                .collect(),
        }
    }

    pub fn get(&self, gltf_material_index: usize) -> Option<&MaterialState> {
        self.states.get(gltf_material_index)
    }

    pub fn get_mut(&mut self, gltf_material_index: usize) -> Option<&mut MaterialState> {
        self.states.get_mut(gltf_material_index)
    }

    /// Restore all mutable values from immutable authored material data.
    /// Expression composition calls this before accumulating base-relative
    /// deltas, making results independent of expression order and frame history.
    pub fn reset_from_assets(&mut self, materials: &[MaterialAsset]) {
        self.states.clear();
        self.states
            .extend(materials.iter().map(MaterialAsset::authored_state));
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

/// Ordered render phases required by the eventual MToon renderer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RenderPhase {
    Opaque,
    Mask,
    MtoonBlendZWrite,
    Blend,
}

impl RenderPhase {
    pub const fn fallback(alpha_mode: MaterialAlphaMode) -> Self {
        match alpha_mode {
            MaterialAlphaMode::Opaque => Self::Opaque,
            MaterialAlphaMode::Mask => Self::Mask,
            MaterialAlphaMode::Blend => Self::Blend,
        }
    }

    pub const fn pass_class(self) -> RenderPassClass {
        match self {
            Self::Opaque | Self::Mask => RenderPassClass::Solid,
            Self::MtoonBlendZWrite | Self::Blend => RenderPassClass::Blend,
        }
    }

    /// Whole-instance fades are a presentation override, not an authored
    /// material mutation. Opaque and cutout surfaces join the ordinary
    /// no-depth-write blend phase while fading; authored transparent surfaces
    /// retain their MToon depth-write choice.
    pub const fn with_presentation_alpha(self, presentation_alpha: f32) -> Self {
        if presentation_alpha < 1.0 && matches!(self, Self::Opaque | Self::Mask) {
            Self::Blend
        } else {
            self
        }
    }
}

/// Coarse render-state class shared by CPU ordering and pipeline selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderPassClass {
    Solid,
    Blend,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderSortKey {
    pub phase: RenderPhase,
    pub queue_offset: i32,
    pub camera_depth: f32,
    pub author_draw_order: u64,
}

impl RenderSortKey {
    pub fn compare(&self, other: &Self) -> Ordering {
        self.phase
            .cmp(&other.phase)
            .then_with(|| self.queue_offset.cmp(&other.queue_offset))
            .then_with(|| {
                if self.phase.pass_class() == RenderPassClass::Blend {
                    // Larger camera-forward depth is farther away, so reverse
                    // the comparison for source-over back-to-front rendering.
                    safe_camera_depth(other.camera_depth)
                        .total_cmp(&safe_camera_depth(self.camera_depth))
                } else {
                    Ordering::Equal
                }
            })
            .then_with(|| self.author_draw_order.cmp(&other.author_draw_order))
    }
}

fn safe_camera_depth(depth: f32) -> f32 {
    if depth.is_finite() { depth } else { 0.0 }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineShadingModel {
    PocketLit,
    Unlit,
    Mtoon,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineBlendDepthFamily {
    OpaqueOrMask,
    BlendNoDepthWrite,
    BlendDepthWrite,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineCullMode {
    Back,
    None,
    Front,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaterialRenderPass {
    Surface,
    Outline,
}

/// Only true GPU/pipeline decisions belong in this key. Colors, scalar MToon
/// parameters, UV animation, and texture-presence flags remain uniforms/data.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialPipelineKey {
    pub shading_model: PipelineShadingModel,
    pub blend_depth_family: PipelineBlendDepthFamily,
    pub cull_mode: PipelineCullMode,
    pub render_pass: MaterialRenderPass,
    pub sample_count: u32,
}

impl MaterialPipelineKey {
    /// Native Stage D supports every surface render mode. Queue offsets are
    /// submission metadata and therefore deliberately absent from this key.
    pub const fn native_stage_d(phase: RenderPhase, double_sided: bool, sample_count: u32) -> Self {
        Self {
            shading_model: PipelineShadingModel::Mtoon,
            blend_depth_family: match phase {
                RenderPhase::Opaque | RenderPhase::Mask => PipelineBlendDepthFamily::OpaqueOrMask,
                RenderPhase::MtoonBlendZWrite => PipelineBlendDepthFamily::BlendDepthWrite,
                RenderPhase::Blend => PipelineBlendDepthFamily::BlendNoDepthWrite,
            },
            cull_mode: if double_sided {
                PipelineCullMode::None
            } else {
                PipelineCullMode::Back
            },
            render_pass: MaterialRenderPass::Surface,
            sample_count,
        }
    }

    /// Stage E outline pipelines share the material's blend/depth family but
    /// always cull front faces. Outline width mode and all authored factors are
    /// shader data, so they deliberately do not add pipeline axes.
    pub const fn native_stage_e_outline(phase: RenderPhase, sample_count: u32) -> Self {
        Self {
            shading_model: PipelineShadingModel::Mtoon,
            blend_depth_family: match phase {
                RenderPhase::Opaque | RenderPhase::Mask => PipelineBlendDepthFamily::OpaqueOrMask,
                RenderPhase::MtoonBlendZWrite => PipelineBlendDepthFamily::BlendDepthWrite,
                RenderPhase::Blend => PipelineBlendDepthFamily::BlendNoDepthWrite,
            },
            cull_mode: PipelineCullMode::Front,
            render_pass: MaterialRenderPass::Outline,
            sample_count,
        }
    }

    pub const fn current_fallback(
        unlit: bool,
        phase: RenderPhase,
        double_sided: bool,
        sample_count: u32,
    ) -> Self {
        Self {
            shading_model: if unlit {
                PipelineShadingModel::Unlit
            } else {
                PipelineShadingModel::PocketLit
            },
            blend_depth_family: match phase {
                RenderPhase::Opaque | RenderPhase::Mask => PipelineBlendDepthFamily::OpaqueOrMask,
                RenderPhase::MtoonBlendZWrite => PipelineBlendDepthFamily::BlendDepthWrite,
                RenderPhase::Blend => PipelineBlendDepthFamily::BlendNoDepthWrite,
            },
            cull_mode: if double_sided {
                PipelineCullMode::None
            } else {
                PipelineCullMode::Back
            },
            render_pass: MaterialRenderPass::Surface,
            sample_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_roles_have_explicit_color_spaces() {
        for role in [
            TextureRole::BaseColor,
            TextureRole::ShadeMultiply,
            TextureRole::Emissive,
            TextureRole::Matcap,
            TextureRole::RimMultiply,
        ] {
            assert_eq!(role.color_space(), TextureColorSpace::Srgb, "{role:?}");
        }
        for role in [
            TextureRole::Normal,
            TextureRole::ShadingShift,
            TextureRole::OutlineWidth,
            TextureRole::UvAnimationMask,
        ] {
            assert_eq!(role.color_space(), TextureColorSpace::Linear, "{role:?}");
        }
    }

    #[test]
    fn sampler_conversion_covers_filters_and_wraps() {
        let sampler = GltfSampler {
            index: Some(4),
            mag_filter: Some(GltfMagFilter::Nearest),
            min_filter: Some(GltfMinFilter::NearestMipmapLinear),
            wrap_s: GltfWrapMode::ClampToEdge,
            wrap_t: GltfWrapMode::MirroredRepeat,
        };
        let descriptor = sampler.to_wgpu_descriptor();
        assert_eq!(descriptor.mag_filter, wgpu::FilterMode::Nearest);
        assert_eq!(descriptor.min_filter, wgpu::FilterMode::Nearest);
        assert_eq!(descriptor.mipmap_filter, wgpu::FilterMode::Linear);
        assert_eq!(descriptor.address_mode_u, wgpu::AddressMode::ClampToEdge);
        assert_eq!(descriptor.address_mode_v, wgpu::AddressMode::MirrorRepeat);

        let defaults = GltfSampler::default().to_wgpu_descriptor();
        assert_eq!(defaults.mag_filter, wgpu::FilterMode::Linear);
        assert_eq!(defaults.min_filter, wgpu::FilterMode::Linear);
        assert_eq!(defaults.mipmap_filter, wgpu::FilterMode::Nearest);
        assert_eq!(defaults.address_mode_u, wgpu::AddressMode::Repeat);
        assert_eq!(defaults.address_mode_v, wgpu::AddressMode::Repeat);

        let cases = [
            (
                GltfMinFilter::Nearest,
                wgpu::FilterMode::Nearest,
                wgpu::FilterMode::Nearest,
            ),
            (
                GltfMinFilter::Linear,
                wgpu::FilterMode::Linear,
                wgpu::FilterMode::Nearest,
            ),
            (
                GltfMinFilter::NearestMipmapNearest,
                wgpu::FilterMode::Nearest,
                wgpu::FilterMode::Nearest,
            ),
            (
                GltfMinFilter::LinearMipmapNearest,
                wgpu::FilterMode::Linear,
                wgpu::FilterMode::Nearest,
            ),
            (
                GltfMinFilter::NearestMipmapLinear,
                wgpu::FilterMode::Nearest,
                wgpu::FilterMode::Linear,
            ),
            (
                GltfMinFilter::LinearMipmapLinear,
                wgpu::FilterMode::Linear,
                wgpu::FilterMode::Linear,
            ),
        ];
        for (filter, expected_min, expected_mip) in cases {
            let descriptor = GltfSampler {
                min_filter: Some(filter),
                ..GltfSampler::default()
            }
            .to_wgpu_descriptor();
            assert_eq!(descriptor.min_filter, expected_min, "{filter:?}");
            assert_eq!(descriptor.mipmap_filter, expected_mip, "{filter:?}");
        }
    }

    #[test]
    fn sampler_cache_key_reuses_equivalent_resolved_state() {
        let authored = GltfSampler {
            index: Some(8),
            mag_filter: Some(GltfMagFilter::Linear),
            min_filter: Some(GltfMinFilter::Linear),
            ..GltfSampler::default()
        };
        assert_eq!(
            GltfSamplerKey::from(authored),
            GltfSamplerKey::from(GltfSampler::default())
        );
        let different = GltfSampler {
            min_filter: Some(GltfMinFilter::LinearMipmapLinear),
            ..GltfSampler::default()
        };
        assert_ne!(
            GltfSamplerKey::from(different),
            GltfSamplerKey::from(GltfSampler::default())
        );
    }

    #[test]
    fn texture_transform_preserves_override_and_effective_texcoord() {
        let info = TextureInfo {
            texture_index: 2,
            image_index: 3,
            sampler: GltfSampler::default(),
            tex_coord: 0,
            transform: TextureTransform {
                offset: [0.25, -0.5],
                rotation: 0.75,
                scale: [2.0, 3.0],
                tex_coord_override: Some(1),
            },
            role: TextureRole::BaseColor,
            color_space: TextureColorSpace::Srgb,
        };
        assert_eq!(info.effective_tex_coord(), 1);
        assert!(!info.current_base_color_fallback_uv_supported());
        assert_eq!(info.transform.offset, [0.25, -0.5]);
        assert_eq!(info.transform.rotation, 0.75);
        assert_eq!(info.transform.scale, [2.0, 3.0]);
        let mut supported = info.clone();
        supported.transform.tex_coord_override = Some(0);
        assert!(supported.current_base_color_fallback_uv_supported());
        supported.transform.tex_coord_override = None;
        supported.tex_coord = 1;
        assert!(!supported.current_base_color_fallback_uv_supported());
    }

    #[test]
    fn material_kinds_and_instance_state_are_independent() {
        let pocket_lit = MaterialAsset::pocket_lit(0, Some("lit".into()));
        let unlit = MaterialAsset {
            gltf_material_index: 1,
            name: Some("unlit".into()),
            inputs: MaterialInputs::default(),
            model: MaterialModel::Unlit(UnlitMaterial),
        };
        let mtoon = MaterialAsset {
            gltf_material_index: 2,
            name: Some("mtoon".into()),
            inputs: MaterialInputs::default(),
            model: MaterialModel::Mtoon(Box::new(MtoonMaterial {
                spec_version: "1.0".into(),
                transparent_with_z_write: false,
                render_queue_offset_number: 0,
                shade_color_factor: [0.0; 3],
                shade_multiply_texture: None,
                shading_shift_factor: 0.0,
                shading_shift_texture: None,
                shading_toony_factor: 0.9,
                gi_equalization_factor: 0.9,
                matcap_factor: [1.0; 3],
                matcap_texture: None,
                parametric_rim_color_factor: [0.0; 3],
                parametric_rim_fresnel_power_factor: 5.0,
                parametric_rim_lift_factor: 0.0,
                rim_multiply_texture: None,
                rim_lighting_mix_factor: 1.0,
                outline_width_mode: MtoonOutlineWidthMode::None,
                outline_width_factor: 0.0,
                outline_width_multiply_texture: None,
                outline_color_factor: [0.0; 3],
                outline_lighting_mix_factor: 1.0,
                uv_animation_mask_texture: None,
                uv_animation_scroll_x_speed_factor: 0.0,
                uv_animation_scroll_y_speed_factor: 0.0,
                uv_animation_rotation_speed_factor: 0.0,
            })),
        };
        assert_eq!(pocket_lit.kind(), MaterialKind::PocketLit);
        assert_eq!(unlit.kind(), MaterialKind::Unlit);
        assert_eq!(mtoon.kind(), MaterialKind::Mtoon);

        let assets = vec![pocket_lit, unlit, mtoon];
        let mut first = MaterialStateSet::from_assets(&assets);
        let second = MaterialStateSet::from_assets(&assets);
        first
            .get_mut(2)
            .unwrap()
            .mtoon
            .as_mut()
            .unwrap()
            .shade_color_factor = [0.5; 3];
        assert_eq!(
            assets[2].authored_state().mtoon.unwrap().shade_color_factor,
            [0.0; 3]
        );
        assert_eq!(
            second
                .get(2)
                .unwrap()
                .mtoon
                .as_ref()
                .unwrap()
                .shade_color_factor,
            [0.0; 3]
        );
    }

    #[test]
    fn render_phases_have_explicit_semantic_priority() {
        assert_eq!(
            RenderPhase::fallback(MaterialAlphaMode::Opaque),
            RenderPhase::Opaque
        );
        assert_eq!(
            RenderPhase::fallback(MaterialAlphaMode::Mask),
            RenderPhase::Mask
        );
        assert_eq!(
            RenderPhase::fallback(MaterialAlphaMode::Blend),
            RenderPhase::Blend
        );
        assert_eq!(RenderPhase::Opaque.pass_class(), RenderPassClass::Solid);
        assert_eq!(RenderPhase::Mask.pass_class(), RenderPassClass::Solid);

        let key = |phase, queue_offset| RenderSortKey {
            phase,
            queue_offset,
            camera_depth: 0.0,
            author_draw_order: 0,
        };
        assert_eq!(
            key(RenderPhase::Opaque, i32::MAX).compare(&key(RenderPhase::Mask, i32::MIN)),
            Ordering::Less,
        );
        assert_eq!(
            key(RenderPhase::Mask, i32::MAX).compare(&key(RenderPhase::MtoonBlendZWrite, i32::MIN)),
            Ordering::Less,
        );
        assert_eq!(
            key(RenderPhase::MtoonBlendZWrite, i32::MAX)
                .compare(&key(RenderPhase::Blend, i32::MIN)),
            Ordering::Less,
        );
    }

    #[test]
    fn transparent_sort_uses_queue_then_back_to_front_then_stable_tie() {
        let key = |queue_offset, camera_depth, author_draw_order| RenderSortKey {
            phase: RenderPhase::Blend,
            queue_offset,
            camera_depth,
            author_draw_order,
        };
        let near_early_queue = key(-1, 2.0, 0);
        let far_late_queue = key(0, 20.0, 1);
        assert_eq!(
            near_early_queue.compare(&far_late_queue),
            Ordering::Less,
            "queue offset must outrank depth",
        );

        let near = key(0, 2.0, 0);
        let far = key(0, 20.0, 1);
        assert_eq!(far.compare(&near), Ordering::Less);
        assert_eq!(near.compare(&far), Ordering::Greater);

        let zwrite_near = RenderSortKey {
            phase: RenderPhase::MtoonBlendZWrite,
            ..near
        };
        let zwrite_far = RenderSortKey {
            phase: RenderPhase::MtoonBlendZWrite,
            ..far
        };
        assert_eq!(
            zwrite_far.compare(&zwrite_near),
            Ordering::Less,
            "equal-queue BLEND+ZWrite must also draw back-to-front"
        );

        let first = key(0, 2.0, 4);
        let second = key(0, 2.0, 5);
        assert_eq!(first.compare(&second), Ordering::Less);
        assert_eq!(first.compare(&first), Ordering::Equal);
    }

    #[test]
    fn non_finite_transparent_depth_has_a_deterministic_safe_fallback() {
        let key = |camera_depth, author_draw_order| RenderSortKey {
            phase: RenderPhase::Blend,
            queue_offset: 0,
            camera_depth,
            author_draw_order,
        };
        assert_eq!(key(f32::NAN, 2).compare(&key(0.0, 3)), Ordering::Less);
        assert_eq!(
            key(f32::INFINITY, 2).compare(&key(f32::NEG_INFINITY, 3)),
            Ordering::Less,
        );
    }

    #[test]
    fn presentation_fade_only_overrides_authored_solid_phases() {
        assert_eq!(
            RenderPhase::Opaque.with_presentation_alpha(0.5),
            RenderPhase::Blend
        );
        assert_eq!(
            RenderPhase::Mask.with_presentation_alpha(0.5),
            RenderPhase::Blend
        );
        assert_eq!(
            RenderPhase::MtoonBlendZWrite.with_presentation_alpha(0.5),
            RenderPhase::MtoonBlendZWrite
        );
        assert_eq!(
            RenderPhase::Blend.with_presentation_alpha(0.5),
            RenderPhase::Blend
        );
        assert_eq!(
            RenderPhase::Mask.with_presentation_alpha(1.0),
            RenderPhase::Mask
        );
    }

    #[test]
    fn mtoon_authorship_does_not_activate_native_pipeline() {
        let key = MaterialPipelineKey::current_fallback(true, RenderPhase::Blend, false, 4);
        assert_eq!(key.shading_model, PipelineShadingModel::Unlit);
        assert_eq!(
            key.blend_depth_family,
            PipelineBlendDepthFamily::BlendNoDepthWrite
        );
        assert_eq!(key.render_pass, MaterialRenderPass::Surface);
    }

    #[test]
    fn stage_d_pipeline_key_covers_four_phases_and_both_cull_modes() {
        for phase in [
            RenderPhase::Opaque,
            RenderPhase::Mask,
            RenderPhase::MtoonBlendZWrite,
            RenderPhase::Blend,
        ] {
            for double_sided in [false, true] {
                let key = MaterialPipelineKey::native_stage_d(phase, double_sided, 4);
                assert_eq!(key.shading_model, PipelineShadingModel::Mtoon);
                assert_eq!(
                    key.blend_depth_family,
                    match phase {
                        RenderPhase::Opaque | RenderPhase::Mask => {
                            PipelineBlendDepthFamily::OpaqueOrMask
                        }
                        RenderPhase::MtoonBlendZWrite => {
                            PipelineBlendDepthFamily::BlendDepthWrite
                        }
                        RenderPhase::Blend => PipelineBlendDepthFamily::BlendNoDepthWrite,
                    }
                );
                assert_eq!(key.sample_count, 4);
                assert_eq!(
                    key.cull_mode,
                    if double_sided {
                        PipelineCullMode::None
                    } else {
                        PipelineCullMode::Back
                    }
                );
            }
        }
        assert_eq!(
            MaterialPipelineKey::current_fallback(false, RenderPhase::Opaque, false, 4)
                .shading_model,
            PipelineShadingModel::PocketLit,
        );
        assert_eq!(
            MaterialPipelineKey::current_fallback(true, RenderPhase::Opaque, false, 4)
                .shading_model,
            PipelineShadingModel::Unlit,
        );
    }

    #[test]
    fn stage_e_outline_pipeline_key_has_four_families_and_fixed_front_culling() {
        for phase in [
            RenderPhase::Opaque,
            RenderPhase::Mask,
            RenderPhase::MtoonBlendZWrite,
            RenderPhase::Blend,
        ] {
            let key = MaterialPipelineKey::native_stage_e_outline(phase, 4);
            assert_eq!(key.shading_model, PipelineShadingModel::Mtoon);
            assert_eq!(key.cull_mode, PipelineCullMode::Front);
            assert_eq!(key.render_pass, MaterialRenderPass::Outline);
            assert_eq!(key.sample_count, 4);
            assert_eq!(
                key.blend_depth_family,
                match phase {
                    RenderPhase::Opaque | RenderPhase::Mask => {
                        PipelineBlendDepthFamily::OpaqueOrMask
                    }
                    RenderPhase::MtoonBlendZWrite => PipelineBlendDepthFamily::BlendDepthWrite,
                    RenderPhase::Blend => PipelineBlendDepthFamily::BlendNoDepthWrite,
                }
            );
        }
    }

    #[test]
    fn non_mipmap_min_filters_clamp_to_base_level() {
        for filter in [
            None,
            Some(GltfMinFilter::Nearest),
            Some(GltfMinFilter::Linear),
        ] {
            assert_eq!(
                GltfSampler {
                    min_filter: filter,
                    ..Default::default()
                }
                .to_wgpu_descriptor()
                .lod_max_clamp,
                0.0,
            );
        }
        assert!(
            GltfSampler {
                min_filter: Some(GltfMinFilter::LinearMipmapLinear),
                ..Default::default()
            }
            .to_wgpu_descriptor()
            .lod_max_clamp
                > 0.0
        );
    }

    #[test]
    fn stage_b_uv_selection_and_khr_transform_order() {
        let transform = TextureTransform {
            offset: [0.1, 0.2],
            rotation: std::f32::consts::FRAC_PI_2,
            scale: [2.0, 3.0],
            tex_coord_override: None,
        };
        let make = |role, tex_coord, override_set| TextureInfo {
            texture_index: 0,
            image_index: 0,
            sampler: GltfSampler::default(),
            tex_coord,
            transform: TextureTransform {
                tex_coord_override: override_set,
                ..transform
            },
            role,
            color_space: role.color_space(),
        };
        let uv0 = [0.25, 0.5];
        let uv1 = [0.5, 0.25];
        let selected = |info: &TextureInfo| {
            info.transform.apply_uv(if info.effective_tex_coord() == 0 {
                uv0
            } else {
                uv1
            })
        };
        let base = make(TextureRole::BaseColor, 0, None);
        let normal = make(TextureRole::Normal, 1, None);
        let shift = make(TextureRole::ShadingShift, 0, Some(1));
        let base_uv = selected(&base);
        let normal_uv = selected(&normal);
        let shift_uv = selected(&shift);
        assert!((base_uv[0] + 1.4).abs() < 1e-6);
        assert!((base_uv[1] - 0.7).abs() < 1e-6);
        assert!((normal_uv[0] + 0.65).abs() < 1e-6);
        assert!((normal_uv[1] - 1.2).abs() < 1e-6);
        assert_eq!(normal_uv, shift_uv);
    }

    fn reference_toon(shading: f32, toony: f32) -> f32 {
        let toony = toony.clamp(0.0, 1.0);
        if toony >= 1.0 {
            return f32::from(shading >= 0.0);
        }
        let a = -1.0 + toony;
        let b = 1.0 - toony;
        ((shading - a) / (b - a)).clamp(0.0, 1.0)
    }

    #[test]
    fn stage_b_shift_toony_gi_and_mask_reference_math() {
        let shift = |factor: f32, texture_r: Option<f32>, scale: f32| {
            factor + texture_r.unwrap_or(0.0) * scale
        };
        assert_eq!(shift(0.0, None, 7.0), 0.0);
        assert_eq!(shift(0.25, Some(0.5), 2.0), 1.25);
        assert_eq!(shift(-0.5, Some(0.25), -2.0), -1.0);
        assert_eq!(reference_toon(-1.0, 0.0), 0.0);
        assert_eq!(reference_toon(0.0, 0.0), 0.5);
        assert_eq!(reference_toon(1.0, 0.0), 1.0);
        assert_eq!(reference_toon(-0.5, 0.5), 0.0);
        assert_eq!(reference_toon(0.0, 0.5), 0.5);
        assert_eq!(reference_toon(0.5, 0.5), 1.0);
        assert!(reference_toon(1e-5, 0.9999).is_finite());
        assert_eq!(reference_toon(-1e-5, 1.0), 0.0);
        assert_eq!(reference_toon(0.0, 1.0), 1.0);
        let shade = [0.2, 0.3, 0.4];
        let lit = [0.8, 0.7, 0.6];
        let toon = reference_toon(0.0, 0.5);
        let mixed: Vec<f32> = shade
            .iter()
            .zip(lit)
            .map(|(a, b)| a * (1.0 - toon) + b * toon)
            .collect();
        assert_eq!(mixed, [0.5, 0.5, 0.5]);
        let raw_gi = 0.8;
        let uniform_gi = 0.5;
        assert_eq!(raw_gi * (1.0 - 0.0) + uniform_gi * 0.0, raw_gi);
        assert_eq!(raw_gi * (1.0 - 1.0) + uniform_gi * 1.0, uniform_gi);
        let alpha = 0.6 * 0.7;
        assert!(alpha < 0.5);
        assert!(alpha >= 0.4);
    }

    #[test]
    fn stage_h_matcap_vertical_degeneracy_is_finite_and_deterministic() {
        let reference_matcap_uv = |normal: glam::Vec3, view: glam::Vec3| {
            let horizontal = glam::Vec3::new(view.z, 0.0, -view.x);
            let view_x = if horizontal.length_squared() > 1.0e-10 {
                horizontal.normalize()
            } else {
                glam::Vec3::X
            };
            let view_y = view.cross(view_x);
            glam::Vec2::new(view_x.dot(normal), view_y.dot(normal)) * 0.495 + glam::Vec2::splat(0.5)
        };
        let normal = glam::Vec3::new(0.3, 0.4, 0.5).normalize();
        let exact_up = reference_matcap_uv(normal, glam::Vec3::Y);
        let exact_down = reference_matcap_uv(normal, -glam::Vec3::Y);
        let near_up = reference_matcap_uv(normal, glam::Vec3::new(0.0, 1.0, 1.0e-6).normalize());
        let near_down = reference_matcap_uv(normal, glam::Vec3::new(0.0, -1.0, 1.0e-6).normalize());
        for uv in [exact_up, exact_down, near_up, near_down] {
            assert!(uv.is_finite());
        }
        assert_eq!(exact_up, reference_matcap_uv(normal, glam::Vec3::Y));
        assert_eq!(exact_down, reference_matcap_uv(normal, -glam::Vec3::Y));
        assert!((near_up - exact_up).length() < 1.0e-5);
        assert!((near_down - exact_down).length() < 1.0e-5);

        let shader = include_str!("shaders/mtoon.wgsl");
        assert!(shader.contains("var view_x = vec3f(1.0, 0.0, 0.0)"));
        assert!(shader.contains("if dot(horizontal, horizontal) > 1e-10"));
    }
}
