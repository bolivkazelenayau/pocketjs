//! Skinned/static model assets (glTF) and scene instances.

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::anim::{AnimState, Channel, ChannelPath, Clip, Interpolation, NodeTrs, Skeleton};
use crate::gpu::Gpu;
use crate::material::{
    GltfMagFilter, GltfMinFilter, GltfSampler, GltfWrapMode, MaterialAsset, MaterialInputs,
    MaterialModel, MaterialStateSet, MtoonMaterialDescriptor, PocketLitMaterial, RenderPhase,
    ScaledTextureInfo, TextureColorSpace, TextureInfo, TextureRole, TextureTransform,
    UnlitMaterial,
};
use crate::texture::{GpuTexture, Samplers, create_rgba_texture};

pub use crate::material::MaterialAlphaMode;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ModelVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub uv1: [f32; 2],
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}

impl ModelVertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ModelVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Uint32x4, 4 => Float32x4,
            5 => Float32x2
        ],
    };
}

/// Optional authored treatment of a material's final base color.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MaterialBaseColorMode {
    #[default]
    Authored,
    /// Convert the final RGB value to luminance while preserving alpha.
    Monochrome,
}

/// Replace the base-color texture of semantically tagged glTF materials.
///
/// Matching prefers `extras.pocket3d_role`; `name_prefix` is used only when
/// that role is absent. The texture and sampler are captured by the primitive
/// bind group, so callers may render new pixels into the same texture without
/// rebuilding the model.
pub struct MaterialTextureOverride<'a> {
    pub role: &'a str,
    pub name_prefix: Option<&'a str>,
    pub texture_view: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub expected_primitive_count: Option<usize>,
    pub force_white: bool,
    pub force_unlit: bool,
    pub force_opaque: bool,
    pub force_blend: bool,
    pub require_normalized_texcoord0: bool,
}

impl<'a> MaterialTextureOverride<'a> {
    pub fn new(
        role: &'a str,
        name_prefix: Option<&'a str>,
        texture_view: &'a wgpu::TextureView,
        sampler: &'a wgpu::Sampler,
    ) -> Self {
        Self {
            role,
            name_prefix,
            texture_view,
            sampler,
            expected_primitive_count: None,
            force_white: false,
            force_unlit: false,
            force_opaque: false,
            force_blend: false,
            require_normalized_texcoord0: false,
        }
    }

    pub fn expect_primitives(mut self, count: usize) -> Self {
        self.expected_primitive_count = Some(count);
        self
    }

    pub fn force_white(mut self) -> Self {
        self.force_white = true;
        self
    }

    pub fn force_unlit(mut self) -> Self {
        self.force_unlit = true;
        self
    }

    pub fn force_opaque(mut self) -> Self {
        self.force_opaque = true;
        self
    }

    /// Force alpha blending even when the authored glTF material is opaque.
    /// This is useful for replacing a cosmetic layer with a transparent
    /// texture without modifying the source mesh.
    pub fn force_blend(mut self) -> Self {
        self.force_blend = true;
        self
    }

    /// Reject matching primitives unless `TEXCOORD_0` is finite, normalized,
    /// and spans enough of each axis for a live 2D surface.
    pub fn require_normalized_texcoord0(mut self) -> Self {
        self.require_normalized_texcoord0 = true;
        self
    }
}

#[derive(Eq, Hash, PartialEq)]
struct ModelTextureCacheKey {
    width: u32,
    height: u32,
    rgba: Box<[u8]>,
    color_space: TextureColorSpace,
}

/// Explicit content-addressed cache for textures shared by multiple models.
///
/// Keys use the exact RGBA pixels after [`ModelLoadOptions`] downsampling plus
/// their dimensions, so identical images in independently cooked LOD files
/// share one GPU allocation without relying on image indices or names. The
/// cache retains those CPU pixels to make equality collision-free; callers
/// loading a fixed batch can drop it afterwards because every [`ModelAsset`]
/// keeps its shared [`GpuTexture`] allocations alive with [`Arc`].
#[derive(Default)]
pub struct ModelTextureCache {
    entries: HashMap<ModelTextureCacheKey, Arc<GpuTexture>>,
    hits: usize,
}

impl ModelTextureCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn hit_count(&self) -> usize {
        self.hits
    }

    fn get_or_upload(
        &mut self,
        gpu: &Gpu,
        label: &str,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        color_space: TextureColorSpace,
    ) -> Arc<GpuTexture> {
        let key = ModelTextureCacheKey {
            width,
            height,
            rgba: rgba.into_boxed_slice(),
            color_space,
        };
        match self.entries.entry(key) {
            Entry::Occupied(entry) => {
                self.hits += 1;
                entry.get().clone()
            }
            Entry::Vacant(entry) => {
                let key = entry.key();
                let texture = Arc::new(create_rgba_texture(
                    gpu,
                    label,
                    key.width,
                    key.height,
                    &key.rgba,
                    key.color_space == TextureColorSpace::Srgb,
                    true,
                ));
                entry.insert(texture.clone());
                texture
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialRaw {
    base_color_factor: [f32; 4],
    /// x: unlit, y: alpha cutoff (0 = off), z: double-sided, w: blend.
    params: [f32; 4],
    /// x: monochrome base color, y/z/w reserved.
    style: [f32; 4],
}

struct PrimitiveUpload {
    first_index: u32,
    index_count: u32,
    image: Option<usize>,
    base_color_factor: [f32; 4],
    alpha_mode: MaterialAlphaMode,
    alpha_cutoff: f32,
    double_sided: bool,
    unlit: bool,
    base_color_mode: MaterialBaseColorMode,
    material_name: Option<String>,
    material_role: Option<String>,
    texture_override: Option<usize>,
    material_index: Option<usize>,
}

pub struct Primitive {
    pub first_index: u32,
    pub index_count: u32,
    pub bind_group: wgpu::BindGroup,
    pub alpha_mode: MaterialAlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub unlit: bool,
    pub base_color_mode: MaterialBaseColorMode,
    pub material_name: Option<String>,
    pub material_role: Option<String>,
    pub material_index: Option<usize>,
    pub render_phase: RenderPhase,
    /// Kept alive explicitly alongside the bind group.
    #[allow(dead_code)]
    material_buf: wgpu::Buffer,
}

#[derive(Default, Clone, Copy)]
pub struct ModelLoadOptions {
    /// Halve textures until no side exceeds this (mip-friendly box filter).
    /// Character/prop authoring resolutions routinely exceed what small
    /// windows can display; this is the biggest single memory lever.
    pub max_texture_dim: Option<u32>,
}

pub struct Skin {
    /// Node index per joint.
    pub joints: Vec<usize>,
    pub inverse_bind: Vec<Mat4>,
}

/// One morph target of a primitive, stored sparse: only vertices the target
/// actually displaces. Deltas are in object space (bake transform applied).
pub struct MorphTargetData {
    /// (vertex index within the primitive, position delta).
    pub pos: Vec<(u32, Vec3)>,
    /// (vertex index within the primitive, normal delta).
    pub normal: Vec<(u32, Vec3)>,
}

/// A primitive that carries morph targets. Its base vertices are kept on the
/// CPU; morphed copies are written into a per-instance overlay buffer.
pub struct MorphPrim {
    /// Index into [`ModelAsset::primitives`].
    pub primitive: usize,
    /// First vertex of this primitive in the shared asset vertex buffer.
    pub vertex_base: u32,
    pub vertex_count: u32,
    /// First vertex of this primitive in a [`MorphState`] overlay buffer.
    pub overlay_offset: u32,
    base: Vec<ModelVertex>,
    pub targets: Vec<MorphTargetData>,
}

/// All morph-bearing primitives of one glTF mesh. Morph bindings address
/// targets as (glTF mesh index, target index), which maps here.
pub struct MorphMesh {
    /// glTF mesh index.
    pub mesh: usize,
    pub target_count: usize,
    pub prims: Vec<MorphPrim>,
}

/// Per-instance morph weights + the GPU overlay holding morphed vertices.
/// Weights are set by game code; the overlay upload happens lazily during
/// render prepare, and costs nothing on frames where no weight changed.
pub struct MorphState {
    /// Parallel to [`ModelAsset::morph_meshes`]; one weight per target.
    weights: Vec<Vec<f32>>,
    /// Bitmask of morph meshes needing recompute (interior mutability so the
    /// renderer can flush during prepare without a `&mut Scene`).
    dirty: Cell<u64>,
    buffer: wgpu::Buffer,
}

impl MorphState {
    /// `mesh_slot` indexes [`ModelAsset::morph_meshes`].
    pub fn set_weight(&mut self, mesh_slot: usize, target: usize, weight: f32) {
        let Some(w) = self
            .weights
            .get_mut(mesh_slot)
            .and_then(|m| m.get_mut(target))
        else {
            return;
        };
        if *w != weight {
            *w = weight;
            let bit = if mesh_slot < 64 {
                1u64 << mesh_slot
            } else {
                u64::MAX
            };
            self.dirty.set(self.dirty.get() | bit);
        }
    }

    pub fn weight(&self, mesh_slot: usize, target: usize) -> f32 {
        self.weights
            .get(mesh_slot)
            .and_then(|m| m.get(target))
            .copied()
            .unwrap_or(0.0)
    }

    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Recompute + upload overlay vertices for meshes whose weights changed.
    pub(crate) fn upload_if_dirty(&self, gpu: &Gpu, asset: &ModelAsset) {
        let dirty = self.dirty.replace(0);
        if dirty == 0 {
            return;
        }
        for (mi, mesh) in asset.morph_meshes.iter().enumerate() {
            if mi < 64 && dirty & (1u64 << mi) == 0 {
                continue;
            }
            let weights = &self.weights[mi];
            for prim in &mesh.prims {
                let mut verts = prim.base.clone();
                for (ti, target) in prim.targets.iter().enumerate() {
                    let w = weights.get(ti).copied().unwrap_or(0.0);
                    if w.abs() < 1e-4 {
                        continue;
                    }
                    for &(vi, d) in &target.pos {
                        let v = &mut verts[vi as usize];
                        v.pos = (Vec3::from(v.pos) + d * w).to_array();
                    }
                    for &(vi, d) in &target.normal {
                        let v = &mut verts[vi as usize];
                        v.normal = (Vec3::from(v.normal) + d * w).to_array();
                    }
                }
                gpu.queue.write_buffer(
                    &self.buffer,
                    prim.overlay_offset as u64 * std::mem::size_of::<ModelVertex>() as u64,
                    bytemuck::cast_slice(&verts),
                );
            }
        }
    }
}

pub struct ModelAsset {
    pub vbuf: wgpu::Buffer,
    pub ibuf: wgpu::Buffer,
    pub primitives: Vec<Primitive>,
    /// Immutable authored materials, indexed exactly by glTF material index.
    materials: Vec<MaterialAsset>,
    pub skeleton: Skeleton,
    /// glTF node names indexed exactly like [`Self::skeleton`]. Procedural
    /// assets have no nodes, so this is empty for `from_geometry*` models.
    node_names: Vec<Option<String>>,
    /// All skins in the file. The joint palette concatenates them in order;
    /// vertex joint indices were remapped at load to address the combined
    /// palette (multi-skin characters: body + visor etc.).
    pub skins: Vec<Skin>,
    pub clips: Vec<Clip>,
    /// Meshes carrying morph targets (facial blend shapes etc.).
    pub morph_meshes: Vec<MorphMesh>,
    /// Parallel to `primitives`: (morph mesh slot, prim slot) when morphing.
    pub prim_morph: Vec<Option<(usize, usize)>>,
    /// Rest-pose bounds (object space; skinned primitives measured through
    /// their rest-pose joint matrices, so units/orientation baked into the
    /// rig — cm exports, Z-up meshes — are already resolved).
    pub aabb: (Vec3, Vec3),
    #[allow(dead_code)]
    textures: Vec<Arc<GpuTexture>>,
}

fn make_material_bind_group(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    material: &MaterialRaw,
) -> (wgpu::BindGroup, wgpu::Buffer) {
    use wgpu::util::DeviceExt;

    let material_buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} params")),
            contents: bytemuck::bytes_of(material),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: material_buf.as_entire_binding(),
            },
        ],
    });
    (bind_group, material_buf)
}

fn pocket3d_material_role(material: &gltf::Material<'_>) -> Option<String> {
    let raw = material.extras().as_ref()?.get();
    pocket3d_role_from_extras(raw)
}

fn pocket3d_material_base_color_mode(material: &gltf::Material<'_>) -> MaterialBaseColorMode {
    let Some(raw) = material.extras().as_ref().map(|extras| extras.get()) else {
        return MaterialBaseColorMode::Authored;
    };
    pocket3d_base_color_mode_from_extras(raw)
}

fn pocket3d_role_from_extras(raw: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(raw)
        .ok()?
        .get("pocket3d_role")?
        .as_str()
        .map(str::to_owned)
}

fn pocket3d_base_color_mode_from_extras(raw: &str) -> MaterialBaseColorMode {
    match serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|extras| {
            extras
                .get("pocket3d_base_color_mode")?
                .as_str()
                .map(str::to_owned)
        })
        .as_deref()
    {
        Some("monochrome") => MaterialBaseColorMode::Monochrome,
        _ => MaterialBaseColorMode::Authored,
    }
}

fn semantic_material_matches(
    material_role: Option<&str>,
    material_name: Option<&str>,
    role: &str,
    name_prefix: Option<&str>,
) -> bool {
    match material_role {
        Some(material_role) => material_role == role,
        None => name_prefix
            .zip(material_name)
            .is_some_and(|(prefix, name)| name.starts_with(prefix)),
    }
}

fn gltf_sampler(sampler: gltf::texture::Sampler<'_>) -> GltfSampler {
    GltfSampler {
        index: sampler.index(),
        mag_filter: sampler.mag_filter().map(|filter| match filter {
            gltf::texture::MagFilter::Nearest => GltfMagFilter::Nearest,
            gltf::texture::MagFilter::Linear => GltfMagFilter::Linear,
        }),
        min_filter: sampler.min_filter().map(|filter| match filter {
            gltf::texture::MinFilter::Nearest => GltfMinFilter::Nearest,
            gltf::texture::MinFilter::Linear => GltfMinFilter::Linear,
            gltf::texture::MinFilter::NearestMipmapNearest => GltfMinFilter::NearestMipmapNearest,
            gltf::texture::MinFilter::LinearMipmapNearest => GltfMinFilter::LinearMipmapNearest,
            gltf::texture::MinFilter::NearestMipmapLinear => GltfMinFilter::NearestMipmapLinear,
            gltf::texture::MinFilter::LinearMipmapLinear => GltfMinFilter::LinearMipmapLinear,
        }),
        wrap_s: match sampler.wrap_s() {
            gltf::texture::WrappingMode::ClampToEdge => GltfWrapMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => GltfWrapMode::MirroredRepeat,
            gltf::texture::WrappingMode::Repeat => GltfWrapMode::Repeat,
        },
        wrap_t: match sampler.wrap_t() {
            gltf::texture::WrappingMode::ClampToEdge => GltfWrapMode::ClampToEdge,
            gltf::texture::WrappingMode::MirroredRepeat => GltfWrapMode::MirroredRepeat,
            gltf::texture::WrappingMode::Repeat => GltfWrapMode::Repeat,
        },
    }
}

fn gltf_texture_info(info: gltf::texture::Info<'_>, role: TextureRole) -> TextureInfo {
    let texture = info.texture();
    let transform = info
        .texture_transform()
        .map(|transform| TextureTransform {
            offset: transform.offset(),
            rotation: transform.rotation(),
            scale: transform.scale(),
            tex_coord_override: transform.tex_coord(),
        })
        .unwrap_or_default();
    TextureInfo {
        texture_index: texture.index(),
        image_index: texture.source().index(),
        sampler: gltf_sampler(texture.sampler()),
        tex_coord: info.tex_coord(),
        transform,
        role,
        color_space: role.color_space(),
    }
}

fn gltf_material_inputs(material: &gltf::Material<'_>) -> MaterialInputs {
    let pbr = material.pbr_metallic_roughness();
    MaterialInputs {
        base_color_factor: pbr.base_color_factor(),
        base_color_texture: pbr
            .base_color_texture()
            .map(|info| gltf_texture_info(info, TextureRole::BaseColor)),
        normal_texture: material.normal_texture().map(|info| {
            let texture = info.texture();
            ScaledTextureInfo {
                texture: TextureInfo {
                    texture_index: texture.index(),
                    image_index: texture.source().index(),
                    sampler: gltf_sampler(texture.sampler()),
                    tex_coord: info.tex_coord(),
                    // gltf-rs does not expose KHR_texture_transform from the
                    // specialized normalTexture wrapper. MToon descriptors
                    // arrive from PocketVRM with the complete transform.
                    transform: TextureTransform::default(),
                    role: TextureRole::Normal,
                    color_space: TextureColorSpace::Linear,
                },
                scale: info.scale(),
            }
        }),
        emissive_factor: material.emissive_factor(),
        emissive_texture: material
            .emissive_texture()
            .map(|info| gltf_texture_info(info, TextureRole::Emissive)),
        alpha_mode: MaterialAlphaMode::from_gltf(material.alpha_mode()),
        alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
        double_sided: material.double_sided(),
    }
}

fn authored_materials(
    doc: &gltf::Document,
    mtoon_descriptors: &[MtoonMaterialDescriptor],
    path: &Path,
) -> Result<Vec<MaterialAsset>> {
    let material_count = doc.materials().count();
    let texture_count = doc.textures().count();
    let image_count = doc.images().count();
    let mut mtoon_by_index = HashMap::new();
    for descriptor in mtoon_descriptors {
        if descriptor.material_index >= material_count {
            bail!(
                "MToon descriptor material index {} exceeds material count {material_count} in {}",
                descriptor.material_index,
                path.display()
            );
        }
        if mtoon_by_index
            .insert(descriptor.material_index, descriptor)
            .is_some()
        {
            bail!(
                "duplicate MToon descriptor for material {} in {}",
                descriptor.material_index,
                path.display()
            );
        }
        for texture in descriptor_textures(descriptor) {
            if texture.texture_index >= texture_count || texture.image_index >= image_count {
                bail!(
                    "MToon material {} texture {:?} references texture {}/image {} outside glTF counts {texture_count}/{image_count} in {}",
                    descriptor.material_index,
                    texture.role,
                    texture.texture_index,
                    texture.image_index,
                    path.display()
                );
            }
            let tex_coord = texture.effective_tex_coord();
            if tex_coord > 1 {
                bail!(
                    "MToon material {} texture {:?} selects TEXCOORD_{tex_coord}, but Pocket3D currently imports only TEXCOORD_0 and TEXCOORD_1 in {}",
                    descriptor.material_index,
                    texture.role,
                    path.display()
                );
            }
        }
    }

    doc.materials()
        .map(|material| {
            let index = material
                .index()
                .expect("document material iterator always has an index");
            let name = material.name().map(str::to_owned);
            if let Some(descriptor) = mtoon_by_index.get(&index) {
                if !material.unlit() {
                    bail!(
                        "MToon material {index} lacks KHR_materials_unlit fallback in {}",
                        path.display()
                    );
                }
                if let Some(base) = descriptor.inputs.base_color_texture.as_ref() {
                    let selected = base.effective_tex_coord();
                    if !base.current_base_color_fallback_uv_supported() {
                        bail!(
                            "MToon material {index} baseColorTexture selects TEXCOORD_{selected}, but the current unlit fallback samples only TEXCOORD_0 in {}",
                            path.display()
                        );
                    }
                }
                Ok(MaterialAsset {
                    gltf_material_index: index,
                    name,
                    inputs: descriptor.inputs.clone(),
                    model: MaterialModel::Mtoon(Box::new(descriptor.mtoon.clone())),
                })
            } else {
                let inputs = gltf_material_inputs(&material);
                if let Some(base) = inputs.base_color_texture.as_ref() {
                    let selected = base.effective_tex_coord();
                    if !base.current_base_color_fallback_uv_supported() {
                        bail!(
                            "glTF material {index} baseColorTexture selects TEXCOORD_{selected}, but the current PocketLit/Unlit shader samples only TEXCOORD_0 in {}",
                            path.display()
                        );
                    }
                }
                Ok(MaterialAsset {
                    gltf_material_index: index,
                    name,
                    inputs,
                    model: if material.unlit() {
                        MaterialModel::Unlit(UnlitMaterial)
                    } else {
                        MaterialModel::PocketLit(PocketLitMaterial)
                    },
                })
            }
        })
        .collect()
}

fn descriptor_textures(descriptor: &MtoonMaterialDescriptor) -> Vec<&TextureInfo> {
    let mut textures = Vec::new();
    if let Some(texture) = descriptor.inputs.base_color_texture.as_ref() {
        textures.push(texture);
    }
    if let Some(texture) = descriptor.inputs.normal_texture.as_ref() {
        textures.push(&texture.texture);
    }
    if let Some(texture) = descriptor.inputs.emissive_texture.as_ref() {
        textures.push(texture);
    }
    textures.extend(
        [
            descriptor.mtoon.shade_multiply_texture.as_ref(),
            descriptor
                .mtoon
                .shading_shift_texture
                .as_ref()
                .map(|value| &value.texture),
            descriptor.mtoon.matcap_texture.as_ref(),
            descriptor.mtoon.rim_multiply_texture.as_ref(),
            descriptor.mtoon.outline_width_multiply_texture.as_ref(),
            descriptor.mtoon.uv_animation_mask_texture.as_ref(),
        ]
        .into_iter()
        .flatten(),
    );
    textures
}

fn validate_normalized_texcoord0(
    texcoords: Option<&[[f32; 2]]>,
    vertex_count: usize,
    material_name: Option<&str>,
    path: &Path,
) -> Result<()> {
    let label = material_name.unwrap_or("<unnamed>");
    let Some(texcoords) = texcoords else {
        bail!(
            "material {label:?} in {} requires TEXCOORD_0, but it is missing",
            path.display()
        );
    };
    if texcoords.len() != vertex_count {
        bail!(
            "material {label:?} in {} has {} TEXCOORD_0 values for {vertex_count} vertices",
            path.display(),
            texcoords.len()
        );
    }

    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for uv in texcoords {
        if !uv[0].is_finite() || !uv[1].is_finite() {
            bail!(
                "material {label:?} in {} has non-finite TEXCOORD_0 values",
                path.display()
            );
        }
        for axis in 0..2 {
            min[axis] = min[axis].min(uv[axis]);
            max[axis] = max[axis].max(uv[axis]);
        }
    }

    const NORMALIZED_TOLERANCE: f32 = 0.01;
    if min.iter().any(|&value| value < -NORMALIZED_TOLERANCE)
        || max.iter().any(|&value| value > 1.0 + NORMALIZED_TOLERANCE)
    {
        bail!(
            "material {label:?} in {} has non-normalized TEXCOORD_0 bounds {:?}..{:?}",
            path.display(),
            min,
            max
        );
    }

    const MIN_AXIS_SPAN: f32 = 0.5;
    let span = [max[0] - min[0], max[1] - min[1]];
    if span[0] < MIN_AXIS_SPAN || span[1] < MIN_AXIS_SPAN {
        bail!(
            "material {label:?} in {} has insufficient TEXCOORD_0 span {:?}; each axis must span at least {MIN_AXIS_SPAN}",
            path.display(),
            span
        );
    }
    Ok(())
}

impl ModelAsset {
    /// Authored material data is shared by instances and cannot be mutated.
    pub fn materials(&self) -> &[MaterialAsset] {
        &self.materials
    }

    /// Return the glTF node index for `name`.
    ///
    /// Node indices address [`Self::skeleton`] and can be sampled with
    /// [`Self::sampled_node_transform`]. If a file contains duplicate names,
    /// the first node in glTF index order wins.
    pub fn node_named(&self, name: &str) -> Option<usize> {
        find_node_named(&self.node_names, name)
    }

    /// Sample one node's object-space global transform for `anim`.
    ///
    /// The returned matrix includes the animated transforms of all node
    /// ancestors. It does not include a [`ModelInstance`] transform or any skin
    /// inverse-bind matrix, so callers can compose it directly with an
    /// instance transform when placing a held object or other socket content.
    /// Invalid node indices return `None`.
    pub fn sampled_node_transform(&self, node: usize, anim: &AnimState) -> Option<Mat4> {
        sample_node_transform(&self.skeleton, &self.clips, node, anim)
    }

    pub fn clip_named(&self, name: &str) -> Option<usize> {
        self.clips.iter().position(|c| c.name == name)
    }

    pub fn height(&self) -> f32 {
        (self.aabb.1.y - self.aabb.0.y).max(0.001)
    }

    /// Joint palette for the given animation state (identity if no skin).
    /// Skins concatenate in file order, matching the load-time joint remap.
    pub fn joint_palette(&self, anim: &AnimState, out: &mut Vec<Mat4>) {
        let mut globals = Vec::new();
        let clip = self.clips.get(anim.clip);
        self.skeleton
            .global_transforms(clip, anim.time, anim.looping, &mut globals);
        self.palette_from_globals(&globals, out);
    }

    /// Joint palette from externally computed node globals (procedural poses:
    /// look-at, physics bones). Same skin concatenation as `joint_palette`.
    pub fn palette_from_globals(&self, globals: &[Mat4], out: &mut Vec<Mat4>) {
        out.clear();
        for skin in &self.skins {
            for (i, &node) in skin.joints.iter().enumerate() {
                out.push(globals[node] * skin.inverse_bind[i]);
            }
        }
        if out.is_empty() {
            out.push(Mat4::IDENTITY);
        }
    }

    /// Which slot in `morph_meshes` a glTF mesh landed in.
    pub fn morph_mesh_slot(&self, gltf_mesh: usize) -> Option<usize> {
        self.morph_meshes.iter().position(|m| m.mesh == gltf_mesh)
    }

    /// Create per-instance morph state (overlay buffer starts at the rest
    /// shape). `None` when the asset has no morph targets.
    pub fn create_morph_state(&self, gpu: &Gpu) -> Option<MorphState> {
        use wgpu::util::DeviceExt;
        if self.morph_meshes.is_empty() {
            return None;
        }
        let mut init: Vec<ModelVertex> = Vec::new();
        for mesh in &self.morph_meshes {
            for prim in &mesh.prims {
                init.extend_from_slice(&prim.base);
            }
        }
        let buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("morph overlay"),
                contents: bytemuck::cast_slice(&init),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        Some(MorphState {
            weights: self
                .morph_meshes
                .iter()
                .map(|m| vec![0.0; m.target_count])
                .collect(),
            dirty: Cell::new(0),
            buffer,
        })
    }

    /// The material bind group layout for model primitives (group 1).
    pub fn material_layout(gpu: &Gpu) -> wgpu::BindGroupLayout {
        gpu.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("model material"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<MaterialRaw>() as u64,
                            ),
                        },
                        count: None,
                    },
                ],
            })
    }

    /// Build an asset from raw geometry (procedural models). `image` is an
    /// optional RGBA8 texture; omit it for a plain white surface.
    pub fn from_geometry(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        label: &str,
        vertices: &[ModelVertex],
        indices: &[u32],
        image: Option<(u32, u32, &[u8])>,
    ) -> Arc<Self> {
        use wgpu::util::DeviceExt;
        let (tw, th, tpx): (u32, u32, &[u8]) = image.unwrap_or((1, 1, &[255u8, 255, 255, 255]));
        let tex = Arc::new(create_rgba_texture(gpu, label, tw, th, tpx, true, false));
        let material = MaterialRaw {
            base_color_factor: [1.0; 4],
            params: [0.0; 4],
            style: [0.0; 4],
        };
        let (bind_group, material_buf) = make_material_bind_group(
            gpu,
            layout,
            label,
            &tex.view,
            &samplers.aniso_repeat,
            &material,
        );
        let vbuf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ibuf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let mut aabb = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in vertices {
            let p = Vec3::from(v.pos);
            aabb.0 = aabb.0.min(p);
            aabb.1 = aabb.1.max(p);
        }
        Arc::new(Self {
            vbuf,
            ibuf,
            primitives: vec![Primitive {
                first_index: 0,
                index_count: indices.len() as u32,
                bind_group,
                alpha_mode: MaterialAlphaMode::Opaque,
                alpha_cutoff: 0.0,
                double_sided: false,
                unlit: false,
                base_color_mode: MaterialBaseColorMode::Authored,
                material_name: Some(label.to_owned()),
                material_role: None,
                material_index: Some(0),
                render_phase: RenderPhase::Opaque,
                material_buf,
            }],
            materials: vec![MaterialAsset::pocket_lit(0, Some(label.to_owned()))],
            skeleton: Skeleton {
                parents: Vec::new(),
                rest: Vec::new(),
                order: Vec::new(),
            },
            node_names: Vec::new(),
            skins: Vec::new(),
            clips: Vec::new(),
            morph_meshes: Vec::new(),
            prim_morph: vec![None],
            aabb,
            textures: vec![tex],
        })
    }

    /// Build an asset from raw geometry whose material samples an external
    /// texture view (an [`crate::gpu::OffscreenTarget`], a video frame, …).
    /// The bind group keeps the underlying texture alive; render into it
    /// each frame and every instance of this asset shows the update — this
    /// is how a live 2D surface lands on a 3D mesh (pocket-widget screens).
    pub fn from_geometry_textured(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        label: &str,
        vertices: &[ModelVertex],
        indices: &[u32],
        view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> Arc<Self> {
        use wgpu::util::DeviceExt;
        let material = MaterialRaw {
            base_color_factor: [1.0; 4],
            params: [0.0; 4],
            style: [0.0; 4],
        };
        let (bind_group, material_buf) =
            make_material_bind_group(gpu, layout, label, view, sampler, &material);
        let vbuf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ibuf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let mut aabb = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in vertices {
            let p = Vec3::from(v.pos);
            aabb.0 = aabb.0.min(p);
            aabb.1 = aabb.1.max(p);
        }
        Arc::new(Self {
            vbuf,
            ibuf,
            primitives: vec![Primitive {
                first_index: 0,
                index_count: indices.len() as u32,
                bind_group,
                alpha_mode: MaterialAlphaMode::Opaque,
                alpha_cutoff: 0.0,
                double_sided: false,
                unlit: false,
                base_color_mode: MaterialBaseColorMode::Authored,
                material_name: Some(label.to_owned()),
                material_role: None,
                material_index: Some(0),
                render_phase: RenderPhase::Opaque,
                material_buf,
            }],
            materials: vec![MaterialAsset::pocket_lit(0, Some(label.to_owned()))],
            skeleton: Skeleton {
                parents: Vec::new(),
                rest: Vec::new(),
                order: Vec::new(),
            },
            node_names: Vec::new(),
            skins: Vec::new(),
            clips: Vec::new(),
            morph_meshes: Vec::new(),
            prim_morph: vec![None],
            aabb,
            textures: Vec::new(),
        })
    }

    pub fn load_glb(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        path: &Path,
    ) -> Result<Arc<Self>> {
        Self::load_glb_opts(gpu, layout, samplers, path, &ModelLoadOptions::default())
    }

    /// Load a self-contained binary glTF model directly from memory.
    ///
    /// `label` is used for diagnostics. External buffer or image URIs are not
    /// supported by this entry point; embed those resources in the GLB.
    pub fn load_glb_bytes(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        bytes: &[u8],
        label: &str,
    ) -> Result<Arc<Self>> {
        Self::load_glb_bytes_opts(
            gpu,
            layout,
            samplers,
            bytes,
            label,
            &ModelLoadOptions::default(),
        )
    }

    /// Load a self-contained binary glTF model from memory with the normal
    /// model load options. The bytes are imported exactly once; no filesystem
    /// lookup is performed by this entry point.
    pub fn load_glb_bytes_opts(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        bytes: &[u8],
        label: &str,
        opts: &ModelLoadOptions,
    ) -> Result<Arc<Self>> {
        Self::load_glb_bytes_opts_with_allowed_required_extensions(
            gpu,
            layout,
            samplers,
            bytes,
            label,
            opts,
            std::iter::empty::<&str>(),
        )
    }

    /// `load_glb_bytes_opts` with an explicit allowlist of required
    /// extensions already validated by the caller. This does not disable
    /// normal glTF validation and does not affect path-based loading.
    pub fn load_glb_bytes_opts_with_allowed_required_extensions<I, S>(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        bytes: &[u8],
        label: &str,
        opts: &ModelLoadOptions,
        allowed_required_extensions: I,
    ) -> Result<Arc<Self>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::load_glb_bytes_opts_with_material_descriptors(
            gpu,
            layout,
            samplers,
            bytes,
            label,
            opts,
            allowed_required_extensions,
            &[],
        )
    }

    /// Byte loading plus a typed, caller-validated MToon semantic handoff.
    /// Descriptors are stored as authored materials, while rendering remains
    /// on the glTF `KHR_materials_unlit` fallback until native MToon lands.
    #[allow(clippy::too_many_arguments)]
    pub fn load_glb_bytes_opts_with_material_descriptors<I, S>(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        bytes: &[u8],
        label: &str,
        opts: &ModelLoadOptions,
        allowed_required_extensions: I,
        mtoon_descriptors: &[MtoonMaterialDescriptor],
    ) -> Result<Arc<Self>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        validate_load_options(opts)?;
        let allowed_required_extensions: Vec<String> = allowed_required_extensions
            .into_iter()
            .map(|extension| extension.as_ref().to_owned())
            .collect();
        let imported = import_glb_slice_with_options(
            bytes,
            label,
            &allowed_required_extensions,
            opts.max_texture_dim,
        )?;
        let mut cache = ModelTextureCache::new();
        Self::load_glb_imported(
            gpu,
            layout,
            samplers,
            Path::new(label),
            opts,
            &[],
            &mut cache,
            mtoon_descriptors,
            imported,
        )
    }

    pub fn load_glb_opts(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        path: &Path,
        opts: &ModelLoadOptions,
    ) -> Result<Arc<Self>> {
        Self::load_glb_opts_with_overrides(gpu, layout, samplers, path, opts, &[])
    }

    /// Load a glTF model while sharing imported textures through `cache`.
    /// The cache must only be used with the [`Gpu`] that created it.
    pub fn load_glb_with_cache(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        path: &Path,
        cache: &mut ModelTextureCache,
    ) -> Result<Arc<Self>> {
        Self::load_glb_opts_with_cache(
            gpu,
            layout,
            samplers,
            path,
            &ModelLoadOptions::default(),
            cache,
        )
    }

    /// `load_glb_with_cache` plus the normal model load options.
    pub fn load_glb_opts_with_cache(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        path: &Path,
        opts: &ModelLoadOptions,
        cache: &mut ModelTextureCache,
    ) -> Result<Arc<Self>> {
        Self::load_glb_opts_with_overrides_and_cache(gpu, layout, samplers, path, opts, &[], cache)
    }

    /// Load a glTF model and replace selected semantic materials with external
    /// texture views. This is intended for live surfaces such as a handheld
    /// screen; existing `load_glb*` entry points remain equivalent to passing
    /// an empty override slice.
    pub fn load_glb_with_overrides(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        path: &Path,
        overrides: &[MaterialTextureOverride<'_>],
    ) -> Result<Arc<Self>> {
        Self::load_glb_opts_with_overrides(
            gpu,
            layout,
            samplers,
            path,
            &ModelLoadOptions::default(),
            overrides,
        )
    }

    /// `load_glb_with_overrides` plus the normal model load options.
    pub fn load_glb_opts_with_overrides(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        path: &Path,
        opts: &ModelLoadOptions,
        overrides: &[MaterialTextureOverride<'_>],
    ) -> Result<Arc<Self>> {
        let mut cache = ModelTextureCache::new();
        Self::load_glb_opts_with_overrides_and_cache(
            gpu, layout, samplers, path, opts, overrides, &mut cache,
        )
    }

    /// `load_glb_with_overrides` with an explicit texture cache shared across
    /// independently loaded assets.
    pub fn load_glb_with_overrides_and_cache(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        path: &Path,
        overrides: &[MaterialTextureOverride<'_>],
        cache: &mut ModelTextureCache,
    ) -> Result<Arc<Self>> {
        Self::load_glb_opts_with_overrides_and_cache(
            gpu,
            layout,
            samplers,
            path,
            &ModelLoadOptions::default(),
            overrides,
            cache,
        )
    }

    /// Fully configurable glTF load with semantic material overrides and an
    /// explicit content-addressed texture cache.
    pub fn load_glb_opts_with_overrides_and_cache(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        path: &Path,
        opts: &ModelLoadOptions,
        overrides: &[MaterialTextureOverride<'_>],
        cache: &mut ModelTextureCache,
    ) -> Result<Arc<Self>> {
        validate_load_options(opts)?;
        for material_override in overrides {
            if material_override.force_opaque && material_override.force_blend {
                bail!(
                    "material override role {:?} cannot force both opaque and blend",
                    material_override.role
                );
            }
        }
        let imported =
            gltf::import(path).with_context(|| format!("importing {}", path.display()))?;

        Self::load_glb_imported(
            gpu,
            layout,
            samplers,
            path,
            opts,
            overrides,
            cache,
            &[],
            imported,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_glb_imported(
        gpu: &Gpu,
        layout: &wgpu::BindGroupLayout,
        samplers: &Samplers,
        path: &Path,
        opts: &ModelLoadOptions,
        overrides: &[MaterialTextureOverride<'_>],
        cache: &mut ModelTextureCache,
        mtoon_descriptors: &[MtoonMaterialDescriptor],
        imported: ImportedGltf,
    ) -> Result<Arc<Self>> {
        let (doc, buffers, images) = imported;
        validate_model_input(&doc, &buffers, path)?;
        let materials = authored_materials(&doc, mtoon_descriptors, path)?;

        // --- textures ------------------------------------------------------
        // Only upload images a material actually samples (some files carry
        // thumbnails and utility maps), and optionally cap texture size —
        // authoring resolutions (4096²) dwarf what a small widget window can
        // ever show, and GPU memory is the dominant cost of a character.
        let used: std::collections::HashSet<usize> = doc
            .materials()
            .filter(|material| {
                let role = pocket3d_material_role(material);
                !overrides.iter().any(|candidate| {
                    semantic_material_matches(
                        role.as_deref(),
                        material.name(),
                        candidate.role,
                        candidate.name_prefix,
                    )
                })
            })
            .filter_map(|m| {
                m.pbr_metallic_roughness()
                    .base_color_texture()
                    .map(|t| t.texture().source().index())
            })
            .collect();
        // Unused image slots and primitives without a base-color image all
        // share this fallback instead of allocating one dummy texture each.
        let white = Arc::new(create_rgba_texture(
            gpu,
            "white",
            1,
            1,
            &[255, 255, 255, 255],
            true,
            false,
        ));
        let mut textures = Vec::with_capacity(images.len() + 1);
        for (i, img) in images.iter().enumerate() {
            if !used.contains(&i) {
                textures.push(white.clone());
                continue;
            }
            let (rgba, w, h) =
                cap_texture_rgba(to_rgba8(img), img.width, img.height, opts.max_texture_dim);
            textures.push(cache.get_or_upload(
                gpu,
                &format!("model img {i}"),
                w,
                h,
                rgba,
                TextureColorSpace::Srgb,
            ));
        }

        // --- nodes / skeleton ----------------------------------------------
        let node_count = doc.nodes().count();
        let mut parents = vec![usize::MAX; node_count];
        let mut rest = vec![NodeTrs::IDENTITY; node_count];
        let mut node_names = vec![None; node_count];
        for node in doc.nodes() {
            let (t, r, s) = node.transform().decomposed();
            node_names[node.index()] = node.name().map(str::to_owned);
            rest[node.index()] = NodeTrs {
                translation: Vec3::from(t),
                rotation: glam::Quat::from_array(r),
                scale: Vec3::from(s),
            };
            for child in node.children() {
                parents[child.index()] = node.index();
            }
        }
        // Parents-first order via DFS from roots.
        let mut order = Vec::with_capacity(node_count);
        let mut stack: Vec<usize> = (0..node_count)
            .filter(|&i| parents[i] == usize::MAX)
            .collect();
        stack.reverse();
        let children_of: Vec<Vec<usize>> = doc
            .nodes()
            .map(|n| n.children().map(|c| c.index()).collect())
            .collect();
        while let Some(i) = stack.pop() {
            order.push(i);
            for &c in children_of[i].iter().rev() {
                stack.push(c);
            }
        }
        let skeleton = Skeleton {
            parents,
            rest,
            order,
        };

        // --- skins -----------------------------------------------------------
        // All of them: multi-skin characters (body + visor) address one
        // concatenated joint palette, so each skin gets a base offset and the
        // vertex joint indices are remapped below.
        let skins: Vec<Skin> = doc
            .skins()
            .map(|s| -> Result<Skin> {
                let reader = s.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
                let joints: Vec<usize> = s.joints().map(|j| j.index()).collect();
                let inverse_bind = match reader.read_inverse_bind_matrices() {
                    Some(it) => {
                        let inverse_bind: Vec<_> =
                            it.map(|m| Mat4::from_cols_array_2d(&m)).collect();
                        if inverse_bind.len() != joints.len() {
                            bail!(
                                "inverse bind matrix count {} does not match skin joint count {} in {}",
                                inverse_bind.len(),
                                joints.len(),
                                path.display()
                            );
                        }
                        inverse_bind
                    }
                    None => vec![Mat4::IDENTITY; joints.len()],
                };
                Ok(Skin {
                    joints,
                    inverse_bind,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let skin_base: Vec<u32> = skins
            .iter()
            .scan(0u32, |acc, s| {
                let base = *acc;
                *acc += s.joints.len() as u32;
                Some(base)
            })
            .collect();

        // --- meshes ----------------------------------------------------------
        // Global transforms of the rest pose, to bake node placement into
        // non-skinned primitives.
        let mut rest_globals = Vec::new();
        skeleton.global_transforms(None, 0.0, false, &mut rest_globals);

        let mut vertices: Vec<ModelVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut primitives_meta: Vec<PrimitiveUpload> = Vec::new();
        let mut override_counts = vec![0usize; overrides.len()];
        let mut aabb = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        let mut morph_meshes: Vec<MorphMesh> = Vec::new();
        let mut prim_morph: Vec<Option<(usize, usize)>> = Vec::new();
        let mut overlay_verts: u32 = 0;

        for node in doc.nodes() {
            let Some(mesh) = node.mesh() else { continue };
            let node_skin = node.skin().map(|s| s.index()).filter(|&i| i < skins.len());
            let bake = if node_skin.is_some() {
                Mat4::IDENTITY
            } else {
                rest_globals[node.index()]
            };
            let normal_bake = bake.inverse().transpose();
            // Rest-pose joint matrices for measuring skinned bounds: raw
            // vertices can live in an arbitrary rig space (cm, Z-up); only
            // the skinned result is in object space.
            let rest_palette: Vec<Mat4> = node_skin
                .map(|si| {
                    let s = &skins[si];
                    s.joints
                        .iter()
                        .enumerate()
                        .map(|(i, &n)| rest_globals[n] * s.inverse_bind[i])
                        .collect()
                })
                .unwrap_or_default();

            for prim in mesh.primitives() {
                let reader = prim.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
                let Some(pos_iter) = reader.read_positions() else {
                    continue;
                };
                let base = vertices.len() as u32;

                let positions: Vec<[f32; 3]> = pos_iter.collect();
                let normals: Vec<[f32; 3]> = match reader.read_normals() {
                    Some(it) => {
                        let normals: Vec<_> = it.collect();
                        validate_attribute_count("NORMAL", positions.len(), normals.len(), path)?;
                        normals
                    }
                    None => vec![[0.0, 1.0, 0.0]; positions.len()],
                };
                let texcoord0: Option<Vec<[f32; 2]>> = match reader.read_tex_coords(0) {
                    Some(it) => {
                        let texcoords: Vec<_> = it.into_f32().collect();
                        validate_attribute_count(
                            "TEXCOORD_0",
                            positions.len(),
                            texcoords.len(),
                            path,
                        )?;
                        Some(texcoords)
                    }
                    None => None,
                };
                let texcoord1: Option<Vec<[f32; 2]>> = match reader.read_tex_coords(1) {
                    Some(it) => {
                        let texcoords: Vec<_> = it.into_f32().collect();
                        validate_attribute_count(
                            "TEXCOORD_1",
                            positions.len(),
                            texcoords.len(),
                            path,
                        )?;
                        Some(texcoords)
                    }
                    None => None,
                };
                let joints: Vec<[u16; 4]> = match reader.read_joints(0) {
                    Some(it) => {
                        let joints: Vec<_> = it.into_u16().collect();
                        validate_attribute_count("JOINTS_0", positions.len(), joints.len(), path)?;
                        joints
                    }
                    None => vec![[0, 0, 0, 0]; positions.len()],
                };
                let weights: Vec<[f32; 4]> = match reader.read_weights(0) {
                    Some(it) => {
                        let weights: Vec<_> = it.into_f32().collect();
                        validate_attribute_count(
                            "WEIGHTS_0",
                            positions.len(),
                            weights.len(),
                            path,
                        )?;
                        weights
                    }
                    None => vec![[1.0, 0.0, 0.0, 0.0]; positions.len()],
                };

                if let Some(si) = node_skin {
                    let joint_count = skins[si].joints.len();
                    for (vertex, joints) in joints.iter().enumerate() {
                        for (slot, &joint) in joints.iter().enumerate() {
                            if joint as usize >= joint_count {
                                bail!(
                                    "JOINTS_0 vertex {vertex} index {joint} at slot {slot} exceeds skin joint count {joint_count} in {}",
                                    path.display()
                                );
                            }
                        }
                    }
                }

                for i in 0..positions.len() {
                    let p = bake.transform_point3(Vec3::from(positions[i]));
                    let n = normal_bake
                        .transform_vector3(Vec3::from(normals[i]))
                        .normalize_or_zero();
                    let (j, w) = match node_skin {
                        Some(si) => {
                            // Bounds from the rest-pose skinned position.
                            let raw = Vec3::from(positions[i]);
                            let mut rest = Vec3::ZERO;
                            for k in 0..4 {
                                let m = rest_palette.get(joints[i][k] as usize);
                                if let Some(m) = m {
                                    rest += m.transform_point3(raw) * weights[i][k];
                                }
                            }
                            aabb.0 = aabb.0.min(rest);
                            aabb.1 = aabb.1.max(rest);
                            let base = skin_base[si];
                            (
                                [
                                    base + joints[i][0] as u32,
                                    base + joints[i][1] as u32,
                                    base + joints[i][2] as u32,
                                    base + joints[i][3] as u32,
                                ],
                                weights[i],
                            )
                        }
                        None => {
                            aabb.0 = aabb.0.min(p);
                            aabb.1 = aabb.1.max(p);
                            ([0; 4], [1.0, 0.0, 0.0, 0.0])
                        }
                    };
                    vertices.push(ModelVertex {
                        pos: p.to_array(),
                        normal: n.to_array(),
                        uv: texcoord0
                            .as_deref()
                            .and_then(|texcoords| texcoords.get(i))
                            .copied()
                            .unwrap_or([0.0, 0.0]),
                        uv1: texcoord1
                            .as_deref()
                            .and_then(|texcoords| texcoords.get(i))
                            .copied()
                            .unwrap_or([0.0, 0.0]),
                        joints: j,
                        weights: w,
                    });
                }

                let first = indices.len() as u32;
                match reader.read_indices() {
                    Some(idx) => {
                        for i in idx.into_u32() {
                            if i as usize >= positions.len() {
                                bail!(
                                    "index {i} exceeds POSITION count {} in {}",
                                    positions.len(),
                                    path.display()
                                );
                            }
                            indices.push(base + i);
                        }
                    }
                    None => indices.extend(base..vertices.len() as u32),
                }
                let count = indices.len() as u32 - first;

                let material = prim.material();
                let material_index = material.index();
                if let Some(index) = material_index
                    && matches!(materials[index].model, MaterialModel::Mtoon(_))
                {
                    for texture in materials[index].textures() {
                        if texture.role == TextureRole::Matcap {
                            continue;
                        }
                        let selected = texture.effective_tex_coord();
                        let present = match selected {
                            0 => texcoord0.is_some(),
                            1 => texcoord1.is_some(),
                            _ => false,
                        };
                        if !present {
                            bail!(
                                "MToon material {index} texture {:?} selects TEXCOORD_{selected}, but the primitive does not provide it in {}",
                                texture.role,
                                path.display()
                            );
                        }
                    }
                }
                let pbr = material.pbr_metallic_roughness();
                let image = pbr
                    .base_color_texture()
                    .map(|t| t.texture().source().index());
                let material_name = material.name().map(str::to_owned);
                let material_role = pocket3d_material_role(&material);
                let base_color_mode = pocket3d_material_base_color_mode(&material);
                let mut texture_override = None;
                for (override_index, candidate) in overrides.iter().enumerate() {
                    if !semantic_material_matches(
                        material_role.as_deref(),
                        material_name.as_deref(),
                        candidate.role,
                        candidate.name_prefix,
                    ) {
                        continue;
                    }
                    if texture_override.is_some() {
                        bail!(
                            "material {:?} in {} matches more than one texture override",
                            material_name.as_deref().unwrap_or("<unnamed>"),
                            path.display()
                        );
                    }
                    texture_override = Some(override_index);
                    override_counts[override_index] += 1;
                }

                let mut base_color_factor = pbr.base_color_factor();
                let mut alpha_mode = MaterialAlphaMode::from_gltf(material.alpha_mode());
                let alpha_cutoff = material.alpha_cutoff().unwrap_or(0.5);
                let double_sided = material.double_sided();
                let mut unlit = material.unlit();
                if let Some(override_index) = texture_override {
                    let material_override = &overrides[override_index];
                    if material_override.require_normalized_texcoord0 {
                        validate_normalized_texcoord0(
                            texcoord0.as_deref(),
                            positions.len(),
                            material_name.as_deref(),
                            path,
                        )?;
                    }
                    if material_override.force_white {
                        base_color_factor = [1.0; 4];
                    }
                    if material_override.force_unlit {
                        unlit = true;
                    }
                    if material_override.force_opaque {
                        alpha_mode = MaterialAlphaMode::Opaque;
                    }
                    if material_override.force_blend {
                        alpha_mode = MaterialAlphaMode::Blend;
                    }
                }

                // --- morph targets (sparse deltas, object space) -----------
                let mut targets: Vec<MorphTargetData> = Vec::new();
                for (tpos, tnorm, _ttan) in reader.read_morph_targets() {
                    let mut target = MorphTargetData {
                        pos: Vec::new(),
                        normal: Vec::new(),
                    };
                    if let Some(it) = tpos {
                        let deltas: Vec<_> = it.collect();
                        validate_attribute_count(
                            "morph POSITION",
                            positions.len(),
                            deltas.len(),
                            path,
                        )?;
                        for (i, d) in deltas.into_iter().enumerate() {
                            let d = bake.transform_vector3(Vec3::from(d));
                            if d.length_squared() > 1e-12 {
                                target.pos.push((i as u32, d));
                            }
                        }
                    }
                    if let Some(it) = tnorm {
                        let deltas: Vec<_> = it.collect();
                        validate_attribute_count(
                            "morph NORMAL",
                            positions.len(),
                            deltas.len(),
                            path,
                        )?;
                        for (i, d) in deltas.into_iter().enumerate() {
                            let d = normal_bake.transform_vector3(Vec3::from(d));
                            if d.length_squared() > 1e-12 {
                                target.normal.push((i as u32, d));
                            }
                        }
                    }
                    targets.push(target);
                }
                if targets.is_empty() {
                    prim_morph.push(None);
                } else {
                    let mesh_idx = mesh.index();
                    let slot = morph_meshes
                        .iter()
                        .position(|m| m.mesh == mesh_idx)
                        .unwrap_or_else(|| {
                            morph_meshes.push(MorphMesh {
                                mesh: mesh_idx,
                                target_count: targets.len(),
                                prims: Vec::new(),
                            });
                            morph_meshes.len() - 1
                        });
                    let vertex_count = (vertices.len() as u32) - base;
                    let mm = &mut morph_meshes[slot];
                    mm.target_count = mm.target_count.max(targets.len());
                    prim_morph.push(Some((slot, mm.prims.len())));
                    mm.prims.push(MorphPrim {
                        primitive: primitives_meta.len(),
                        vertex_base: base,
                        vertex_count,
                        overlay_offset: overlay_verts,
                        base: vertices[base as usize..].to_vec(),
                        targets,
                    });
                    overlay_verts += vertex_count;
                }

                primitives_meta.push(PrimitiveUpload {
                    first_index: first,
                    index_count: count,
                    image,
                    base_color_factor,
                    alpha_mode,
                    alpha_cutoff,
                    double_sided,
                    unlit,
                    base_color_mode,
                    material_name,
                    material_role,
                    texture_override,
                    material_index,
                });
            }
        }

        for (material_override, actual) in overrides.iter().zip(override_counts) {
            if let Some(expected) = material_override.expected_primitive_count
                && actual != expected
            {
                bail!(
                    "material override role {:?} expected {expected} primitive(s) in {}, found {actual}",
                    material_override.role,
                    path.display()
                );
            }
        }

        // --- animations -------------------------------------------------------
        let mut clips = Vec::new();
        for anim in doc.animations() {
            let mut channels = Vec::new();
            let mut duration = 0.0f32;
            for (ch_index, ch) in anim.channels().enumerate() {
                let reader = ch.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));
                let Some(times) = reader.read_inputs().map(|it| it.collect::<Vec<f32>>()) else {
                    bail!(
                        "animation channel {ch_index} in {} has no readable input keys",
                        path.display()
                    );
                };
                if times.is_empty() {
                    bail!(
                        "animation channel {ch_index} in {} has zero input keys",
                        path.display()
                    );
                }
                if let Some(&last) = times.last() {
                    duration = duration.max(last);
                }
                let interpolation = match ch.sampler().interpolation() {
                    gltf::animation::Interpolation::Step => Interpolation::Step,
                    // Cubic spline collapses to linear over its key values.
                    _ => Interpolation::Linear,
                };
                let cubic =
                    ch.sampler().interpolation() == gltf::animation::Interpolation::CubicSpline;
                if cubic && times.len() < 2 {
                    bail!(
                        "animation channel {ch_index} in {} has fewer than two CUBICSPLINE keys",
                        path.display()
                    );
                }
                let (channel_path, values) = match reader.read_outputs() {
                    Some(gltf::animation::util::ReadOutputs::Translations(it)) => {
                        (ChannelPath::Translation, flatten3(it.collect(), cubic))
                    }
                    Some(gltf::animation::util::ReadOutputs::Scales(it)) => {
                        (ChannelPath::Scale, flatten3(it.collect(), cubic))
                    }
                    Some(gltf::animation::util::ReadOutputs::Rotations(rot)) => (
                        ChannelPath::Rotation,
                        flatten4(rot.into_f32().collect(), cubic),
                    ),
                    Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(_)) => continue,
                    None => {
                        bail!(
                            "animation channel {ch_index} in {} has output data incompatible with its target",
                            path.display()
                        );
                    }
                };
                let channel = Channel {
                    node: ch.target().node().index(),
                    path: channel_path,
                    interpolation,
                    times,
                    values,
                };
                channel.validate().with_context(|| {
                    format!(
                        "validating animation channel {ch_index} in {}",
                        path.display()
                    )
                })?;
                channel.validate_rotation_keys(false).with_context(|| {
                    format!(
                        "validating animation channel {ch_index} in {}",
                        path.display()
                    )
                })?;
                channels.push(channel);
            }
            clips.push(Clip {
                name: anim.name().unwrap_or("anim").to_string(),
                duration,
                channels,
            });
        }
        log::info!(
            "{}: {} verts, {} clips {:?}, {} skin(s), {} joints",
            path.display(),
            vertices.len(),
            clips.len(),
            clips.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            skins.len(),
            skins.iter().map(|s| s.joints.len()).sum::<usize>()
        );

        // --- upload ------------------------------------------------------------
        use wgpu::util::DeviceExt;
        let vbuf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("model vbuf"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let ibuf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("model ibuf"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        let primitives = primitives_meta
            .into_iter()
            .map(|meta| {
                let (view, sampler) = match meta.texture_override {
                    Some(override_index) => {
                        let material_override = &overrides[override_index];
                        (material_override.texture_view, material_override.sampler)
                    }
                    None => (
                        meta.image
                            .and_then(|i| textures.get(i))
                            .map(|t| &t.view)
                            .unwrap_or(&white.view),
                        &samplers.aniso_repeat,
                    ),
                };
                let material = MaterialRaw {
                    base_color_factor: meta.base_color_factor,
                    params: [
                        meta.unlit as u8 as f32,
                        if meta.alpha_mode == MaterialAlphaMode::Mask {
                            meta.alpha_cutoff
                        } else {
                            0.0
                        },
                        meta.double_sided as u8 as f32,
                        (meta.alpha_mode == MaterialAlphaMode::Blend) as u8 as f32,
                    ],
                    style: [
                        (meta.base_color_mode == MaterialBaseColorMode::Monochrome) as u8 as f32,
                        0.0,
                        0.0,
                        0.0,
                    ],
                };
                let label = meta.material_name.as_deref().unwrap_or("model material");
                let (bind_group, material_buf) =
                    make_material_bind_group(gpu, layout, label, view, sampler, &material);
                Primitive {
                    first_index: meta.first_index,
                    index_count: meta.index_count,
                    bind_group,
                    alpha_mode: meta.alpha_mode,
                    alpha_cutoff: meta.alpha_cutoff,
                    double_sided: meta.double_sided,
                    unlit: meta.unlit,
                    base_color_mode: meta.base_color_mode,
                    material_name: meta.material_name,
                    material_role: meta.material_role,
                    material_index: meta.material_index,
                    render_phase: RenderPhase::fallback(meta.alpha_mode),
                    material_buf,
                }
            })
            .collect();

        textures.push(white);
        Ok(Arc::new(Self {
            vbuf,
            ibuf,
            primitives,
            materials,
            skeleton,
            node_names,
            skins,
            clips,
            morph_meshes,
            prim_morph,
            aabb,
            textures,
        }))
    }
}

fn find_node_named(node_names: &[Option<String>], name: &str) -> Option<usize> {
    node_names
        .iter()
        .position(|candidate| candidate.as_deref() == Some(name))
}

fn sample_node_transform(
    skeleton: &Skeleton,
    clips: &[Clip],
    node: usize,
    anim: &AnimState,
) -> Option<Mat4> {
    if node >= skeleton.rest.len() {
        return None;
    }
    let mut globals = Vec::new();
    skeleton.global_transforms(clips.get(anim.clip), anim.time, anim.looping, &mut globals);
    globals.get(node).copied()
}

fn flatten3(v: Vec<[f32; 3]>, cubic: bool) -> Vec<f32> {
    strip_cubic(v, cubic)
        .into_iter()
        .flat_map(|a| a.into_iter())
        .collect()
}

fn flatten4(v: Vec<[f32; 4]>, cubic: bool) -> Vec<f32> {
    strip_cubic(v, cubic)
        .into_iter()
        .flat_map(|a| a.into_iter())
        .collect()
}

/// Cubic-spline outputs store [in-tangent, value, out-tangent] per key;
/// keep just the values.
fn strip_cubic<T: Copy>(v: Vec<T>, cubic: bool) -> Vec<T> {
    if !cubic {
        return v;
    }
    v.as_chunks::<3>().0.iter().map(|c| c[1]).collect()
}

fn validate_load_options(opts: &ModelLoadOptions) -> Result<()> {
    if opts.max_texture_dim == Some(0) {
        bail!("max_texture_dim must be greater than zero");
    }
    Ok(())
}

/// Box-filter 2×2 downsample (straight alpha; fine for albedo maps).
fn downsample_rgba(px: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = Vec::with_capacity((nw * nh * 4) as usize);
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = [0u32; 4];
            let mut samples = 0;
            for sy in (y * 2)..((y * 2 + 2).min(h)) {
                for sx in (x * 2)..((x * 2 + 2).min(w)) {
                    let src = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += px[src + c] as u32;
                    }
                    samples += 1;
                }
            }
            out.extend(acc.map(|v| (v / samples) as u8));
        }
    }
    out
}

fn cap_texture_rgba(
    mut rgba: Vec<u8>,
    mut width: u32,
    mut height: u32,
    max_texture_dim: Option<u32>,
) -> (Vec<u8>, u32, u32) {
    let Some(max) = max_texture_dim else {
        return (rgba, width, height);
    };
    while width.max(height) > max {
        rgba = downsample_rgba(&rgba, width, height);
        width = width.div_ceil(2);
        height = height.div_ceil(2);
    }
    (rgba, width, height)
}

fn to_rgba8(img: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;
    let n = (img.width * img.height) as usize;
    match img.format {
        Format::R8G8B8A8 => img.pixels.clone(),
        Format::R8G8B8 => {
            let mut out = Vec::with_capacity(n * 4);
            for c in img.pixels.as_chunks::<3>().0 {
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
            out
        }
        Format::R8 => {
            let mut out = Vec::with_capacity(n * 4);
            for &g in &img.pixels {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            out
        }
        Format::R8G8 => {
            let mut out = Vec::with_capacity(n * 4);
            for c in img.pixels.as_chunks::<2>().0 {
                // PNG's two-channel color type is grayscale + alpha. The
                // glTF importer exposes that decoded storage as R8G8, so a
                // base-color texture must replicate luminance into RGB and
                // retain the second channel as alpha. Treating the channels
                // as red + green turns white decals neon green and opaque.
                out.extend_from_slice(&[c[0], c[0], c[0], c[1]]);
            }
            out
        }
        // 16-bit and float formats: take the high byte / clamp.
        _ => {
            log::warn!("unsupported glTF image format {:?}; using grey", img.format);
            vec![128; n * 4]
        }
    }
}

/// A model placed in the scene.
pub struct ModelInstance {
    pub asset: Arc<ModelAsset>,
    /// Mutable material values owned by this instance and indexed by authored
    /// glTF material index. Stage G can upload these lazily without mutating
    /// the shared asset.
    pub materials: MaterialStateSet,
    pub transform: Mat4,
    pub tint: [f32; 4],
    pub anim: AnimState,
    /// 0..1 how strongly lighting applies (1 = fully lit by sun/ambient).
    pub lit: f32,
    /// Explicit node globals (from `Skeleton::globals_from_locals`) override
    /// `anim` when set — for procedurally posed characters.
    pub pose: Option<Vec<Mat4>>,
    /// Morph weights + overlay buffer; create via `asset.create_morph_state`.
    pub morph: Option<MorphState>,
    /// Alpha-test threshold for this instance's primitives (0 = off).
    /// Anime-style characters use cutout textures for hair/lashes.
    pub cutout: f32,
}

impl ModelInstance {
    pub fn new(asset: Arc<ModelAsset>) -> Self {
        let materials = MaterialStateSet::from_assets(&asset.materials);
        Self {
            asset,
            materials,
            transform: Mat4::IDENTITY,
            tint: [1.0; 4],
            anim: AnimState::default(),
            lit: 1.0,
            pose: None,
            morph: None,
            cutout: 0.0,
        }
    }
}

type ImportedGltf = (
    gltf::Document,
    Vec<gltf::buffer::Data>,
    Vec<gltf::image::Data>,
);

fn validate_attribute_count(
    attribute: &str,
    position_count: usize,
    actual_count: usize,
    path: &Path,
) -> Result<()> {
    if actual_count != position_count {
        bail!(
            "{attribute} accessor count {actual_count} does not match POSITION count {position_count} in {}",
            path.display()
        );
    }
    Ok(())
}

const MAX_RENDERER_JOINTS: usize = 512;

fn validate_joint_palette_count(joint_count: usize, path: &Path) -> Result<()> {
    if joint_count > MAX_RENDERER_JOINTS {
        bail!(
            "combined skin joint count {joint_count} exceeds renderer joint palette limit {MAX_RENDERER_JOINTS} in {}",
            path.display()
        );
    }
    Ok(())
}

fn validate_model_input(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    path: &Path,
) -> Result<()> {
    let node_count = doc.nodes().count();
    let skin_joint_counts: Vec<Vec<usize>> = doc
        .skins()
        .map(|skin| skin.joints().map(|joint| joint.index()).collect())
        .collect();
    let combined_joint_count = skin_joint_counts.iter().map(Vec::len).sum();
    validate_joint_palette_count(combined_joint_count, path)?;

    for (skin_index, skin) in doc.skins().enumerate() {
        let joint_count = skin_joint_counts[skin_index].len();
        let reader = skin.reader(|b| buffers.get(b.index()).map(|data| data.0.as_slice()));
        if let Some(matrices) = reader.read_inverse_bind_matrices() {
            let matrix_count = matrices.count();
            if matrix_count != joint_count {
                bail!(
                    "inverse bind matrix count {matrix_count} does not match skin joint count {joint_count} in {}",
                    path.display()
                );
            }
        }
        for (joint_index, &node) in skin_joint_counts[skin_index].iter().enumerate() {
            if node >= node_count {
                bail!(
                    "skin joint {joint_index} references node {node}, but the model has {node_count} nodes in {}",
                    path.display()
                );
            }
        }
    }

    for node in doc.nodes() {
        let node_skin = node.skin().map(|skin| skin.index());
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let reader =
                    primitive.reader(|b| buffers.get(b.index()).map(|data| data.0.as_slice()));
                let Some(positions) = reader.read_positions() else {
                    continue;
                };
                let position_count = positions.count();

                if let Some(normals) = reader.read_normals() {
                    validate_attribute_count("NORMAL", position_count, normals.count(), path)?;
                }
                if let Some(texcoords) = reader.read_tex_coords(0) {
                    validate_attribute_count(
                        "TEXCOORD_0",
                        position_count,
                        texcoords.into_f32().count(),
                        path,
                    )?;
                }
                if let Some(texcoords) = reader.read_tex_coords(1) {
                    validate_attribute_count(
                        "TEXCOORD_1",
                        position_count,
                        texcoords.into_f32().count(),
                        path,
                    )?;
                }
                if let Some(joints) = reader.read_joints(0) {
                    let joints: Vec<_> = joints.into_u16().collect();
                    validate_attribute_count("JOINTS_0", position_count, joints.len(), path)?;
                    if let Some(skin_index) = node_skin {
                        let joint_count = skin_joint_counts[skin_index].len();
                        for (vertex, joints) in joints.iter().enumerate() {
                            for (slot, &joint) in joints.iter().enumerate() {
                                if joint as usize >= joint_count {
                                    bail!(
                                        "JOINTS_0 vertex {vertex} index {joint} at slot {slot} exceeds skin joint count {joint_count} in {}",
                                        path.display()
                                    );
                                }
                            }
                        }
                    }
                }
                if let Some(weights) = reader.read_weights(0) {
                    validate_attribute_count(
                        "WEIGHTS_0",
                        position_count,
                        weights.into_f32().count(),
                        path,
                    )?;
                }
                if let Some(indices) = reader.read_indices() {
                    for index in indices.into_u32() {
                        if index as usize >= position_count {
                            bail!(
                                "index {index} exceeds POSITION count {position_count} in {}",
                                path.display()
                            );
                        }
                    }
                }

                for (target_index, (positions, normals, _tangents)) in
                    reader.read_morph_targets().enumerate()
                {
                    if let Some(positions) = positions {
                        validate_attribute_count(
                            &format!("morph POSITION target {target_index}"),
                            position_count,
                            positions.count(),
                            path,
                        )?;
                    }
                    if let Some(normals) = normals {
                        validate_attribute_count(
                            &format!("morph NORMAL target {target_index}"),
                            position_count,
                            normals.count(),
                            path,
                        )?;
                    }
                }
            }
        }
    }

    for (animation_index, animation) in doc.animations().enumerate() {
        for (channel_index, channel) in animation.channels().enumerate() {
            let input = channel.sampler().input();
            if input.count() == 0 {
                bail!(
                    "animation channel {channel_index} in animation {animation_index} in {} has zero input keys",
                    path.display()
                );
            }
            let reader = channel.reader(|b| buffers.get(b.index()).map(|data| data.0.as_slice()));
            let Some(times) = reader.read_inputs() else {
                bail!(
                    "animation channel {channel_index} in animation {animation_index} in {} has no readable input keys",
                    path.display()
                );
            };
            let times: Vec<_> = times.collect();
            if times.is_empty() {
                bail!(
                    "animation channel {channel_index} in animation {animation_index} in {} has zero input keys",
                    path.display()
                );
            }
            if times.iter().any(|time| !time.is_finite() || *time < 0.0)
                || times.windows(2).any(|window| window[1] <= window[0])
            {
                bail!(
                    "animation channel {channel_index} in animation {animation_index} has invalid or unsorted input keys in {}",
                    path.display()
                );
            }

            let property = channel.target().property();
            if matches!(property, gltf::animation::Property::MorphTargetWeights) {
                continue;
            }
            let cubic =
                channel.sampler().interpolation() == gltf::animation::Interpolation::CubicSpline;
            if cubic && times.len() < 2 {
                bail!(
                    "animation channel {channel_index} in animation {animation_index} has fewer than two CUBICSPLINE keys in {}",
                    path.display()
                );
            }
            let output = channel.sampler().output();
            let expected_dimensions = match property {
                gltf::animation::Property::Translation | gltf::animation::Property::Scale => {
                    gltf::accessor::Dimensions::Vec3
                }
                gltf::animation::Property::Rotation => gltf::accessor::Dimensions::Vec4,
                gltf::animation::Property::MorphTargetWeights => unreachable!(),
            };
            if output.dimensions() != expected_dimensions {
                bail!(
                    "animation channel {channel_index} in animation {animation_index} has output dimensions incompatible with its target in {}",
                    path.display()
                );
            }
            let expected_output_count = input
                .count()
                .checked_mul(if cubic { 3 } else { 1 })
                .ok_or_else(|| anyhow::anyhow!("animation output count overflows"))?;
            if output.count() != expected_output_count {
                bail!(
                    "animation channel output count {} does not match input key count {}{} in {}",
                    output.count(),
                    input.count(),
                    if cubic { " × 3 for CUBICSPLINE" } else { "" },
                    path.display()
                );
            }
            let Some(outputs) = reader.read_outputs() else {
                bail!(
                    "animation channel {channel_index} in animation {animation_index} in {} has output data incompatible with its target",
                    path.display()
                );
            };
            let output_count = match outputs {
                gltf::animation::util::ReadOutputs::Translations(values) => values.count(),
                gltf::animation::util::ReadOutputs::Scales(values) => values.count(),
                gltf::animation::util::ReadOutputs::Rotations(values) => {
                    let values: Vec<_> = values.into_f32().collect();
                    let stride = if cubic { 3 } else { 1 };
                    for key in 0..times.len() {
                        let value_index = key * stride + usize::from(cubic);
                        let Some(&quaternion) = values.get(value_index) else {
                            bail!(
                                "animation channel {channel_index} in animation {animation_index} has truncated rotation output in {}",
                                path.display()
                            );
                        };
                        let length_squared =
                            quaternion.iter().map(|value| value * value).sum::<f32>();
                        if !length_squared.is_finite() || length_squared <= 1.0e-12 {
                            bail!(
                                "animation channel {channel_index} in animation {animation_index} has a zero or non-finite rotation quaternion at key {key} in {}",
                                path.display()
                            );
                        }
                    }
                    values.len()
                }
                gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => continue,
            };
            if output_count != expected_output_count {
                bail!(
                    "animation channel output count {output_count} does not match input key count {}{} in {}",
                    times.len(),
                    if cubic { " × 3 for CUBICSPLINE" } else { "" },
                    path.display()
                );
            }
        }
    }
    Ok(())
}

fn normalize_glb_container<'a>(bytes: &'a [u8], label: &str) -> Result<Cow<'a, [u8]>> {
    if bytes.len() < 12 {
        bail!("invalid GLB {label}: header is truncated");
    }
    if &bytes[..4] != b"glTF" {
        bail!("invalid GLB {label}: bad magic");
    }
    let mut version_bytes = [0u8; 4];
    version_bytes.copy_from_slice(&bytes[4..8]);
    let version = u32::from_le_bytes(version_bytes);
    if version != 2 {
        bail!("invalid GLB {label}: unsupported version {version}");
    }
    let mut length_bytes = [0u8; 4];
    length_bytes.copy_from_slice(&bytes[8..12]);
    let declared_length = u32::from_le_bytes(length_bytes) as usize;
    if declared_length != bytes.len() {
        bail!(
            "invalid GLB {label}: declared length {declared_length} does not match byte length {}",
            bytes.len()
        );
    }

    let read_chunk = |offset: usize| -> Result<([u8; 4], usize)> {
        if offset > declared_length || declared_length - offset < 8 {
            bail!("invalid GLB {label}: truncated chunk header");
        }
        let mut chunk_length_bytes = [0u8; 4];
        chunk_length_bytes.copy_from_slice(&bytes[offset..offset + 4]);
        let chunk_length = u32::from_le_bytes(chunk_length_bytes) as usize;
        let data_start = offset + 8;
        let data_end = data_start
            .checked_add(chunk_length)
            .ok_or_else(|| anyhow::anyhow!("invalid GLB {label}: chunk length overflows"))?;
        if data_end > declared_length {
            bail!("invalid GLB {label}: chunk exceeds declared length");
        }
        let mut chunk_type = [0u8; 4];
        chunk_type.copy_from_slice(&bytes[offset + 4..offset + 8]);
        Ok((chunk_type, data_end))
    };

    let (json_type, json_end) = read_chunk(12)?;
    if &json_type != b"JSON" {
        bail!("invalid GLB {label}: first chunk is not JSON");
    }

    let mut offset = json_end;
    let mut bin_range = None;
    let mut has_unknown_chunks = false;
    while offset < declared_length {
        let chunk_start = offset;
        let (chunk_type, chunk_end) = read_chunk(offset)?;
        if &chunk_type == b"BIN\0" {
            if bin_range.is_some() {
                bail!("invalid GLB {label}: multiple BIN chunks");
            }
            bin_range = Some((chunk_start, chunk_end));
        } else if &chunk_type == b"JSON" {
            bail!("invalid GLB {label}: multiple JSON chunks");
        } else {
            has_unknown_chunks = true;
        }
        offset = chunk_end;
    }

    if !has_unknown_chunks {
        return Ok(Cow::Borrowed(bytes));
    }

    // gltf 1.4.1 rejects unknown chunks while parsing, even though the GLB
    // format requires clients to ignore them. Repack only the recognized
    // JSON/BIN chunks for that parser; all chunk bounds were checked above.
    let bin_len = bin_range.map_or(0, |(start, end)| end - start);
    let normalized_len = 12 + (json_end - 12) + bin_len;
    let normalized_len_u32 = u32::try_from(normalized_len)
        .with_context(|| format!("normalizing GLB {label}: normalized length overflows u32"))?;
    let mut normalized = Vec::with_capacity(normalized_len);
    normalized.extend_from_slice(&bytes[..8]);
    normalized.extend_from_slice(&normalized_len_u32.to_le_bytes());
    normalized.extend_from_slice(&bytes[12..json_end]);
    if let Some((bin_start, bin_end)) = bin_range {
        normalized.extend_from_slice(&bytes[bin_start..bin_end]);
    }
    Ok(Cow::Owned(normalized))
}

fn validate_self_contained_resources(doc: &gltf::Document) -> Result<()> {
    for buffer in doc.buffers() {
        if let gltf::buffer::Source::Uri(uri) = buffer.source()
            && !uri.starts_with("data:")
        {
            bail!("external buffer URI {uri:?} is not allowed for byte-loaded GLB");
        }
    }
    for image in doc.images() {
        if let gltf::image::Source::Uri { uri, .. } = image.source() {
            if !uri.starts_with("data:") {
                bail!("external image URI {uri:?} is not allowed for byte-loaded GLB");
            }
        }
    }
    Ok(())
}

const MAX_BYTE_IMAGE_DECODE_DIM: u32 = 16_384;
const DEFAULT_BYTE_IMAGE_DECODE_DIM: u32 = 8_192;
const MAX_BYTE_IMAGE_DECODE_ALLOC: u64 = 512 * 1024 * 1024;
const MAX_BYTE_IMAGE_COUNT: usize = 64;
const MAX_BYTE_IMAGE_BYTES: u64 = 256 * 1024 * 1024;

fn byte_image_decode_dim(max_texture_dim: Option<u32>) -> u32 {
    max_texture_dim
        .map(|max| max.saturating_mul(4).clamp(2, MAX_BYTE_IMAGE_DECODE_DIM))
        .unwrap_or(DEFAULT_BYTE_IMAGE_DECODE_DIM)
}

fn image_decode_limits(max_decode_dim: u32) -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(max_decode_dim);
    limits.max_image_height = Some(max_decode_dim);
    limits.max_alloc = Some(MAX_BYTE_IMAGE_DECODE_ALLOC);
    limits
}

#[derive(Default)]
struct ByteImageBudget {
    image_count: usize,
    decoded_bytes: u64,
}

impl ByteImageBudget {
    fn reserve(&mut self, image_index: usize, width: u32, height: u32, label: &str) -> Result<()> {
        if self.image_count >= MAX_BYTE_IMAGE_COUNT {
            bail!(
                "byte-loaded GLB {label} references more than {MAX_BYTE_IMAGE_COUNT} decoded images"
            );
        }
        let bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| {
                anyhow::anyhow!("decoded image {image_index} in {label} is too large")
            })?;
        let next = self
            .decoded_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("decoded image byte budget overflows in {label}"))?;
        if next > MAX_BYTE_IMAGE_BYTES {
            bail!(
                "decoded image byte budget exceeds {MAX_BYTE_IMAGE_BYTES} bytes at image {image_index} in {label}"
            );
        }
        self.image_count += 1;
        self.decoded_bytes = next;
        Ok(())
    }
}

fn decode_byte_image(
    encoded: &[u8],
    mime_type: Option<&str>,
    max_decode_dim: u32,
    budget: &mut ByteImageBudget,
    label: &str,
    image_index: usize,
) -> Result<gltf::image::Data> {
    let make_reader = || -> Result<image::ImageReader<Cursor<&[u8]>>> {
        match mime_type {
            Some("image/png") => Ok(image::ImageReader::with_format(
                Cursor::new(encoded),
                image::ImageFormat::Png,
            )),
            Some("image/jpeg") => Ok(image::ImageReader::with_format(
                Cursor::new(encoded),
                image::ImageFormat::Jpeg,
            )),
            _ => image::ImageReader::new(Cursor::new(encoded))
                .with_guessed_format()
                .with_context(|| format!("guessing format for image {image_index} in {label}")),
        }
    };
    let mut reader = make_reader()?;
    reader.limits(image_decode_limits(max_decode_dim));
    let (width, height) = reader
        .into_dimensions()
        .with_context(|| format!("reading dimensions for image {image_index} in {label}"))?;
    budget.reserve(image_index, width, height, label)?;

    let mut reader = make_reader()?;
    reader.limits(image_decode_limits(max_decode_dim));
    let decoded = reader
        .decode()
        .with_context(|| format!("decoding image {image_index} in {label}"))?;
    let rgba = decoded.to_rgba8();
    let width = rgba.width();
    let height = rgba.height();
    Ok(gltf::image::Data {
        pixels: rgba.into_raw(),
        format: gltf::image::Format::R8G8B8A8,
        width,
        height,
    })
}

fn decode_data_uri(uri: &str, label: &str, image_index: usize) -> Result<Vec<u8>> {
    let encoded = uri
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(";base64,").map(|(_, encoded)| encoded))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "image {image_index} in {label} has an unsupported data URI; expected base64"
            )
        })?;
    base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .with_context(|| format!("decoding data URI for image {image_index} in {label}"))
}

fn decode_byte_image_source(
    source: gltf::image::Source<'_>,
    buffers: &[gltf::buffer::Data],
    max_decode_dim: u32,
    budget: &mut ByteImageBudget,
    label: &str,
    image_index: usize,
) -> Result<gltf::image::Data> {
    match source {
        gltf::image::Source::Uri { uri, mime_type } => {
            let encoded = decode_data_uri(uri, label, image_index)?;
            decode_byte_image(
                &encoded,
                mime_type,
                max_decode_dim,
                budget,
                label,
                image_index,
            )
        }
        gltf::image::Source::View { view, mime_type } => {
            let data = buffers.get(view.buffer().index()).ok_or_else(|| {
                anyhow::anyhow!("image {image_index} in {label} references a missing buffer")
            })?;
            let begin = view.offset();
            let end = begin.checked_add(view.length()).ok_or_else(|| {
                anyhow::anyhow!("image {image_index} in {label} buffer view overflows")
            })?;
            let encoded = data.0.get(begin..end).ok_or_else(|| {
                anyhow::anyhow!("image {image_index} in {label} buffer view exceeds its buffer")
            })?;
            decode_byte_image(
                encoded,
                Some(mime_type),
                max_decode_dim,
                budget,
                label,
                image_index,
            )
        }
    }
}

fn import_byte_images(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    max_texture_dim: Option<u32>,
    label: &str,
) -> Result<Vec<gltf::image::Data>> {
    let image_slot_count = doc.images().count();
    if image_slot_count > MAX_BYTE_IMAGE_COUNT {
        bail!(
            "byte-loaded GLB {label} declares {image_slot_count} images, exceeding the limit of {MAX_BYTE_IMAGE_COUNT}"
        );
    }
    let used: HashSet<usize> = doc
        .nodes()
        .filter_map(|node| node.mesh())
        .flat_map(|mesh| mesh.primitives())
        .filter_map(|primitive| {
            primitive
                .material()
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|texture| texture.texture().source().index())
        })
        .collect();
    let max_decode_dim = byte_image_decode_dim(max_texture_dim);
    let mut budget = ByteImageBudget::default();
    let fallback = gltf::image::Data {
        pixels: vec![255, 255, 255, 255],
        format: gltf::image::Format::R8G8B8A8,
        width: 1,
        height: 1,
    };
    let mut images = vec![fallback; doc.images().count()];
    for image in doc.images() {
        if !used.contains(&image.index()) {
            continue;
        }
        images[image.index()] = decode_byte_image_source(
            image.source(),
            buffers,
            max_decode_dim,
            &mut budget,
            label,
            image.index(),
        )?;
    }
    Ok(images)
}

#[cfg(test)]
fn import_glb_slice(
    bytes: &[u8],
    label: &str,
    allowed_required_extensions: &[String],
) -> Result<ImportedGltf> {
    import_glb_slice_with_options(bytes, label, allowed_required_extensions, None)
}

fn import_glb_slice_with_options(
    bytes: &[u8],
    label: &str,
    allowed_required_extensions: &[String],
    max_texture_dim: Option<u32>,
) -> Result<ImportedGltf> {
    let normalized = normalize_glb_container(bytes, label)?;
    let mut parsed = gltf::Gltf::from_slice_without_validation(&normalized)
        .with_context(|| format!("parsing embedded GLB {label}"))?;
    let mut root = parsed.document.into_json();
    let required_extensions = root.extensions_required.clone();
    let document = match gltf::Document::from_json(root.clone()) {
        // This is the normal glTF crate validation path. It accepts every
        // extension compiled into the dependency, including KHR_materials_unlit.
        Ok(document) => document,
        Err(validation_error) => {
            // Retry with the required-extension list removed only to tell an
            // unsupported required extension apart from an ordinary malformed
            // core document. The retry never becomes the accepted document.
            let mut core_root = root.clone();
            core_root.extensions_required.clear();
            if gltf::Document::from_json(core_root).is_err() {
                return Err(validation_error)
                    .with_context(|| format!("validating embedded GLB {label}"));
            }
            for extension in required_extensions.iter().filter(|extension| {
                !allowed_required_extensions
                    .iter()
                    .any(|allowed| allowed == *extension)
            }) {
                // Probe each name in isolation. This distinguishes an
                // unknown extension from a supported required extension when
                // both appear in the same document.
                let mut trial_root = root.clone();
                trial_root.extensions_required = vec![extension.clone()];
                if gltf::Document::from_json(trial_root).is_err() {
                    bail!(
                        "required extension {extension} is not enabled or caller-validated for byte-loaded GLB {label}"
                    );
                }
            }
            root.extensions_required.retain(|extension| {
                !allowed_required_extensions
                    .iter()
                    .any(|allowed| allowed == extension)
            });
            gltf::Document::from_json(root)
                .with_context(|| format!("validating embedded GLB {label}"))?
        }
    };
    validate_self_contained_resources(&document)?;
    let buffers = gltf::import_buffers(&document, None, parsed.blob.take())
        .with_context(|| format!("importing embedded GLB buffers {label}"))?;
    let images = import_byte_images(&document, &buffers, max_texture_dim, label)
        .with_context(|| format!("importing embedded GLB images {label}"))?;
    Ok((document, buffers, images))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use base64::Engine as _;
    use glam::{Quat, Vec3};
    use serde_json::{Value, json};

    use crate::anim::{AnimState, Channel, ChannelPath, Clip, Interpolation, NodeTrs, Skeleton};
    use crate::gpu::Gpu;
    use crate::material::TextureColorSpace;
    use crate::texture::Samplers;

    use super::{
        ByteImageBudget, MAX_BYTE_IMAGE_BYTES, MAX_BYTE_IMAGE_COUNT, MaterialBaseColorMode,
        MaterialRaw, ModelAsset, ModelLoadOptions, ModelTextureCacheKey, authored_materials,
        cap_texture_rgba, find_node_named, import_glb_slice, import_glb_slice_with_options,
        pocket3d_base_color_mode_from_extras, pocket3d_role_from_extras, sample_node_transform,
        semantic_material_matches, to_rgba8, validate_joint_palette_count, validate_model_input,
        validate_normalized_texcoord0,
    };

    fn glb_from_json(value: Value, bin: &[u8]) -> Vec<u8> {
        let mut json = serde_json::to_vec(&value).unwrap();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let include_bin = !bin.is_empty() || value.get("buffers").is_some();
        let padded_bin_len = (bin.len() + 3) & !3;
        let total_len = 12 + 8 + json.len() + if include_bin { 8 + padded_bin_len } else { 0 };
        let mut glb = Vec::with_capacity(total_len);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total_len as u32).to_le_bytes());
        glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json);
        if include_bin {
            glb.extend_from_slice(&(padded_bin_len as u32).to_le_bytes());
            glb.extend_from_slice(b"BIN\0");
            glb.extend_from_slice(bin);
            glb.resize(glb.len() + padded_bin_len - bin.len(), 0);
        }
        glb
    }

    fn minimal_glb() -> Vec<u8> {
        glb_from_json(
            json!({
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": []}]
            }),
            &[],
        )
    }

    fn append_unknown_chunk(mut glb: Vec<u8>) -> Vec<u8> {
        let payload = [1u8, 2, 3, 4];
        glb.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"TEST");
        glb.extend_from_slice(&payload);
        let length = glb.len() as u32;
        glb[8..12].copy_from_slice(&length.to_le_bytes());
        glb
    }

    fn png_bytes(width: u32, height: u32) -> Vec<u8> {
        let mut encoded = Vec::new();
        let mut encoder = png::Encoder::new(&mut encoded, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_image_data(&vec![255; (width * height * 4) as usize])
            .unwrap();
        drop(writer);
        encoded
    }

    fn data_uri_image_glb(width: u32, height: u32) -> Vec<u8> {
        let encoded = base64::engine::general_purpose::STANDARD.encode(png_bytes(width, height));
        let mut bin = Vec::new();
        let mut views = Vec::new();
        let mut accessors = Vec::new();
        let position = append_f32_accessor(&mut bin, &mut views, &mut accessors, 3, 3);
        glb_from_json(
            json!({
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{
                    "attributes": {"POSITION": position},
                    "material": 0
                }]}],
                "images": [{"uri": format!("data:image/png;base64,{encoded}")}],
                "textures": [{"source": 0}],
                "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
                "buffers": [{"byteLength": bin.len()}],
                "bufferViews": views,
                "accessors": accessors
            }),
            &bin,
        )
    }

    fn append_f32_accessor(
        bin: &mut Vec<u8>,
        views: &mut Vec<Value>,
        accessors: &mut Vec<Value>,
        count: usize,
        components: usize,
    ) -> usize {
        let offset = bin.len();
        for i in 0..count * components {
            bin.extend_from_slice(&(i as f32).to_le_bytes());
        }
        let view = views.len();
        views.push(json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": count * components * 4,
            "target": 34962
        }));
        let accessor_index = accessors.len();
        let kind = match components {
            1 => "SCALAR",
            2 => "VEC2",
            3 => "VEC3",
            4 => "VEC4",
            _ => panic!("unsupported test accessor width"),
        };
        let mut accessor = json!({
            "bufferView": view,
            "componentType": 5126,
            "count": count,
            "type": kind
        });
        if components == 3 {
            accessor["min"] = json!([0.0, 0.0, 0.0]);
            accessor["max"] = json!([1.0, 1.0, 1.0]);
        }
        accessors.push(accessor);
        accessor_index
    }

    fn append_u16_vec4_accessor(
        bin: &mut Vec<u8>,
        views: &mut Vec<Value>,
        accessors: &mut Vec<Value>,
        count: usize,
    ) -> usize {
        let offset = bin.len();
        for _ in 0..count {
            bin.extend_from_slice(&[0u8; 8]);
        }
        let view = views.len();
        views.push(json!({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": count * 8,
            "target": 34962
        }));
        let accessor = accessors.len();
        accessors.push(json!({
            "bufferView": view,
            "componentType": 5123,
            "count": count,
            "type": "VEC4"
        }));
        accessor
    }

    fn mesh_fixture(
        normal_count: Option<usize>,
        joints_count: Option<usize>,
        weights_count: Option<usize>,
        morph_position_count: Option<usize>,
        morph_normal_count: Option<usize>,
    ) -> Vec<u8> {
        let mut bin = Vec::new();
        let mut views = Vec::new();
        let mut accessors = Vec::new();
        let position = append_f32_accessor(&mut bin, &mut views, &mut accessors, 4, 3);
        let normal = normal_count
            .map(|count| append_f32_accessor(&mut bin, &mut views, &mut accessors, count, 3));
        let joints = joints_count
            .map(|count| append_u16_vec4_accessor(&mut bin, &mut views, &mut accessors, count));
        let weights = weights_count
            .map(|count| append_f32_accessor(&mut bin, &mut views, &mut accessors, count, 4));
        let morph_position = morph_position_count
            .map(|count| append_f32_accessor(&mut bin, &mut views, &mut accessors, count, 3));
        let morph_normal = morph_normal_count
            .map(|count| append_f32_accessor(&mut bin, &mut views, &mut accessors, count, 3));

        let mut attributes = serde_json::Map::new();
        attributes.insert("POSITION".to_owned(), json!(position));
        if let Some(normal) = normal {
            attributes.insert("NORMAL".to_owned(), json!(normal));
        }
        if let Some(joints) = joints {
            attributes.insert("JOINTS_0".to_owned(), json!(joints));
        }
        if let Some(weights) = weights {
            attributes.insert("WEIGHTS_0".to_owned(), json!(weights));
        }
        let mut primitive = json!({"attributes": attributes});
        let mut targets = Vec::new();
        let mut target = serde_json::Map::new();
        if let Some(position) = morph_position {
            target.insert("POSITION".to_owned(), json!(position));
        }
        if let Some(normal) = morph_normal {
            target.insert("NORMAL".to_owned(), json!(normal));
        }
        if !target.is_empty() {
            targets.push(Value::Object(target));
            primitive["targets"] = Value::Array(targets);
        }

        glb_from_json(
            json!({
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [primitive]}],
                "buffers": [{"byteLength": bin.len()}],
                "bufferViews": views,
                "accessors": accessors
            }),
            &bin,
        )
    }

    fn animation_fixture(
        input_count: usize,
        output_count: usize,
        output_components: usize,
        path: &str,
    ) -> Vec<u8> {
        animation_fixture_with_interpolation(
            input_count,
            output_count,
            output_components,
            path,
            "LINEAR",
        )
    }

    fn animation_fixture_with_interpolation(
        input_count: usize,
        output_count: usize,
        output_components: usize,
        path: &str,
        interpolation: &str,
    ) -> Vec<u8> {
        let mut bin = Vec::new();
        let mut views = Vec::new();
        let mut accessors = Vec::new();
        let input = append_f32_accessor(&mut bin, &mut views, &mut accessors, input_count, 1);
        let output = append_f32_accessor(
            &mut bin,
            &mut views,
            &mut accessors,
            output_count,
            output_components,
        );
        glb_from_json(
            json!({
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{}],
                "animations": [{
                    "samplers": [{"input": input, "output": output, "interpolation": interpolation}],
                    "channels": [{"sampler": 0, "target": {"node": 0, "path": path}}]
                }],
                "buffers": [{"byteLength": bin.len()}],
                "bufferViews": views,
                "accessors": accessors
            }),
            &bin,
        )
    }

    fn animation_fixture_with_duplicate_times() -> Vec<u8> {
        let mut bytes = animation_fixture(2, 2, 3, "translation");
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let input_start = 12 + 8 + json_len + 8;
        bytes[input_start + 4..input_start + 8].copy_from_slice(&0.0f32.to_le_bytes());
        bytes
    }

    fn zero_rotation_animation_fixture() -> Vec<u8> {
        let mut bytes = animation_fixture(1, 1, 4, "rotation");
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let output_start = 12 + 8 + json_len + 8 + 4;
        for offset in (output_start..output_start + 16).step_by(4) {
            bytes[offset..offset + 4].copy_from_slice(&0.0f32.to_le_bytes());
        }
        bytes
    }

    fn unlit_fixture() -> Vec<u8> {
        let mut bin = Vec::new();
        let mut views = Vec::new();
        let mut accessors = Vec::new();
        let position = append_f32_accessor(&mut bin, &mut views, &mut accessors, 3, 3);
        glb_from_json(
            json!({
                "asset": {"version": "2.0"},
                "extensionsUsed": ["KHR_materials_unlit", "X_TEST_VALIDATED"],
                "extensionsRequired": ["KHR_materials_unlit", "X_TEST_VALIDATED"],
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": position}, "material": 0}]}],
                "materials": [{"extensions": {"KHR_materials_unlit": {}}, "doubleSided": true}],
                "buffers": [{"byteLength": bin.len()}],
                "bufferViews": views,
                "accessors": accessors
            }),
            &bin,
        )
    }

    #[test]
    fn authored_base_color_uv1_fails_instead_of_sampling_uv0() {
        for texture_info in [
            json!({"index":0,"texCoord":1}),
            json!({"index":0,"extensions":{"KHR_texture_transform":{"texCoord":1}}}),
        ] {
            let bytes = glb_from_json(
                json!({
                    "asset":{"version":"2.0"},
                    "materials":[{"pbrMetallicRoughness":{"baseColorTexture":texture_info}}],
                    "textures":[{"source":0}],
                    "images":[{"uri":"data:image/png;base64,AA=="}]
                }),
                &[],
            );
            let doc = gltf::Gltf::from_slice_without_validation(&bytes)
                .unwrap()
                .document;
            let error = authored_materials(&doc, &[], Path::new("uv1.glb"))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("material 0") && error.contains("TEXCOORD_1"),
                "{error}"
            );
        }
    }

    #[test]
    fn embedded_glb_imports_from_memory() {
        let (doc, buffers, images) =
            import_glb_slice(&minimal_glb(), "embedded-test.glb", &[]).unwrap();
        assert_eq!(doc.scenes().count(), 1);
        assert!(buffers.is_empty());
        assert!(images.is_empty());
    }

    #[test]
    fn unknown_glb_chunks_are_ignored() {
        let (doc, buffers, images) = import_glb_slice(
            &append_unknown_chunk(minimal_glb()),
            "unknown-chunk.glb",
            &[],
        )
        .unwrap();
        assert_eq!(doc.scenes().count(), 1);
        assert!(buffers.is_empty());
        assert!(images.is_empty());
    }

    #[test]
    fn embedded_glb_error_includes_label() {
        let err = import_glb_slice(b"not a glb", "player.glb", &[]).unwrap_err();
        assert!(format!("{err:#}").contains("player.glb"));
    }

    #[test]
    fn byte_loader_rejects_external_buffer_uri() {
        let bytes = glb_from_json(
            json!({
                "asset": {"version": "2.0"},
                "buffers": [{"uri": "relative.bin", "byteLength": 4}],
                "scene": 0,
                "scenes": [{"nodes": []}]
            }),
            &[],
        );
        let err = import_glb_slice(&bytes, "external.glb", &[]).unwrap_err();
        assert!(format!("{err:#}").contains("external buffer URI"));
        assert!(format!("{err:#}").contains("not allowed for byte-loaded GLB"));
    }

    #[test]
    fn data_uri_buffer_imports_from_memory() {
        let bytes = glb_from_json(
            json!({
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": []}],
                "buffers": [{"uri": "data:application/octet-stream;base64,AQIDBA==", "byteLength": 4}]
            }),
            &[],
        );
        let (_, buffers, _) = import_glb_slice(&bytes, "data-buffer.glb", &[]).unwrap();
        assert_eq!(&buffers[0].0[..4], &[1, 2, 3, 4]);
    }

    #[test]
    fn data_uri_image_imports_from_memory() {
        let (_, _, images) =
            import_glb_slice(&data_uri_image_glb(2, 1), "data-image.glb", &[]).unwrap();
        assert_eq!((images[0].width, images[0].height), (2, 1));
        assert_eq!(images[0].format, gltf::image::Format::R8G8B8A8);
    }

    #[test]
    fn bytes_image_decode_preserves_downsampling_behavior() {
        let (_, _, images) = import_glb_slice_with_options(
            &data_uri_image_glb(2, 1),
            "limited-image.glb",
            &[],
            Some(1),
        )
        .unwrap();
        let (pixels, width, height) = cap_texture_rgba(
            to_rgba8(&images[0]),
            images[0].width,
            images[0].height,
            Some(1),
        );
        assert_eq!((width, height), (1, 1));
        assert_eq!(pixels.len(), 4);
    }

    #[test]
    fn byte_image_budget_rejects_cumulative_decoded_bytes() {
        let mut budget = ByteImageBudget::default();
        for image_index in 0..(MAX_BYTE_IMAGE_BYTES / (4096 * 4096 * 4)) as usize {
            budget
                .reserve(image_index, 4096, 4096, "budget.glb")
                .unwrap();
        }
        let err = budget.reserve(4, 4096, 4096, "budget.glb").unwrap_err();
        assert!(format!("{err:#}").contains("decoded image byte budget"));
    }

    #[test]
    fn byte_image_budget_rejects_excessive_image_count() {
        let mut budget = ByteImageBudget::default();
        for image_index in 0..MAX_BYTE_IMAGE_COUNT {
            budget.reserve(image_index, 1, 1, "count.glb").unwrap();
        }
        let err = budget
            .reserve(MAX_BYTE_IMAGE_COUNT, 1, 1, "count.glb")
            .unwrap_err();
        assert!(format!("{err:#}").contains("decoded images"));
    }

    #[test]
    fn oversized_joint_palette_returns_loader_error() {
        let err = validate_joint_palette_count(513, Path::new("oversized-rig.glb")).unwrap_err();
        assert!(format!("{err:#}").contains("combined skin joint count 513"));
        assert!(format!("{err:#}").contains("limit 512"));
    }

    #[test]
    fn caller_validated_required_extension_is_tolerated_without_disabling_core_validation() {
        let bytes = glb_from_json(
            json!({
                "asset": {"version": "2.0"},
                "extensionsUsed": ["X_TEST_VALIDATED"],
                "extensionsRequired": ["X_TEST_VALIDATED"],
                "scene": 0,
                "scenes": [{"nodes": []}]
            }),
            &[],
        );
        let (doc, buffers, images) = import_glb_slice(
            &bytes,
            "validated-extension.glb",
            &["X_TEST_VALIDATED".to_owned()],
        )
        .unwrap();
        assert_eq!(doc.scenes().count(), 1);
        assert!(buffers.is_empty());
        assert!(images.is_empty());
    }

    #[test]
    fn unknown_required_extension_is_rejected() {
        let bytes = glb_from_json(
            json!({
                "asset": {"version": "2.0"},
                "extensionsRequired": ["X_TEST_UNKNOWN"],
                "scene": 0,
                "scenes": [{"nodes": []}]
            }),
            &[],
        );
        let err = import_glb_slice(&bytes, "unknown-extension.glb", &[]).unwrap_err();
        assert!(format!("{err:#}").contains("required extension X_TEST_UNKNOWN"));
        assert!(format!("{err:#}").contains("not enabled or caller-validated"));
    }

    #[test]
    fn odd_texture_dimensions_are_capped_by_max_texture_dim() {
        let rgba = vec![255u8; 5 * 3 * 4];
        let (rgba, width, height) = cap_texture_rgba(rgba, 5, 3, Some(2));
        assert_eq!((width, height), (2, 1));
        assert_eq!(rgba.len(), (width * height * 4) as usize);
    }

    fn validate_fixture(bytes: &[u8]) -> anyhow::Error {
        let (doc, buffers, _) = import_glb_slice(bytes, "malformed.glb", &[]).unwrap();
        validate_model_input(&doc, &buffers, Path::new("malformed.glb")).unwrap_err()
    }

    #[test]
    fn short_normal_accessor_returns_loader_error() {
        let err = validate_fixture(&mesh_fixture(Some(3), None, None, None, None));
        assert!(format!("{err:#}").contains("NORMAL accessor count 3"));
        assert!(format!("{err:#}").contains("POSITION count 4"));
    }

    #[test]
    fn short_joints_accessor_returns_loader_error() {
        let err = validate_fixture(&mesh_fixture(None, Some(3), None, None, None));
        assert!(format!("{err:#}").contains("JOINTS_0 accessor count 3"));
        assert!(format!("{err:#}").contains("POSITION count 4"));
    }

    #[test]
    fn short_weights_accessor_returns_loader_error() {
        let err = validate_fixture(&mesh_fixture(None, None, Some(3), None, None));
        assert!(format!("{err:#}").contains("WEIGHTS_0 accessor count 3"));
        assert!(format!("{err:#}").contains("POSITION count 4"));
    }

    #[test]
    fn morph_target_count_mismatch_returns_loader_error() {
        let err = validate_fixture(&mesh_fixture(None, None, None, Some(3), None));
        assert!(format!("{err:#}").contains("morph POSITION target 0 accessor count 3"));
        assert!(format!("{err:#}").contains("POSITION count 4"));
    }

    #[test]
    fn zero_animation_keys_return_loader_error() {
        let err = validate_fixture(&animation_fixture(0, 1, 3, "translation"));
        assert!(format!("{err:#}").contains("has zero input keys"));
    }

    #[test]
    fn animation_input_output_mismatch_returns_loader_error() {
        let err = validate_fixture(&animation_fixture(2, 1, 3, "translation"));
        assert!(format!("{err:#}").contains("animation channel output count 1"));
        assert!(format!("{err:#}").contains("input key count 2"));
    }

    #[test]
    fn duplicate_animation_times_return_loader_error() {
        let err = validate_fixture(&animation_fixture_with_duplicate_times());
        assert!(format!("{err:#}").contains("invalid or unsorted input keys"));
    }

    #[test]
    fn cubic_spline_requires_two_keys() {
        let err = validate_fixture(&animation_fixture_with_interpolation(
            1,
            3,
            3,
            "translation",
            "CUBICSPLINE",
        ));
        assert!(format!("{err:#}").contains("fewer than two CUBICSPLINE keys"));
    }

    #[test]
    fn zero_rotation_quaternion_returns_loader_error() {
        let err = validate_fixture(&zero_rotation_animation_fixture());
        assert!(format!("{err:#}").contains("zero or non-finite rotation quaternion"));
    }

    #[test]
    fn malformed_rotation_output_returns_loader_error() {
        let err = validate_fixture(&animation_fixture(1, 1, 3, "rotation"));
        assert!(format!("{err:#}").contains("output dimensions incompatible with its target"));
    }

    #[test]
    fn valid_animation_fixture_passes_loader_validation() {
        let bytes = animation_fixture(2, 2, 3, "translation");
        let (doc, buffers, _) = import_glb_slice(&bytes, "valid-animation.glb", &[]).unwrap();
        validate_model_input(&doc, &buffers, Path::new("valid-animation.glb")).unwrap();
    }

    #[test]
    fn bytes_options_and_unlit_material_load_into_existing_primitive_state() {
        let gpu = Gpu::new_headless().expect("headless GPU is required for this loader test");
        let samplers = Samplers::new(&gpu);
        let layout = ModelAsset::material_layout(&gpu);
        let opts = ModelLoadOptions {
            max_texture_dim: Some(4),
            ..Default::default()
        };
        let asset = ModelAsset::load_glb_bytes_opts_with_allowed_required_extensions(
            &gpu,
            &layout,
            &samplers,
            &unlit_fixture(),
            "unlit.glb",
            &opts,
            ["X_TEST_VALIDATED"],
        )
        .unwrap();
        assert_eq!(asset.primitives.len(), 1);
        assert!(asset.primitives[0].unlit);
        assert!(asset.primitives[0].double_sided);
    }

    #[test]
    fn named_node_lookup_returns_first_duplicate_in_index_order() {
        let names = vec![
            Some("root".to_owned()),
            Some("hand.R".to_owned()),
            None,
            Some("hand.R".to_owned()),
        ];
        assert_eq!(find_node_named(&names, "root"), Some(0));
        assert_eq!(find_node_named(&names, "hand.R"), Some(1));
        assert_eq!(find_node_named(&names, "missing"), None);
        assert_eq!(find_node_named(&[], "root"), None);
    }

    #[test]
    fn sampled_node_transform_accumulates_animated_ancestors() {
        let skeleton = Skeleton {
            parents: vec![usize::MAX, 0, 1],
            rest: vec![
                NodeTrs {
                    translation: Vec3::X,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                },
                NodeTrs {
                    translation: Vec3::Y * 2.0,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                },
                NodeTrs {
                    translation: Vec3::Z * 3.0,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                },
            ],
            order: vec![0, 1, 2],
        };
        let clips = vec![Clip {
            name: "Reach".to_owned(),
            duration: 1.0,
            channels: vec![Channel {
                node: 1,
                path: ChannelPath::Translation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![0.0, 2.0, 0.0, 0.0, 4.0, 0.0],
            }],
        }];
        let anim = AnimState {
            clip: 0,
            time: 0.5,
            speed: 1.0,
            looping: false,
        };

        let socket = sample_node_transform(&skeleton, &clips, 2, &anim).unwrap();
        let translation = socket.transform_point3(Vec3::ZERO);
        assert!(translation.distance(Vec3::new(1.0, 3.0, 3.0)) < 1e-6);
        assert!(sample_node_transform(&skeleton, &clips, 3, &anim).is_none());
        assert!(sample_node_transform(&skeleton, &clips, usize::MAX, &anim).is_none());
    }

    #[test]
    fn material_role_reads_pocket3d_extras() {
        assert_eq!(
            pocket3d_role_from_extras(r#"{"pocket3d_role":"dynamic_screen","unrelated":true}"#)
                .as_deref(),
            Some("dynamic_screen")
        );
        assert_eq!(pocket3d_role_from_extras("{}"), None);
        assert_eq!(pocket3d_role_from_extras("not json"), None);
    }

    #[test]
    fn explicit_role_wins_over_name_prefix_fallback() {
        assert!(semantic_material_matches(
            Some("dynamic_screen"),
            Some("anything"),
            "dynamic_screen",
            Some("P3D_dynamic_screen__"),
        ));
        assert!(semantic_material_matches(
            None,
            Some("P3D_dynamic_screen__panel"),
            "dynamic_screen",
            Some("P3D_dynamic_screen__"),
        ));
        assert!(!semantic_material_matches(
            Some("different_role"),
            Some("P3D_dynamic_screen__panel"),
            "dynamic_screen",
            Some("P3D_dynamic_screen__"),
        ));
    }

    #[test]
    fn material_uniform_has_portable_uniform_alignment() {
        assert_eq!(std::mem::size_of::<MaterialRaw>(), 48);
        assert_eq!(std::mem::align_of::<MaterialRaw>(), 4);
    }

    #[test]
    fn material_base_color_mode_reads_pocket3d_extras() {
        assert_eq!(
            pocket3d_base_color_mode_from_extras(r#"{"pocket3d_base_color_mode":"monochrome"}"#,),
            MaterialBaseColorMode::Monochrome
        );
        assert_eq!(
            pocket3d_base_color_mode_from_extras(r#"{"pocket3d_base_color_mode":"unknown"}"#),
            MaterialBaseColorMode::Authored
        );
        assert_eq!(
            pocket3d_base_color_mode_from_extras("not json"),
            MaterialBaseColorMode::Authored
        );
    }

    #[test]
    fn texture_cache_key_uses_exact_pixels_and_dimensions() {
        let key = |width, height, rgba: &[u8], color_space| ModelTextureCacheKey {
            width,
            height,
            rgba: rgba.into(),
            color_space,
        };
        let reference = key(2, 1, &[1, 2, 3, 4, 5, 6, 7, 8], TextureColorSpace::Srgb);
        assert!(reference == key(2, 1, &[1, 2, 3, 4, 5, 6, 7, 8], TextureColorSpace::Srgb));
        assert!(reference != key(1, 2, &[1, 2, 3, 4, 5, 6, 7, 8], TextureColorSpace::Srgb));
        assert!(reference != key(2, 1, &[1, 2, 3, 4, 5, 6, 7, 9], TextureColorSpace::Srgb));
        assert!(reference != key(2, 1, &[1, 2, 3, 4, 5, 6, 7, 8], TextureColorSpace::Linear));
    }

    #[test]
    fn gray_alpha_base_color_expands_to_rgba() {
        let image = gltf::image::Data {
            pixels: vec![32, 0, 180, 255],
            format: gltf::image::Format::R8G8,
            width: 2,
            height: 1,
        };
        assert_eq!(to_rgba8(&image), vec![32, 32, 32, 0, 180, 180, 180, 255]);
    }

    #[test]
    fn live_surface_texcoord_validation_accepts_normalized_full_span() {
        let texcoords = [[-0.005, 0.0], [1.005, 0.0], [1.0, 1.0], [0.0, 1.0]];
        validate_normalized_texcoord0(
            Some(&texcoords),
            texcoords.len(),
            Some("screen"),
            Path::new("device.glb"),
        )
        .unwrap();
    }

    #[test]
    fn live_surface_texcoord_validation_rejects_missing_or_invalid_uvs() {
        let path = Path::new("device.glb");
        assert!(validate_normalized_texcoord0(None, 4, Some("screen"), path).is_err());
        assert!(
            validate_normalized_texcoord0(
                Some(&[[0.0, 0.0], [f32::NAN, 1.0]]),
                2,
                Some("screen"),
                path,
            )
            .is_err()
        );
        assert!(
            validate_normalized_texcoord0(
                Some(&[[0.0, 0.0], [1.1, 1.0]]),
                2,
                Some("screen"),
                path,
            )
            .is_err()
        );
        assert!(
            validate_normalized_texcoord0(
                Some(&[[0.1, 0.1], [0.2, 0.2]]),
                2,
                Some("screen"),
                path,
            )
            .is_err()
        );
    }
}
