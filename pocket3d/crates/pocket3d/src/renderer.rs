//! Forward renderer. One `Renderer` owns all pipelines; each frame it draws
//! a `Scene` (3D) then a `Hud` (2D overlay) into any color view.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::camera::Camera;
use crate::gpu::{DEPTH_FORMAT, DepthTarget, Gpu};
use crate::hud::{ATLAS_H, ATLAS_W, Hud, HudVertex, build_font_atlas};
use crate::model::{ModelAsset, ModelInstance, ModelVertex};
use crate::scene::Scene;
use crate::texture::{GpuTexture, Samplers, create_rgba_texture};
use crate::world::{WorldBatchKind, WorldVertex};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalsRaw {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
}

pub struct Renderer {
    pub color_format: wgpu::TextureFormat,
    pub samplers: Samplers,
    pub world_material_layout: wgpu::BindGroupLayout,
    pub model_material_layout: wgpu::BindGroupLayout,
    globals_layout: wgpu::BindGroupLayout,
    depth: Option<DepthTarget>,
    msaa_color: Option<MsaaColorTarget>,
    smaa: Option<crate::smaa::SmaaPass>,
    requested_sample_count: u32,
    effective_sample_count: u32,
    supported_sample_counts: Vec<u32>,
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    world_opaque: wgpu::RenderPipeline,
    world_alphatest: wgpu::RenderPipeline,
    world_sky: wgpu::RenderPipeline,
    models: ModelPass,
    sprites: SpritePass,
    hud: HudPass,
}

/// Generic renderer initialization settings. Product settings should be
/// translated to this representation at the application boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RendererConfig {
    pub requested_sample_count: u32,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            requested_sample_count: 1,
        }
    }
}

const CANDIDATE_SAMPLE_COUNTS: [u32; 4] = [1, 2, 4, 8];

fn sanitize_requested_sample_count(requested: u32) -> u32 {
    match requested {
        1 | 2 | 4 | 8 => requested,
        _ => 1,
    }
}

/// Select the highest supported candidate that does not exceed the request.
/// The caller supplies the already-intersected hardware capabilities so this
/// pure policy is testable without manufacturing a GPU adapter.
pub fn select_effective_sample_count(requested: u32, supported: &[u32]) -> u32 {
    let requested = sanitize_requested_sample_count(requested);

    CANDIDATE_SAMPLE_COUNTS
        .iter()
        .rev()
        .copied()
        .find(|&count| count <= requested && supported.contains(&count))
        .unwrap_or(1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SampleCountTransition {
    requested_sample_count: u32,
    effective_sample_count: u32,
    rebuild_pipelines: bool,
}

fn sample_count_transition(
    requested: u32,
    current_effective: u32,
    supported: &[u32],
) -> SampleCountTransition {
    let requested_sample_count = sanitize_requested_sample_count(requested);
    let effective_sample_count = select_effective_sample_count(requested_sample_count, supported);
    SampleCountTransition {
        requested_sample_count,
        effective_sample_count,
        rebuild_pipelines: effective_sample_count != current_effective,
    }
}

struct MsaaColorTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
    sample_count: u32,
}

impl MsaaColorTarget {
    fn new(gpu: &Gpu, format: wgpu::TextureFormat, size: (u32, u32), sample_count: u32) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("msaa color"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
            size,
            sample_count,
        }
    }
}

fn multisample_state(sample_count: u32) -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: sample_count,
        mask: !0,
        alpha_to_coverage_enabled: false,
    }
}

fn supported_sample_counts(gpu: &Gpu, color_format: wgpu::TextureFormat) -> Vec<u32> {
    let color_features = texture_format_features(gpu, color_format);
    let depth_features = texture_format_features(gpu, DEPTH_FORMAT);
    let color_renderable = color_features
        .allowed_usages
        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT);
    let depth_renderable = depth_features
        .allowed_usages
        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT);
    let can_resolve_color = color_features
        .flags
        .contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE);

    let supported: Vec<u32> = CANDIDATE_SAMPLE_COUNTS
        .into_iter()
        .filter(|&count| {
            color_renderable
                && depth_renderable
                && color_features.flags.sample_count_supported(count)
                && depth_features.flags.sample_count_supported(count)
                && (count == 1 || can_resolve_color)
        })
        .collect();

    log::debug!(
        "MSAA capabilities: color {color_format:?} {:?}, depth {DEPTH_FORMAT:?} {:?}, supported {:?}",
        color_features.flags,
        depth_features.flags,
        supported
    );
    supported
}

fn texture_format_features(gpu: &Gpu, format: wgpu::TextureFormat) -> wgpu::TextureFormatFeatures {
    if gpu
        .device
        .features()
        .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
    {
        gpu.adapter.get_texture_format_features(format)
    } else {
        format.guaranteed_format_features(gpu.device.features())
    }
}

fn create_world_pipelines(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    globals_layout: &wgpu::BindGroupLayout,
    material_layout: &wgpu::BindGroupLayout,
    sample_count: u32,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("world.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/world.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("world layout"),
        bind_group_layouts: &[globals_layout, material_layout],
        push_constant_ranges: &[],
    });
    let make_pipeline = |label: &str, fs_entry: &str| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[WorldVertex::LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fs_entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: multisample_state(sample_count),
            multiview: None,
            cache: None,
        })
    };
    (
        make_pipeline("world opaque", "fs_opaque"),
        make_pipeline("world alphatest", "fs_alphatest"),
        make_pipeline("world sky", "fs_sky"),
    )
}

impl Renderer {
    pub fn new(gpu: &Gpu, color_format: wgpu::TextureFormat) -> Result<Self> {
        Self::new_with_internal_config(gpu, color_format, RendererConfig::default(), false)
    }

    pub fn new_with_config(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        config: RendererConfig,
    ) -> Result<Self> {
        Self::new_with_internal_config(gpu, color_format, config, false)
    }

    /// Initializes the renderer with the internal SMAA proof-of-concept path.
    ///
    /// This is intentionally crate-private: SMAA is a renderer experiment and
    /// is not part of the public RendererConfig API yet.
    #[allow(dead_code)]
    pub(crate) fn new_with_smaa_for_poc(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
    ) -> Result<Self> {
        Self::new_with_internal_config(gpu, color_format, RendererConfig::default(), true)
    }

    fn new_with_internal_config(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        config: RendererConfig,
        smaa_enabled: bool,
    ) -> Result<Self> {
        let device = &gpu.device;
        let supported_sample_counts = supported_sample_counts(gpu, color_format);
        let requested_sample_count = sanitize_requested_sample_count(config.requested_sample_count);
        let effective_sample_count =
            select_effective_sample_count(requested_sample_count, &supported_sample_counts);
        log::info!(
            "requested AA: {}x, effective MSAA: {}x",
            requested_sample_count,
            effective_sample_count
        );
        let samplers = Samplers::new(gpu);

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<GlobalsRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals bg"),
            layout: &globals_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        });

        // --- world pipelines ---------------------------------------------
        let world_material_layout = crate::world::WorldModel::material_layout(gpu);
        let (world_opaque, world_alphatest, world_sky) = create_world_pipelines(
            device,
            color_format,
            &globals_bgl,
            &world_material_layout,
            effective_sample_count,
        );

        let models = ModelPass::new(gpu, color_format, &globals_bgl, effective_sample_count);
        let model_material_layout = models.material_layout.clone();
        let sprites = SpritePass::new(gpu, color_format, &globals_bgl, effective_sample_count);
        let smaa = smaa_enabled.then(|| crate::smaa::SmaaPass::new(gpu, color_format));
        if smaa.is_some() {
            log::info!("post-process AA: SMAA 1x");
        }

        Ok(Self {
            color_format,
            samplers,
            world_material_layout,
            model_material_layout,
            globals_layout: globals_bgl,
            depth: None,
            msaa_color: None,
            smaa,
            requested_sample_count,
            effective_sample_count,
            supported_sample_counts,
            globals_buf,
            globals_bg,
            world_opaque,
            world_alphatest,
            world_sky,
            models,
            sprites,
            hud: HudPass::new(gpu, color_format),
        })
    }

    /// Ensure the internal targets match the current effective sample count.
    fn ensure_targets(&mut self, gpu: &Gpu, size: (u32, u32)) {
        let depth_needs_recreate = self.depth.as_ref().is_none_or(|depth| {
            depth.size != size || depth.sample_count != self.effective_sample_count
        });
        if depth_needs_recreate {
            self.depth = Some(DepthTarget::new_with_sample_count(
                gpu,
                size.0,
                size.1,
                self.effective_sample_count,
            ));
        }

        if self.effective_sample_count > 1 {
            let msaa_needs_recreate = self.msaa_color.as_ref().is_none_or(|target| {
                target.size != size || target.sample_count != self.effective_sample_count
            });
            if msaa_needs_recreate {
                self.msaa_color = Some(MsaaColorTarget::new(
                    gpu,
                    self.color_format,
                    size,
                    self.effective_sample_count,
                ));
            }
        } else {
            self.msaa_color = None;
        }

        if let Some(smaa) = self.smaa.as_mut() {
            smaa.ensure_targets(gpu, size);
        }
    }

    /// Request a new generic MSAA sample count between frames.
    ///
    /// The request is sanitized and resolved against the color/depth
    /// capabilities discovered at construction. If the effective count
    /// changes, only the sample-count-dependent pipelines and internal
    /// attachments are replaced. Callers must invoke this outside any active
    /// render pass, before the next call to [`Self::render`].
    pub fn set_requested_sample_count(&mut self, gpu: &Gpu, requested: u32) -> u32 {
        let transition = sample_count_transition(
            requested,
            self.effective_sample_count,
            &self.supported_sample_counts,
        );
        let requested_changed = transition.requested_sample_count != self.requested_sample_count;
        if requested_changed {
            log::info!(
                "requested AA changed: {}x -> {}x",
                self.requested_sample_count,
                transition.requested_sample_count
            );
            self.requested_sample_count = transition.requested_sample_count;
        }

        if !transition.rebuild_pipelines {
            if requested_changed {
                log::info!("effective MSAA remains {}x", self.effective_sample_count);
            }
            return self.effective_sample_count;
        }

        let previous_effective_sample_count = self.effective_sample_count;
        let (world_opaque, world_alphatest, world_sky) = create_world_pipelines(
            &gpu.device,
            self.color_format,
            &self.globals_layout,
            &self.world_material_layout,
            transition.effective_sample_count,
        );
        self.models.rebuild_pipeline(
            &gpu.device,
            self.color_format,
            &self.globals_layout,
            transition.effective_sample_count,
        );
        self.sprites.rebuild_pipeline(
            &gpu.device,
            self.color_format,
            &self.globals_layout,
            transition.effective_sample_count,
        );
        self.world_opaque = world_opaque;
        self.world_alphatest = world_alphatest;
        self.world_sky = world_sky;
        self.effective_sample_count = transition.effective_sample_count;

        // The next render lazily creates targets at the current size. This
        // handles 1x <-> Nx transitions and resize-before-reconfigure without
        // allocating targets that may never be used.
        self.depth = None;
        self.msaa_color = None;
        log::info!(
            "effective MSAA changed: {}x -> {}x",
            previous_effective_sample_count,
            self.effective_sample_count
        );
        self.effective_sample_count
    }

    pub fn requested_sample_count(&self) -> u32 {
        self.requested_sample_count
    }

    pub fn effective_sample_count(&self) -> u32 {
        self.effective_sample_count
    }

    pub fn render(
        &mut self,
        gpu: &Gpu,
        color_view: &wgpu::TextureView,
        size: (u32, u32),
        scene: &Scene,
        camera: &Camera,
        hud: &Hud,
    ) {
        self.ensure_targets(gpu, size);
        if let Some(smaa) = self.smaa.as_mut() {
            smaa.update_metrics(gpu, size, scene.transparent_clear);
        }
        let aspect = size.0 as f32 / size.1 as f32;

        let globals = GlobalsRaw {
            view_proj: camera.view_proj(aspect).to_cols_array_2d(),
            cam_pos: camera.pos.extend(scene.time).to_array(),
            sky_zenith: scene.sky.zenith.extend(1.0).to_array(),
            sky_horizon: scene.sky.horizon.extend(1.0).to_array(),
            sun_dir: scene.sky.sun_dir.extend(0.0).to_array(),
            sun_color: scene.sky.sun_color.extend(1.0).to_array(),
        };
        gpu.queue
            .write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        // Upload per-instance data (joint palettes etc.) before recording.
        let (model_draws, viewmodel_draw) = self.models.prepare(gpu, scene);
        let sprite_verts = self.sprites.prepare(gpu, scene, camera);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // --- 3D scene pass ------------------------------------------------
        {
            let h = scene.sky.horizon;
            let depth_view = &self.depth.as_ref().unwrap().view;
            let scene_color_view = if let Some(msaa_color) = self.msaa_color.as_ref() {
                &msaa_color.view
            } else if let Some(smaa) = self.smaa.as_ref() {
                smaa.scene_view()
            } else {
                color_view
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if scene.transparent_clear {
                            wgpu::Color::TRANSPARENT
                        } else {
                            wgpu::Color {
                                r: h.x as f64,
                                g: h.y as f64,
                                b: h.z as f64,
                                a: 1.0,
                            }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_bind_group(0, &self.globals_bg, &[]);

            if let Some(world) = &scene.world {
                pass.set_vertex_buffer(0, world.vbuf.slice(..));
                pass.set_index_buffer(world.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                for kind in [
                    WorldBatchKind::Opaque,
                    WorldBatchKind::AlphaTest,
                    WorldBatchKind::Sky,
                ] {
                    let pipeline = match kind {
                        WorldBatchKind::Opaque => &self.world_opaque,
                        WorldBatchKind::AlphaTest => &self.world_alphatest,
                        WorldBatchKind::Sky => &self.world_sky,
                    };
                    let mut bound = false;
                    for batch in world.batches.iter().filter(|b| b.kind == kind) {
                        if !bound {
                            pass.set_pipeline(pipeline);
                            bound = true;
                        }
                        pass.set_bind_group(1, &batch.bind_group, &[]);
                        pass.draw_indexed(
                            batch.first_index..batch.first_index + batch.index_count,
                            0,
                            0..1,
                        );
                    }
                }
            }

            self.models.draw(&mut pass, &model_draws);
            self.sprites.draw(&mut pass, sprite_verts);
        }

        // --- viewmodel pass (own depth range so the gun never clips) -------
        if let Some(vm) = &viewmodel_draw {
            let depth_view = &self.depth.as_ref().unwrap().view;
            let scene_color_view = if let Some(msaa_color) = self.msaa_color.as_ref() {
                &msaa_color.view
            } else if let Some(smaa) = self.smaa.as_ref() {
                smaa.scene_view()
            } else {
                color_view
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewmodel"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: scene_color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.globals_bg, &[]);
            self.models.draw(&mut pass, std::slice::from_ref(vm));
        }

        // Resolve the complete 3D image only after both 3D passes. The final
        // target remains single-sampled so HUD and application overlays keep
        // their existing 1x path.
        if let Some(msaa_color) = &self.msaa_color {
            let resolve_target = self
                .smaa
                .as_ref()
                .map(|smaa| smaa.scene_view())
                .unwrap_or(color_view);
            let _resolve_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("msaa resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &msaa_color.view,
                    resolve_target: Some(resolve_target),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        if let Some(smaa) = self.smaa.as_ref() {
            smaa.render(&mut encoder, color_view);
        }

        // --- HUD overlay pass ----------------------------------------------
        if !hud.verts.is_empty() {
            self.hud.upload(gpu, hud, size);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hud"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.hud.draw(&mut pass, hud);
        }

        gpu.queue.submit([encoder.finish()]);
    }
}

// ---------------------------------------------------------------------------
// Model pass (skinned + static, dynamic-offset instance/joint buffers)
// ---------------------------------------------------------------------------

const INSTANCE_STRIDE: u64 = 256;
const JOINT_ALIGN: u64 = 256;
/// Fixed window each draw binds from the joints buffer: 512 mat4s (32 KB —
/// VRoid-style humanoid rigs carry ~270 joints across their skins).
const JOINT_WINDOW: u64 = 512 * 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    tint: [f32; 4],
    params: [f32; 4],
}

pub(crate) struct ModelDraw {
    asset: std::sync::Arc<ModelAsset>,
    inst_offset: u32,
    joints_offset: u32,
    /// The instance's morph overlay buffer (wgpu buffers are ref-counted).
    morph: Option<wgpu::Buffer>,
}

struct ModelPass {
    pipeline: wgpu::RenderPipeline,
    material_layout: wgpu::BindGroupLayout,
    object_layout: wgpu::BindGroupLayout,
    object_bg: wgpu::BindGroup,
    instance_buf: wgpu::Buffer,
    joints_buf: wgpu::Buffer,
    instance_capacity: u64,
    joints_capacity: u64,
}

impl ModelPass {
    fn new(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        globals_bgl: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        let device = &gpu.device;
        let material_layout = ModelAsset::material_layout(gpu);
        let object_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model object"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline = Self::create_pipeline(
            device,
            color_format,
            globals_bgl,
            &material_layout,
            &object_layout,
            sample_count,
        );

        let instance_capacity = 64 * INSTANCE_STRIDE;
        let joints_capacity = 256 * 1024;
        let instance_buf = Self::make_instance_buf(device, instance_capacity);
        let joints_buf = Self::make_joints_buf(device, joints_capacity);
        let object_bg = Self::make_object_bg(device, &object_layout, &instance_buf, &joints_buf);

        Self {
            pipeline,
            material_layout,
            object_layout,
            object_bg,
            instance_buf,
            joints_buf,
            instance_capacity,
            joints_capacity,
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        globals_layout: &wgpu::BindGroupLayout,
        material_layout: &wgpu::BindGroupLayout,
        object_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("model.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/model.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("model layout"),
            bind_group_layouts: &[globals_layout, material_layout, object_layout],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("model pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[ModelVertex::LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: multisample_state(sample_count),
            multiview: None,
            cache: None,
        })
    }

    fn rebuild_pipeline(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        globals_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) {
        self.pipeline = Self::create_pipeline(
            device,
            color_format,
            globals_layout,
            &self.material_layout,
            &self.object_layout,
            sample_count,
        );
    }

    fn make_instance_buf(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model instances"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_joints_buf(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model joints"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_object_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        instances: &wgpu::Buffer,
        joints: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model object bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: instances,
                        offset: 0,
                        size: wgpu::BufferSize::new(INSTANCE_STRIDE),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: joints,
                        offset: 0,
                        size: wgpu::BufferSize::new(JOINT_WINDOW),
                    }),
                },
            ],
        })
    }

    /// Compute palettes + instance data for everything in the scene and
    /// upload once. Returns draw entries (scene models, viewmodel).
    fn prepare(&mut self, gpu: &Gpu, scene: &Scene) -> (Vec<ModelDraw>, Option<ModelDraw>) {
        let all: Vec<&ModelInstance> = scene.models.iter().chain(scene.viewmodel.iter()).collect();
        if all.is_empty() {
            return (Vec::new(), None);
        }

        let mut inst_bytes = vec![0u8; all.len() * INSTANCE_STRIDE as usize];
        let mut joint_bytes: Vec<u8> = Vec::with_capacity(all.len() * 64 * 4);
        let mut draws = Vec::with_capacity(all.len());
        let mut palette: Vec<Mat4> = Vec::new();

        for (i, inst) in all.iter().enumerate() {
            let raw = InstanceRaw {
                model: inst.transform.to_cols_array_2d(),
                tint: inst.tint,
                params: [inst.lit, inst.cutout, 0.0, 0.0],
            };
            let off = i * INSTANCE_STRIDE as usize;
            inst_bytes[off..off + std::mem::size_of::<InstanceRaw>()]
                .copy_from_slice(bytemuck::bytes_of(&raw));

            if let Some(morph) = &inst.morph {
                morph.upload_if_dirty(gpu, &inst.asset);
            }
            match &inst.pose {
                Some(globals) => inst.asset.palette_from_globals(globals, &mut palette),
                None => inst.asset.joint_palette(&inst.anim, &mut palette),
            }
            if palette.len() as u64 * 64 > JOINT_WINDOW {
                log::warn!(
                    "model has {} joints; truncating to {}",
                    palette.len(),
                    JOINT_WINDOW / 64
                );
                palette.truncate((JOINT_WINDOW / 64) as usize);
            }
            let joints_offset = joint_bytes.len() as u32;
            for m in &palette {
                joint_bytes.extend_from_slice(bytemuck::cast_slice(&m.to_cols_array()));
            }
            // Align the next palette.
            let pad = (JOINT_ALIGN as usize - joint_bytes.len() % JOINT_ALIGN as usize)
                % JOINT_ALIGN as usize;
            joint_bytes.extend(std::iter::repeat_n(0u8, pad));

            draws.push(ModelDraw {
                asset: inst.asset.clone(),
                inst_offset: off as u32,
                joints_offset,
                morph: inst.morph.as_ref().map(|m| m.buffer().clone()),
            });
        }

        // Every dynamic offset must leave a full JOINT_WINDOW in range.
        if let Some(last) = draws.last() {
            let need = last.joints_offset as usize + JOINT_WINDOW as usize;
            if joint_bytes.len() < need {
                joint_bytes.resize(need, 0);
            }
        }

        // Grow buffers if needed (recreates the bind group).
        let device = &gpu.device;
        let mut recreate = false;
        if inst_bytes.len() as u64 > self.instance_capacity {
            self.instance_capacity = (inst_bytes.len() as u64).next_power_of_two();
            self.instance_buf = Self::make_instance_buf(device, self.instance_capacity);
            recreate = true;
        }
        if joint_bytes.len() as u64 > self.joints_capacity {
            self.joints_capacity = (joint_bytes.len() as u64).next_power_of_two();
            self.joints_buf = Self::make_joints_buf(device, self.joints_capacity);
            recreate = true;
        }
        if recreate {
            self.object_bg = Self::make_object_bg(
                device,
                &self.object_layout,
                &self.instance_buf,
                &self.joints_buf,
            );
        }
        gpu.queue.write_buffer(&self.instance_buf, 0, &inst_bytes);
        if !joint_bytes.is_empty() {
            gpu.queue.write_buffer(&self.joints_buf, 0, &joint_bytes);
        }

        let viewmodel = scene.viewmodel.is_some().then(|| draws.pop()).flatten();
        (draws, viewmodel)
    }

    fn draw<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, draws: &'p [ModelDraw]) {
        if draws.is_empty() {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        for d in draws {
            pass.set_bind_group(2, &self.object_bg, &[d.inst_offset, d.joints_offset]);
            pass.set_vertex_buffer(0, d.asset.vbuf.slice(..));
            pass.set_index_buffer(d.asset.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            let mut overlay_bound = false;
            for (pi, prim) in d.asset.primitives.iter().enumerate() {
                pass.set_bind_group(1, &prim.bind_group, &[]);
                // Morphing primitives read vertices from the instance's
                // overlay buffer; base_vertex redirects the shared indices.
                let morph = d
                    .morph
                    .as_ref()
                    .zip(d.asset.prim_morph.get(pi).copied().flatten());
                match morph {
                    Some((buf, (mi, pj))) => {
                        let mp = &d.asset.morph_meshes[mi].prims[pj];
                        if !overlay_bound {
                            pass.set_vertex_buffer(0, buf.slice(..));
                            overlay_bound = true;
                        }
                        pass.draw_indexed(
                            prim.first_index..prim.first_index + prim.index_count,
                            mp.overlay_offset as i32 - mp.vertex_base as i32,
                            0..1,
                        );
                    }
                    None => {
                        if overlay_bound {
                            pass.set_vertex_buffer(0, d.asset.vbuf.slice(..));
                            overlay_bound = false;
                        }
                        pass.draw_indexed(
                            prim.first_index..prim.first_index + prim.index_count,
                            0,
                            0..1,
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sprite pass (additive billboards + beams)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpriteVertex {
    pos: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

struct SpritePass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    vbuf: wgpu::Buffer,
    vbuf_capacity: u64,
    #[allow(dead_code)]
    texture: GpuTexture,
}

impl SpritePass {
    fn new(
        gpu: &Gpu,
        color_format: wgpu::TextureFormat,
        globals_bgl: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        let device = &gpu.device;

        // Soft radial glow texture.
        let size = 64u32;
        let mut px = vec![0u8; (size * size * 4) as usize];
        for y in 0..size {
            for x in 0..size {
                let dx = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let dy = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let r = (dx * dx + dy * dy).sqrt().min(1.0);
                let a = ((1.0 - r).powf(2.0) * 255.0) as u8;
                let i = ((y * size + x) * 4) as usize;
                px[i..i + 4].copy_from_slice(&[255, 255, 255, a]);
            }
        }
        let texture = create_rgba_texture(gpu, "sprite glow", size, size, &px, false, true);

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sprite bgl"),
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
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sprite sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let pipeline = Self::create_pipeline(device, color_format, globals_bgl, &bgl, sample_count);

        let vbuf_capacity = 64 * 1024;
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite vbuf"),
            size: vbuf_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout: bgl,
            bind_group,
            vbuf,
            vbuf_capacity,
            texture,
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        globals_layout: &wgpu::BindGroupLayout,
        bind_group_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sprite.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sprite.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprite layout"),
            bind_group_layouts: &[globals_layout, bind_group_layout],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sprite pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SpriteVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2, 2 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: multisample_state(sample_count),
            multiview: None,
            cache: None,
        })
    }

    fn rebuild_pipeline(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        globals_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) {
        self.pipeline = Self::create_pipeline(
            device,
            color_format,
            globals_layout,
            &self.bind_group_layout,
            sample_count,
        );
    }

    fn prepare(&mut self, gpu: &Gpu, scene: &Scene, camera: &Camera) -> u32 {
        let mut verts: Vec<SpriteVertex> = Vec::new();
        let fwd = camera.forward();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(fwd);

        let mut quad = |corners: [Vec3; 4], color: [f32; 4]| {
            let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            let idx = [0, 1, 2, 0, 2, 3];
            for &i in &idx {
                verts.push(SpriteVertex {
                    pos: corners[i].to_array(),
                    uv: uv[i],
                    color,
                });
            }
        };

        for s in &scene.sprites {
            let h = s.size * 0.5;
            quad(
                [
                    s.pos - right * h + up * h,
                    s.pos + right * h + up * h,
                    s.pos + right * h - up * h,
                    s.pos - right * h - up * h,
                ],
                s.color,
            );
        }
        for b in &scene.beams {
            let mid = (b.a + b.b) * 0.5;
            let axis = b.b - b.a;
            let side = axis.cross(camera.pos - mid).normalize_or_zero() * (b.width * 0.5);
            quad([b.a - side, b.b - side, b.b + side, b.a + side], b.color);
        }

        if verts.is_empty() {
            return 0;
        }
        let bytes: &[u8] = bytemuck::cast_slice(&verts);
        if bytes.len() as u64 > self.vbuf_capacity {
            self.vbuf_capacity = (bytes.len() as u64).next_power_of_two();
            self.vbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sprite vbuf"),
                size: self.vbuf_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        gpu.queue.write_buffer(&self.vbuf, 0, bytes);
        verts.len() as u32
    }

    fn draw<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, vert_count: u32) {
        if vert_count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.draw(0..vert_count, 0..1);
    }
}

// ---------------------------------------------------------------------------
// HUD pass
// ---------------------------------------------------------------------------

struct HudPass {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    globals: wgpu::Buffer,
    vbuf: wgpu::Buffer,
    vbuf_capacity: u64,
}

impl HudPass {
    fn new(gpu: &Gpu, color_format: wgpu::TextureFormat) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/hud.wgsl").into()),
        });

        // Font atlas texture (R8).
        let atlas_pixels = build_font_atlas();
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hud font atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_W,
                height: ATLAS_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &atlas,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ATLAS_W),
                rows_per_image: Some(ATLAS_H),
            },
            wgpu::Extent3d {
                width: ATLAS_W,
                height: ATLAS_H,
                depth_or_array_layers: 1,
            },
        );
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hud sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud globals"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hud bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hud layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<HudVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vbuf_capacity = 64 * 1024;
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud vbuf"),
            size: vbuf_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group,
            globals,
            vbuf,
            vbuf_capacity,
        }
    }

    fn upload(&mut self, gpu: &Gpu, hud: &Hud, size: (u32, u32)) {
        let bytes: &[u8] = bytemuck::cast_slice(&hud.verts);
        if bytes.len() as u64 > self.vbuf_capacity {
            self.vbuf_capacity = (bytes.len() as u64).next_power_of_two();
            self.vbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hud vbuf"),
                size: self.vbuf_capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        gpu.queue.write_buffer(&self.vbuf, 0, bytes);
        let globals = [size.0 as f32, size.1 as f32, 0.0, 0.0];
        gpu.queue
            .write_buffer(&self.globals, 0, bytemuck::cast_slice(&globals));
    }

    fn draw<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, hud: &Hud) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.draw(0..hud.verts.len() as u32, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::{RendererConfig, sample_count_transition, select_effective_sample_count};

    #[test]
    fn default_config_preserves_public_api_defaults() {
        assert_eq!(RendererConfig::default().requested_sample_count, 1);
    }

    #[test]
    fn selects_requested_count_when_supported() {
        assert_eq!(select_effective_sample_count(8, &[1, 2, 4, 8]), 8);
    }

    #[test]
    fn falls_down_to_the_highest_supported_count() {
        assert_eq!(select_effective_sample_count(8, &[1, 2, 4]), 4);
        assert_eq!(select_effective_sample_count(4, &[1, 2]), 2);
        assert_eq!(select_effective_sample_count(2, &[1, 4]), 1);
    }

    #[test]
    fn one_x_request_stays_one_x() {
        assert_eq!(select_effective_sample_count(1, &[1, 2, 4, 8]), 1);
    }

    #[test]
    fn changing_effective_count_requires_pipeline_rebuild() {
        let transition = sample_count_transition(4, 1, &[1, 2, 4]);
        assert_eq!(transition.requested_sample_count, 4);
        assert_eq!(transition.effective_sample_count, 4);
        assert!(transition.rebuild_pipelines);
    }

    #[test]
    fn fallback_request_updates_requested_without_rebuilding() {
        let transition = sample_count_transition(8, 4, &[1, 2, 4]);
        assert_eq!(transition.requested_sample_count, 8);
        assert_eq!(transition.effective_sample_count, 4);
        assert!(!transition.rebuild_pipelines);
    }

    #[test]
    fn returning_to_one_x_requires_pipeline_rebuild() {
        let transition = sample_count_transition(1, 4, &[1, 2, 4]);
        assert_eq!(transition.effective_sample_count, 1);
        assert!(transition.rebuild_pipelines);
    }

    #[test]
    fn repeated_effective_request_does_not_rebuild() {
        let transition = sample_count_transition(4, 4, &[1, 2, 4]);
        assert!(!transition.rebuild_pipelines);
    }

    #[test]
    fn model_shader_keeps_surviving_cutout_fragments_opaque() {
        let shader = include_str!("shaders/model.wgsl");
        assert!(shader.contains("discard;"));
        assert!(shader.contains("return vec4f(albedo.rgb * lighting, 1.0);"));
        assert!(!shader.contains("return vec4f(albedo.rgb * lighting, albedo.a);"));
    }
}
