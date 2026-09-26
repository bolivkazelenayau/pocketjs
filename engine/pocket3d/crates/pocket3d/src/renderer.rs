//! Forward renderer. One `Renderer` owns all pipelines; each frame it draws
//! a `Scene` (3D) then a `Hud` (2D overlay) into any color view.

use anyhow::{Result, anyhow, bail};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::camera::Camera;
use crate::gpu::{DEPTH_FORMAT, DepthTarget, Gpu};
use crate::hud::{ATLAS_H, ATLAS_W, Hud, HudVertex, build_font_atlas};
use crate::material::{MaterialPipelineKey, PipelineShadingModel, RenderPhase, RenderSortKey};
use crate::model::{MaterialInstanceRaw, ModelAsset, ModelInstance, ModelVertex, MtoonRenderMode};
use crate::scene::Scene;
use crate::texture::{GpuTexture, Samplers, create_rgba_texture};
use crate::world::{WorldBatchKind, WorldVertex};

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn aligned_size(size: u64, alignment: u64) -> u64 {
    size.div_ceil(alignment) * alignment
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalsRaw {
    view_proj: [[f32; 4]; 4],
    inverse_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    sky_sun_dir: [f32; 4],
    sky_sun_color: [f32; 4],
    model_sun_dir: [f32; 4],
    model_sun_color: [f32; 4],
    model_ambient: [f32; 4],
    /// x: band count, y: wrap, z: enabled.
    toon: [f32; 4],
    /// rgb: color, w: strength.
    rim_color: [f32; 4],
    /// x: exponent.
    rim_params: [f32; 4],
    /// rgb: color, w: enabled.
    fog_color: [f32; 4],
    /// x: start, y: end.
    fog_params: [f32; 4],
    /// xy: viewport dimensions, zw: reciprocal dimensions.
    viewport: [f32; 4],
}

pub struct Renderer {
    pub color_format: wgpu::TextureFormat,
    pub samplers: Samplers,
    pub world_material_layout: wgpu::BindGroupLayout,
    pub model_material_layout: wgpu::BindGroupLayout,
    pub mtoon_material_layout: wgpu::BindGroupLayout,
    depth: Option<DepthTarget>,
    msaa_color: Option<MsaaColorTarget>,
    smaa: Option<crate::smaa::SmaaPass>,
    requested_sample_count: u32,
    effective_sample_count: u32,
    supported_sample_counts: Vec<u32>,
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    world_shader: wgpu::ShaderModule,
    world_layout: wgpu::PipelineLayout,
    background_sky_layout: wgpu::PipelineLayout,
    world_opaque: wgpu::RenderPipeline,
    world_alphatest: wgpu::RenderPipeline,
    world_sky: wgpu::RenderPipeline,
    world_background_sky: wgpu::RenderPipeline,
    models: ModelPass,
    sprites: SpritePass,
    hud: HudPass,
}

/// Generic renderer initialization settings. Product-specific preferences
/// should be translated to this representation at the application boundary.
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TargetResizePlan {
    recreate_depth: bool,
    recreate_msaa: bool,
    recreate_smaa: bool,
}

impl TargetResizePlan {
    fn is_empty(self) -> bool {
        !self.recreate_depth && !self.recreate_msaa && !self.recreate_smaa
    }
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
    let device_features = gpu.device.features();
    let use_adapter_format_features = device_features
        .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        || !gpu
            .adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::WEBGPU_TEXTURE_FORMAT_SUPPORT);
    let format_features = |format: wgpu::TextureFormat| {
        if use_adapter_format_features {
            gpu.adapter.get_texture_format_features(format)
        } else {
            format.guaranteed_format_features(device_features)
        }
    };
    let color_features = format_features(color_format);
    let depth_features = format_features(DEPTH_FORMAT);
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

fn create_world_pipelines(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    world_shader: &wgpu::ShaderModule,
    world_layout: &wgpu::PipelineLayout,
    background_sky_layout: &wgpu::PipelineLayout,
    sample_count: u32,
) -> (
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
) {
    let make_world_pipeline = |label: &str, fs_entry: &str| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(world_layout),
            vertex: wgpu::VertexState {
                module: world_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[WorldVertex::LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: world_shader,
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

    let world_background_sky = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("world background sky"),
        layout: Some(background_sky_layout),
        vertex: wgpu::VertexState {
            module: world_shader,
            entry_point: Some("vs_background_sky"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: world_shader,
            entry_point: Some("fs_background_sky"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
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
    });

    (
        make_world_pipeline("world opaque", "fs_opaque"),
        make_world_pipeline("world alphatest", "fs_alphatest"),
        make_world_pipeline("world sky", "fs_sky"),
        world_background_sky,
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
    /// This is intentionally crate-private: applications should use
    /// [`Self::set_smaa_enabled`] to change the post-process state at runtime.
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
        let world_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/world.wgsl").into()),
        });
        let world_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world layout"),
            bind_group_layouts: &[&globals_bgl, &world_material_layout],
            push_constant_ranges: &[],
        });
        let background_sky_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("world background sky layout"),
                bind_group_layouts: &[&globals_bgl],
                push_constant_ranges: &[],
            });
        let (world_opaque, world_alphatest, world_sky, world_background_sky) =
            create_world_pipelines(
                device,
                color_format,
                &world_shader,
                &world_layout,
                &background_sky_layout,
                effective_sample_count,
            );

        let models = ModelPass::new(gpu, color_format, &globals_bgl, effective_sample_count);
        let model_material_layout = models.material_layout.clone();
        let mtoon_material_layout = models.mtoon_material_layout.clone();
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
            mtoon_material_layout,
            depth: None,
            msaa_color: None,
            smaa,
            requested_sample_count,
            effective_sample_count,
            supported_sample_counts,
            globals_buf,
            globals_bg,
            world_shader,
            world_layout,
            background_sky_layout,
            world_opaque,
            world_alphatest,
            world_sky,
            world_background_sky,
            models,
            sprites,
            hud: HudPass::new(gpu, color_format),
        })
    }

    /// Ensure the internal targets match the current effective sample count.
    fn ensure_targets(&mut self, gpu: &Gpu, size: (u32, u32)) {
        let plan = self.target_resize_plan(size);
        if plan.recreate_depth {
            self.depth = Some(DepthTarget::new_with_sample_count(
                gpu,
                size.0,
                size.1,
                self.effective_sample_count,
            ));
        }

        if plan.recreate_msaa {
            self.msaa_color = Some(MsaaColorTarget::new(
                gpu,
                self.color_format,
                size,
                self.effective_sample_count,
            ));
        }

        if plan.recreate_smaa
            && let Some(smaa) = self.smaa.as_mut()
        {
            smaa.ensure_targets(gpu, size);
        }
    }

    fn target_resize_plan(&self, size: (u32, u32)) -> TargetResizePlan {
        TargetResizePlan {
            recreate_depth: self.depth.as_ref().is_none_or(|depth| depth.size != size),
            recreate_msaa: self.effective_sample_count > 1
                && self.msaa_color.as_ref().is_none_or(|target| {
                    target.size != size || target.sample_count != self.effective_sample_count
                }),
            recreate_smaa: self
                .smaa
                .as_ref()
                .is_some_and(|smaa| smaa.targets_size() != Some(size)),
        }
    }

    /// Recreate only targets whose dimensions depend on the live viewport.
    ///
    /// The surface is owned by the application loop and must be configured
    /// before this is called. A false result means all existing targets already
    /// match the requested size. Sample-count negotiation and SMAA enablement
    /// are deliberately not touched here.
    pub fn resize(&mut self, gpu: &Gpu, size: (u32, u32)) -> bool {
        if size.0 == 0 || size.1 == 0 {
            return false;
        }

        let plan = self.target_resize_plan(size);
        if plan.is_empty() {
            return false;
        }
        self.ensure_targets(gpu, size);
        true
    }

    /// Release size-dependent targets while a window is minimized.
    ///
    /// Pipelines, model resources, the GPU device, and the enabled AA modes are
    /// retained so restoring the window only allocates targets for its new
    /// non-zero viewport.
    pub fn suspend(&mut self) {
        self.depth = None;
        self.msaa_color = None;
        if let Some(smaa) = self.smaa.as_mut() {
            smaa.suspend();
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
        let (world_opaque, world_alphatest, world_sky, world_background_sky) =
            create_world_pipelines(
                &gpu.device,
                self.color_format,
                &self.world_shader,
                &self.world_layout,
                &self.background_sky_layout,
                transition.effective_sample_count,
            );
        self.models.rebuild_pipelines(
            &gpu.device,
            self.color_format,
            transition.effective_sample_count,
        );
        self.sprites.rebuild_pipeline(
            &gpu.device,
            self.color_format,
            transition.effective_sample_count,
        );
        self.world_opaque = world_opaque;
        self.world_alphatest = world_alphatest;
        self.world_sky = world_sky;
        self.world_background_sky = world_background_sky;
        self.effective_sample_count = transition.effective_sample_count;

        // The next render lazily creates targets at the current size. This
        // handles 1x <-> Nx transitions and avoids allocating targets that
        // may never be used.
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

    /// Returns whether the SMAA post-process pass is currently enabled.
    pub fn smaa_enabled(&self) -> bool {
        self.smaa.is_some()
    }

    /// Enable or disable the SMAA post-process pass between frames.
    ///
    /// Enabling creates only SMAA's pipelines, lookup textures, and lazy
    /// size-dependent targets. Disabling drops only that pass. The GPU device,
    /// surface-dependent MSAA state, and all scene resources are preserved.
    /// Callers must invoke this outside any active render pass, before the next
    /// call to [`Self::render`].
    pub fn set_smaa_enabled(&mut self, gpu: &Gpu, enabled: bool) {
        if enabled == self.smaa_enabled() {
            return;
        }

        if enabled {
            self.smaa = Some(crate::smaa::SmaaPass::new(gpu, self.color_format));
            log::info!("post-process AA: SMAA 1x enabled");
        } else {
            self.smaa = None;
            log::info!("post-process AA: SMAA disabled");
        }
    }

    /// If a frame's combined joint data exceeds one device buffer, logs the
    /// preparation error and renders the rest of the frame without models.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        color_view: &wgpu::TextureView,
        size: (u32, u32),
        scene: &Scene,
        camera: &Camera,
        hud: &Hud,
    ) {
        let Some(aspect) = Camera::aspect_for_viewport(size) else {
            return;
        };
        self.ensure_targets(gpu, size);
        if let Some(smaa) = self.smaa.as_mut() {
            smaa.update_metrics(gpu, size, scene.transparent_clear);
        }

        let view_proj = camera.view_proj(aspect);
        let toon = match scene.lighting.toon {
            Some(toon) => [
                toon.steps as f32,
                finite_or(toon.wrap, 0.0).clamp(0.0, 1.0),
                1.0,
                0.0,
            ],
            None => [0.0; 4],
        };
        let (rim_color, rim_params) = match scene.lighting.rim {
            Some(rim) => (
                rim.color
                    .extend(finite_or(rim.strength, 0.0).max(0.0))
                    .to_array(),
                [finite_or(rim.power, 1.0).max(0.0001), 0.0, 0.0, 0.0],
            ),
            None => ([0.0; 4], [1.0, 0.0, 0.0, 0.0]),
        };
        let (fog_color, fog_params) = match scene.lighting.fog {
            Some(fog)
                if fog.color.is_finite()
                    && fog.start.is_finite()
                    && fog.end.is_finite()
                    && fog.end > fog.start =>
            {
                (
                    fog.color.extend(1.0).to_array(),
                    [fog.start, fog.end, 0.0, 0.0],
                )
            }
            _ => ([0.0; 4], [0.0, 1.0, 0.0, 0.0]),
        };
        let globals = GlobalsRaw {
            view_proj: view_proj.to_cols_array_2d(),
            inverse_view_proj: view_proj.inverse().to_cols_array_2d(),
            cam_pos: camera.pos.extend(scene.time).to_array(),
            sky_zenith: scene.sky.zenith.extend(1.0).to_array(),
            sky_horizon: scene.sky.horizon.extend(1.0).to_array(),
            sky_sun_dir: scene.sky.sun_dir.extend(0.0).to_array(),
            sky_sun_color: scene.sky.sun_color.extend(1.0).to_array(),
            model_sun_dir: scene.lighting.sun_dir.extend(0.0).to_array(),
            model_sun_color: scene.lighting.sun_color.extend(1.0).to_array(),
            model_ambient: scene.lighting.ambient.extend(1.0).to_array(),
            toon,
            rim_color,
            rim_params,
            fog_color,
            fog_params,
            viewport: [
                size.0 as f32,
                size.1 as f32,
                1.0 / size.0 as f32,
                1.0 / size.1 as f32,
            ],
        };
        gpu.queue
            .write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        // Upload per-instance data (joint palettes etc.) before recording.
        let (model_draws, viewmodel_draw) = match self.models.prepare(gpu, scene, camera) {
            Ok(draws) => draws,
            Err(error) => {
                log::error!("model frame preparation failed: {error:#}");
                (Vec::new(), None)
            }
        };
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

            if scene.draw_sky {
                pass.set_pipeline(&self.world_background_sky);
                pass.draw(0..3, 0..1);
            }

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

            self.models.draw_scene(&mut pass, &model_draws);
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
            self.models
                .draw_viewmodel(&mut pass, std::slice::from_ref(vm));
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
const JOINT_MATRIX_BYTES: u64 = std::mem::size_of::<Mat4>() as u64;

fn joint_palette_bytes(count: usize, binding_limit: u64) -> Result<u64> {
    let bytes = u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(JOINT_MATRIX_BYTES))
        .ok_or_else(|| anyhow!("joint palette byte size overflows"))?;
    if bytes > binding_limit {
        bail!(
            "joint palette requires {count} joints ({bytes} bytes), but the device storage binding supports {binding_limit} bytes ({} joints)",
            binding_limit / JOINT_MATRIX_BYTES
        );
    }
    Ok(bytes)
}

fn joint_palette_offset(
    end: u64,
    alignment: u64,
    palette_bytes: u64,
    buffer_limit: u64,
) -> Result<(u32, u64)> {
    let padding = (alignment - end % alignment) % alignment;
    let offset = end
        .checked_add(padding)
        .ok_or_else(|| anyhow!("joint buffer offset overflows"))?;
    let offset_u32 = u32::try_from(offset)
        .map_err(|_| anyhow!("joint dynamic offset {offset} exceeds u32 range"))?;
    let next_end = offset
        .checked_add(palette_bytes)
        .ok_or_else(|| anyhow!("joint buffer size overflows"))?;
    if next_end > buffer_limit {
        bail!(
            "frame joint palettes require {next_end} bytes, but the device max_buffer_size is {buffer_limit} bytes"
        );
    }
    Ok((offset_u32, next_end))
}

fn joint_buffer_capacity(required: u64, current: u64, limit: u64) -> Result<u64> {
    if required > limit {
        bail!(
            "frame joint palettes require {required} bytes, but the device max_buffer_size is {limit} bytes"
        );
    }
    if required <= current {
        return Ok(current);
    }
    Ok(required
        .checked_next_power_of_two()
        .unwrap_or(limit)
        .min(limit))
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal_model: [[f32; 4]; 4],
    tint: [f32; 4],
    params: [f32; 4],
}

/// CPU-side inverse-transpose for instance transforms. A singular transform
/// cannot define a true normal matrix, so retain the old linear transform as
/// a finite fallback for zero-scale hiding and transition animations.
fn model_normal_matrix(model: Mat4) -> Mat4 {
    let determinant = model.determinant();
    if determinant.is_finite() && determinant.abs() > 1e-8 {
        model.inverse().transpose()
    } else {
        Mat4::from_cols(model.x_axis, model.y_axis, model.z_axis, glam::Vec4::W)
    }
}

fn presentation_alpha(alpha: f32) -> f32 {
    finite_or(alpha, 1.0).clamp(0.0, 1.0)
}

fn camera_relative_depth(camera: &Camera, model: Mat4, rest_bounds_center: Vec3) -> f32 {
    let center = model.transform_point3(rest_bounds_center);
    (center - camera.pos).dot(camera.forward())
}

pub(crate) struct ModelDraw {
    asset: std::sync::Arc<ModelAsset>,
    mtoon_draw_route: MtoonRenderMode,
    inst_offset: u32,
    joints_offset: u32,
    model: Mat4,
    presentation_alpha: f32,
    material_offsets: Vec<u32>,
    /// The instance's morph overlay buffer (wgpu buffers are ref-counted).
    morph: Option<wgpu::Buffer>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ModelSubmission {
    draw_index: usize,
    primitive_index: usize,
    sort_key: RenderSortKey,
}

struct ModelPass {
    sample_count: u32,
    opaque: wgpu::RenderPipeline,
    opaque_double_sided: wgpu::RenderPipeline,
    blend: wgpu::RenderPipeline,
    blend_double_sided: wgpu::RenderPipeline,
    blend_zwrite: wgpu::RenderPipeline,
    blend_zwrite_double_sided: wgpu::RenderPipeline,
    mtoon_opaque: wgpu::RenderPipeline,
    mtoon_opaque_double_sided: wgpu::RenderPipeline,
    mtoon_mask: wgpu::RenderPipeline,
    mtoon_mask_double_sided: wgpu::RenderPipeline,
    mtoon_blend_zwrite: wgpu::RenderPipeline,
    mtoon_blend_zwrite_double_sided: wgpu::RenderPipeline,
    mtoon_blend: wgpu::RenderPipeline,
    mtoon_blend_double_sided: wgpu::RenderPipeline,
    mtoon_outline_opaque: wgpu::RenderPipeline,
    mtoon_outline_mask: wgpu::RenderPipeline,
    mtoon_outline_blend_zwrite: wgpu::RenderPipeline,
    mtoon_outline_blend: wgpu::RenderPipeline,
    material_layout: wgpu::BindGroupLayout,
    mtoon_material_layout: wgpu::BindGroupLayout,
    object_layout: wgpu::BindGroupLayout,
    shader: wgpu::ShaderModule,
    mtoon_shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    mtoon_pipeline_layout: wgpu::PipelineLayout,
    object_bg: wgpu::BindGroup,
    instance_buf: wgpu::Buffer,
    joints_buf: wgpu::Buffer,
    material_instance_buf: wgpu::Buffer,
    instance_capacity: u64,
    joints_capacity: u64,
    joint_binding_size: u64,
    material_instance_capacity: u64,
    material_instance_stride: u64,
    scene_submissions: Vec<ModelSubmission>,
    viewmodel_submissions: Vec<ModelSubmission>,
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
        let mtoon_material_layout = ModelAsset::mtoon_material_layout(gpu);
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            MaterialInstanceRaw,
                        >() as u64),
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("model.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/model.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("model layout"),
            bind_group_layouts: &[globals_bgl, &material_layout, &object_layout],
            push_constant_ranges: &[],
        });
        let mtoon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mtoon.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mtoon.wgsl").into()),
        });
        let mtoon_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MToon surface layout"),
                bind_group_layouts: &[globals_bgl, &mtoon_material_layout, &object_layout],
                push_constant_ranges: &[],
            });
        let (
            mtoon_opaque,
            mtoon_opaque_double_sided,
            mtoon_mask,
            mtoon_mask_double_sided,
            mtoon_blend_zwrite,
            mtoon_blend_zwrite_double_sided,
            mtoon_blend,
            mtoon_blend_double_sided,
        ) = Self::create_mtoon_pipelines(
            device,
            color_format,
            &mtoon_shader,
            &mtoon_pipeline_layout,
            sample_count,
        );
        let (
            mtoon_outline_opaque,
            mtoon_outline_mask,
            mtoon_outline_blend_zwrite,
            mtoon_outline_blend,
        ) = Self::create_mtoon_outline_pipelines(
            device,
            color_format,
            &mtoon_shader,
            &mtoon_pipeline_layout,
            sample_count,
        );
        let (
            opaque,
            opaque_double_sided,
            blend,
            blend_double_sided,
            blend_zwrite,
            blend_zwrite_double_sided,
        ) = Self::create_pipelines(device, color_format, &shader, &layout, sample_count);

        let instance_capacity = 64 * INSTANCE_STRIDE;
        let joints_capacity = (256 * 1024).min(device.limits().max_buffer_size);
        let joint_binding_size = JOINT_MATRIX_BYTES;
        let material_instance_stride = aligned_size(
            std::mem::size_of::<MaterialInstanceRaw>() as u64,
            device.limits().min_uniform_buffer_offset_alignment as u64,
        );
        let material_instance_capacity = 64 * material_instance_stride;
        let instance_buf = Self::make_instance_buf(device, instance_capacity);
        let joints_buf = Self::make_joints_buf(device, joints_capacity);
        let material_instance_buf =
            Self::make_material_instance_buf(device, material_instance_capacity);
        let object_bg = Self::make_object_bg(
            device,
            &object_layout,
            &instance_buf,
            &joints_buf,
            &material_instance_buf,
            joint_binding_size,
        );

        Self {
            sample_count,
            opaque,
            opaque_double_sided,
            blend,
            blend_double_sided,
            blend_zwrite,
            blend_zwrite_double_sided,
            mtoon_opaque,
            mtoon_opaque_double_sided,
            mtoon_mask,
            mtoon_mask_double_sided,
            mtoon_blend_zwrite,
            mtoon_blend_zwrite_double_sided,
            mtoon_blend,
            mtoon_blend_double_sided,
            mtoon_outline_opaque,
            mtoon_outline_mask,
            mtoon_outline_blend_zwrite,
            mtoon_outline_blend,
            material_layout,
            mtoon_material_layout,
            object_layout,
            shader,
            mtoon_shader,
            pipeline_layout: layout,
            mtoon_pipeline_layout,
            object_bg,
            instance_buf,
            joints_buf,
            material_instance_buf,
            instance_capacity,
            joints_capacity,
            joint_binding_size,
            material_instance_capacity,
            material_instance_stride,
            scene_submissions: Vec::new(),
            viewmodel_submissions: Vec::new(),
        }
    }

    fn create_pipelines(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        sample_count: u32,
    ) -> (
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
    ) {
        let make_pipeline = |label: &str,
                             blend: Option<wgpu::BlendState>,
                             depth_write_enabled: bool,
                             cull_mode: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[ModelVertex::LAYOUT],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled,
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
            make_pipeline("model opaque", None, true, Some(wgpu::Face::Back)),
            make_pipeline("model opaque double-sided", None, true, None),
            make_pipeline(
                "model blend",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                false,
                Some(wgpu::Face::Back),
            ),
            make_pipeline(
                "model blend double-sided",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                false,
                None,
            ),
            make_pipeline(
                "model blend depth-write",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                true,
                Some(wgpu::Face::Back),
            ),
            make_pipeline(
                "model blend depth-write double-sided",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                true,
                None,
            ),
        )
    }

    fn create_mtoon_pipelines(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        sample_count: u32,
    ) -> (
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
    ) {
        let make = |label: &str,
                    blend: Option<wgpu::BlendState>,
                    depth_write_enabled: bool,
                    double_sided: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[ModelVertex::LAYOUT],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: if double_sided {
                        None
                    } else {
                        Some(wgpu::Face::Back)
                    },
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled,
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
            make("MToon opaque", None, true, false),
            make("MToon opaque double-sided", None, true, true),
            make("MToon mask", None, true, false),
            make("MToon mask double-sided", None, true, true),
            make(
                "MToon blend depth-write",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                true,
                false,
            ),
            make(
                "MToon blend depth-write double-sided",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                true,
                true,
            ),
            make(
                "MToon blend",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                false,
                false,
            ),
            make(
                "MToon blend double-sided",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                false,
                true,
            ),
        )
    }

    fn create_mtoon_outline_pipelines(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        sample_count: u32,
    ) -> (
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
        wgpu::RenderPipeline,
    ) {
        let make = |label: &str, blend: Option<wgpu::BlendState>, depth_write_enabled: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_outline"),
                    compilation_options: Default::default(),
                    buffers: &[ModelVertex::LAYOUT],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_outline"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Front),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled,
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
            make("MToon outline opaque", None, true),
            make("MToon outline mask", None, true),
            make(
                "MToon outline blend depth-write",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                true,
            ),
            make(
                "MToon outline blend",
                Some(wgpu::BlendState::ALPHA_BLENDING),
                false,
            ),
        )
    }

    fn rebuild_pipelines(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) {
        let (
            opaque,
            opaque_double_sided,
            blend,
            blend_double_sided,
            blend_zwrite,
            blend_zwrite_double_sided,
        ) = Self::create_pipelines(
            device,
            color_format,
            &self.shader,
            &self.pipeline_layout,
            sample_count,
        );
        self.opaque = opaque;
        self.opaque_double_sided = opaque_double_sided;
        self.blend = blend;
        self.blend_double_sided = blend_double_sided;
        self.blend_zwrite = blend_zwrite;
        self.blend_zwrite_double_sided = blend_zwrite_double_sided;
        self.sample_count = sample_count;
        (
            self.mtoon_opaque,
            self.mtoon_opaque_double_sided,
            self.mtoon_mask,
            self.mtoon_mask_double_sided,
            self.mtoon_blend_zwrite,
            self.mtoon_blend_zwrite_double_sided,
            self.mtoon_blend,
            self.mtoon_blend_double_sided,
        ) = Self::create_mtoon_pipelines(
            device,
            color_format,
            &self.mtoon_shader,
            &self.mtoon_pipeline_layout,
            sample_count,
        );
        (
            self.mtoon_outline_opaque,
            self.mtoon_outline_mask,
            self.mtoon_outline_blend_zwrite,
            self.mtoon_outline_blend,
        ) = Self::create_mtoon_outline_pipelines(
            device,
            color_format,
            &self.mtoon_shader,
            &self.mtoon_pipeline_layout,
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

    fn make_material_instance_buf(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("model instance materials"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_object_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        instances: &wgpu::Buffer,
        joints: &wgpu::Buffer,
        materials: &wgpu::Buffer,
        joint_binding_size: u64,
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
                        size: wgpu::BufferSize::new(joint_binding_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: materials,
                        offset: 0,
                        size: wgpu::BufferSize::new(
                            std::mem::size_of::<MaterialInstanceRaw>() as u64
                        ),
                    }),
                },
            ],
        })
    }

    /// Compute palettes + instance data for everything in the scene and
    /// upload once. Returns draw entries (scene models, viewmodel).
    fn prepare(
        &mut self,
        gpu: &Gpu,
        scene: &Scene,
        camera: &Camera,
    ) -> Result<(Vec<ModelDraw>, Option<ModelDraw>)> {
        self.scene_submissions.clear();
        self.viewmodel_submissions.clear();
        let all: Vec<&ModelInstance> = scene.models.iter().chain(scene.viewmodel.iter()).collect();
        if all.is_empty() {
            return Ok((Vec::new(), None));
        }

        let limits = gpu.device.limits();
        let binding_limit = limits.max_storage_buffer_binding_size as u64;
        let buffer_limit = limits.max_buffer_size;
        let joint_alignment = limits.min_storage_buffer_offset_alignment as u64;
        let mut inst_bytes = vec![0u8; all.len() * INSTANCE_STRIDE as usize];
        let mut joint_bytes: Vec<u8> = Vec::new();
        let mut joint_binding_size = JOINT_MATRIX_BYTES;
        let mut material_bytes = Vec::new();
        let mut draws = Vec::with_capacity(all.len());
        let mut palette: Vec<Mat4> = Vec::new();

        for (i, inst) in all.iter().enumerate() {
            let presentation_alpha = presentation_alpha(inst.tint[3]);
            let mut tint = inst.tint;
            tint[3] = presentation_alpha;
            let raw = InstanceRaw {
                model: inst.transform.to_cols_array_2d(),
                normal_model: model_normal_matrix(inst.transform).to_cols_array_2d(),
                tint,
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
            let palette_bytes = joint_palette_bytes(palette.len(), binding_limit)?;
            joint_binding_size = joint_binding_size.max(palette_bytes);
            let (joints_offset, end) = joint_palette_offset(
                joint_bytes.len() as u64,
                joint_alignment,
                palette_bytes,
                buffer_limit,
            )?;
            let end = usize::try_from(end)
                .map_err(|_| anyhow!("joint buffer size exceeds addressable memory"))?;
            joint_bytes.resize(joints_offset as usize, 0);
            for m in &palette {
                joint_bytes.extend_from_slice(bytemuck::cast_slice(&m.to_cols_array()));
            }
            debug_assert_eq!(joint_bytes.len(), end);

            let material_offsets = (0..inst.asset.primitives.len())
                .map(|primitive_index| {
                    let offset = material_bytes.len();
                    material_bytes.resize(offset + self.material_instance_stride as usize, 0);
                    let raw = inst
                        .asset
                        .instance_material_raw(primitive_index, &inst.materials);
                    let bytes = bytemuck::bytes_of(&raw);
                    material_bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
                    offset as u32
                })
                .collect();

            draws.push(ModelDraw {
                asset: inst.asset.clone(),
                mtoon_draw_route: inst.mtoon_draw_route,
                inst_offset: off as u32,
                joints_offset,
                model: inst.transform,
                presentation_alpha,
                material_offsets,
                morph: inst.morph.as_ref().map(|m| m.buffer().clone()),
            });
        }

        // A dynamic offset moves the start, while the bind group's range is
        // shared by every draw. Only the last palette may need tail padding.
        if let Some(last) = draws.last() {
            let need = (last.joints_offset as u64)
                .checked_add(joint_binding_size)
                .ok_or_else(|| anyhow!("joint binding end overflows"))?;
            if need > buffer_limit {
                bail!(
                    "frame joint palettes require {need} bytes including the shared binding range, but the device max_buffer_size is {buffer_limit} bytes"
                );
            }
            let need = usize::try_from(need)
                .map_err(|_| anyhow!("joint buffer size exceeds addressable memory"))?;
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
            self.joints_capacity = joint_buffer_capacity(
                joint_bytes.len() as u64,
                self.joints_capacity,
                buffer_limit,
            )?;
            self.joints_buf = Self::make_joints_buf(device, self.joints_capacity);
            recreate = true;
        }
        if joint_binding_size != self.joint_binding_size {
            self.joint_binding_size = joint_binding_size;
            recreate = true;
        }
        if material_bytes.len() as u64 > self.material_instance_capacity {
            self.material_instance_capacity = (material_bytes.len() as u64).next_power_of_two();
            self.material_instance_buf =
                Self::make_material_instance_buf(device, self.material_instance_capacity);
            recreate = true;
        }
        if recreate {
            self.object_bg = Self::make_object_bg(
                device,
                &self.object_layout,
                &self.instance_buf,
                &self.joints_buf,
                &self.material_instance_buf,
                self.joint_binding_size,
            );
        }
        gpu.queue.write_buffer(&self.instance_buf, 0, &inst_bytes);
        if !joint_bytes.is_empty() {
            gpu.queue.write_buffer(&self.joints_buf, 0, &joint_bytes);
        }
        if !material_bytes.is_empty() {
            gpu.queue
                .write_buffer(&self.material_instance_buf, 0, &material_bytes);
        }

        let viewmodel = scene.viewmodel.is_some().then(|| draws.pop()).flatten();
        Self::collect_submissions(&draws, camera, &mut self.scene_submissions);
        Self::collect_submissions(
            viewmodel.as_slice(),
            camera,
            &mut self.viewmodel_submissions,
        );
        Ok((draws, viewmodel))
    }

    fn collect_submissions(
        draws: &[ModelDraw],
        camera: &Camera,
        output: &mut Vec<ModelSubmission>,
    ) {
        output.clear();
        let primitive_count = draws.iter().map(|draw| draw.asset.primitives.len()).sum();
        output.reserve(primitive_count);
        let mut author_draw_order = 0_u64;
        for (draw_index, draw) in draws.iter().enumerate() {
            if draw.presentation_alpha == 0.0 {
                author_draw_order += draw.asset.primitives.len() as u64;
                continue;
            }
            for (primitive_index, primitive) in draw.asset.primitives.iter().enumerate() {
                let route = primitive.draw_route(draw.mtoon_draw_route);
                let phase = route.phase.with_presentation_alpha(draw.presentation_alpha);
                let camera_depth = if phase.pass_class() == crate::material::RenderPassClass::Blend
                {
                    camera_relative_depth(camera, draw.model, primitive.rest_bounds_center)
                } else {
                    0.0
                };
                output.push(ModelSubmission {
                    draw_index,
                    primitive_index,
                    sort_key: RenderSortKey {
                        phase,
                        queue_offset: route.queue_offset,
                        camera_depth,
                        author_draw_order,
                    },
                });
                author_draw_order += 1;
            }
        }
        output.sort_by(|left, right| left.sort_key.compare(&right.sort_key));
    }

    fn draw_scene<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, draws: &'p [ModelDraw]) {
        self.draw_submissions(pass, draws, &self.scene_submissions);
    }

    fn draw_viewmodel<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, draws: &'p [ModelDraw]) {
        self.draw_submissions(pass, draws, &self.viewmodel_submissions);
    }

    /// Draw one globally ordered submission list. The scene list spans every
    /// `ModelInstance`, so transparent primitives from separate assets sort
    /// against each other rather than only within their source model.
    fn draw_submissions<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        draws: &'p [ModelDraw],
        submissions: &[ModelSubmission],
    ) {
        for submission in submissions {
            let d = &draws[submission.draw_index];
            let pi = submission.primitive_index;
            let prim = &d.asset.primitives[pi];
            pass.set_bind_group(
                2,
                &self.object_bg,
                &[d.inst_offset, d.joints_offset, d.material_offsets[pi]],
            );
            pass.set_index_buffer(d.asset.ibuf.slice(..), wgpu::IndexFormat::Uint32);
            let phase = submission.sort_key.phase;
            let route = prim.draw_route(d.mtoon_draw_route);
            let pipeline_key = if route.native {
                MaterialPipelineKey::native_stage_d(phase, prim.double_sided, self.sample_count)
            } else {
                MaterialPipelineKey::current_fallback(
                    prim.unlit,
                    phase,
                    prim.double_sided,
                    self.sample_count,
                )
            };
            let pipeline = if pipeline_key.shading_model == PipelineShadingModel::Mtoon {
                match (phase, prim.double_sided) {
                    (RenderPhase::Opaque, false) => &self.mtoon_opaque,
                    (RenderPhase::Opaque, true) => &self.mtoon_opaque_double_sided,
                    (RenderPhase::Mask, false) => &self.mtoon_mask,
                    (RenderPhase::Mask, true) => &self.mtoon_mask_double_sided,
                    (RenderPhase::MtoonBlendZWrite, false) => &self.mtoon_blend_zwrite,
                    (RenderPhase::MtoonBlendZWrite, true) => &self.mtoon_blend_zwrite_double_sided,
                    (RenderPhase::Blend, false) => &self.mtoon_blend,
                    (RenderPhase::Blend, true) => &self.mtoon_blend_double_sided,
                }
            } else {
                match (phase, prim.double_sided) {
                    (RenderPhase::Opaque | RenderPhase::Mask, false) => &self.opaque,
                    (RenderPhase::Opaque | RenderPhase::Mask, true) => &self.opaque_double_sided,
                    (RenderPhase::MtoonBlendZWrite, false) => &self.blend_zwrite,
                    (RenderPhase::MtoonBlendZWrite, true) => &self.blend_zwrite_double_sided,
                    (RenderPhase::Blend, false) => &self.blend,
                    (RenderPhase::Blend, true) => &self.blend_double_sided,
                }
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(
                1,
                if route.native {
                    prim.mtoon_bind_group
                        .as_deref()
                        .expect("native route has bind group")
                } else {
                    &prim.bind_group
                },
                &[],
            );
            // Morphing primitives read vertices from the instance's overlay
            // buffer; base_vertex redirects the shared indices.
            let morph = d
                .morph
                .as_ref()
                .zip(d.asset.prim_morph.get(pi).copied().flatten());
            let base_vertex = match morph {
                Some((buf, (mi, pj))) => {
                    let mp = &d.asset.morph_meshes[mi].prims[pj];
                    pass.set_vertex_buffer(0, buf.slice(..));
                    mp.overlay_offset as i32 - mp.vertex_base as i32
                }
                None => {
                    pass.set_vertex_buffer(0, d.asset.vbuf.slice(..));
                    0
                }
            };
            let index_range = prim.first_index..prim.first_index + prim.index_count;
            pass.draw_indexed(index_range.clone(), base_vertex, 0..1);

            // Surface and outline are one sorted submission. The second draw
            // reuses the exact index/vertex or morph-overlay binding and the
            // same instance/joint dynamic offsets, so no other transparent
            // primitive can interleave between the pair.
            if route.outline {
                debug_assert!(prim.mtoon_bind_group.is_some());
                let outline_pipeline = match phase {
                    RenderPhase::Opaque => &self.mtoon_outline_opaque,
                    RenderPhase::Mask => &self.mtoon_outline_mask,
                    RenderPhase::MtoonBlendZWrite => &self.mtoon_outline_blend_zwrite,
                    RenderPhase::Blend => &self.mtoon_outline_blend,
                };
                pass.set_pipeline(outline_pipeline);
                pass.draw_indexed(index_range, base_vertex, 0..1);
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
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sprite.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/sprite.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprite layout"),
            bind_group_layouts: &[globals_bgl, &bgl],
            push_constant_ranges: &[],
        });
        let pipeline = Self::create_pipeline(device, color_format, &shader, &layout, sample_count);

        let vbuf_capacity = 64 * 1024;
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite vbuf"),
            size: vbuf_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            shader,
            pipeline_layout: layout,
            bind_group,
            vbuf,
            vbuf_capacity,
            texture,
        }
    }

    fn create_pipeline(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        sample_count: u32,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sprite pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SpriteVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x3,
                        1 => Float32x2,
                        2 => Float32x4
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
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
        sample_count: u32,
    ) {
        self.pipeline = Self::create_pipeline(
            device,
            color_format,
            &self.shader,
            &self.pipeline_layout,
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
    use crate::camera::Camera;
    use crate::gpu::{Gpu, OFFSCREEN_FORMAT, OffscreenTarget};
    use crate::hud::Hud;
    use crate::material::{RenderPhase, RenderSortKey};
    use crate::scene::Scene;
    use glam::{Mat4, Quat, Vec2, Vec3};
    #[cfg(not(target_arch = "wasm32"))]
    use wgpu::naga::{
        front::wgsl::parse_str,
        valid::{Capabilities, ValidationFlags, Validator},
    };

    use super::{
        GlobalsRaw, InstanceRaw, RendererConfig, TargetResizePlan, camera_relative_depth,
        joint_buffer_capacity, joint_palette_bytes, joint_palette_offset, model_normal_matrix,
        presentation_alpha, sample_count_transition, select_effective_sample_count,
    };

    #[test]
    fn joint_binding_and_packing_follow_injected_device_limits() {
        assert_eq!(joint_palette_bytes(512, 512 * 64).unwrap(), 32768);
        assert_eq!(joint_palette_bytes(513, 513 * 64).unwrap(), 32832);
        assert!(joint_palette_bytes(514, 513 * 64).is_err());
        assert!(joint_palette_bytes(usize::MAX, u64::MAX).is_err());
        assert_eq!(
            joint_palette_offset(32832, 256, 64, 65536).unwrap(),
            (33024, 33088)
        );
        assert!(joint_palette_offset(u64::MAX, 256, 64, u64::MAX).is_err());
        assert!(joint_palette_offset(65536, 256, 64, 65536).is_err());
        assert_eq!(joint_buffer_capacity(900, 512, 1000).unwrap(), 1000);
        assert_eq!(joint_buffer_capacity(1000, 512, 1000).unwrap(), 1000);
        assert!(joint_buffer_capacity(1001, 512, 1000).is_err());
    }

    #[test]
    fn aggregate_joint_frame_overflow_is_reported_as_frame_capacity() {
        let error = joint_palette_offset(900, 256, 128, 1024).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("frame joint palettes require 1152 bytes")
        );
        assert!(error.to_string().contains("max_buffer_size is 1024 bytes"));
    }

    #[test]
    fn presentation_alpha_is_bounded_and_non_finite_is_opaque() {
        assert_eq!(presentation_alpha(-1.0), 0.0);
        assert_eq!(presentation_alpha(0.25), 0.25);
        assert_eq!(presentation_alpha(2.0), 1.0);
        assert_eq!(presentation_alpha(f32::NAN), 1.0);
        assert_eq!(presentation_alpha(f32::INFINITY), 1.0);
    }

    #[test]
    fn camera_relative_depth_tracks_camera_movement_and_direction() {
        let model = glam::Mat4::IDENTITY;
        let forward = Camera::default();
        let near_center = Vec3::new(0.0, 0.0, -1.0);
        let far_center = Vec3::new(0.0, 0.0, -2.0);
        assert_eq!(camera_relative_depth(&forward, model, far_center), 2.0);

        let reversed = Camera {
            pos: Vec3::new(0.0, 0.0, -3.0),
            yaw: std::f32::consts::PI,
            ..Camera::default()
        };
        assert_eq!(camera_relative_depth(&reversed, model, near_center), 2.0);

        let key = |camera: &Camera, center, order| RenderSortKey {
            phase: RenderPhase::Blend,
            queue_offset: 0,
            camera_depth: camera_relative_depth(camera, model, center),
            author_draw_order: order,
        };
        assert_eq!(
            key(&forward, far_center, 1).compare(&key(&forward, near_center, 0)),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            key(&reversed, near_center, 0).compare(&key(&reversed, far_center, 1)),
            std::cmp::Ordering::Less,
            "moving through the layers and reversing the camera must reverse their draw order"
        );
    }

    #[test]
    fn default_config_preserves_public_api_defaults() {
        assert_eq!(RendererConfig::default().requested_sample_count, 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_multisample_paths_create_and_resolve_when_supported() {
        let Ok(gpu) = Gpu::new_headless() else {
            return;
        };
        let supported = super::supported_sample_counts(&gpu, OFFSCREEN_FORMAT);
        eprintln!(
            "GPU MSAA smoke: adapter {:?}, device adapter-specific format features {}, supported {:?}",
            gpu.adapter.get_info().name,
            gpu.device
                .features()
                .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES),
            supported
        );

        let mut renderer = super::Renderer::new(&gpu, OFFSCREEN_FORMAT).unwrap();
        let target = OffscreenTarget::new(&gpu, 4, 4);
        let mut scene = Scene::default();
        scene.draw_sky = true;

        for requested in [1, 2, 4, 8, 4, 1] {
            if !supported.contains(&requested) {
                continue;
            }

            assert_eq!(
                renderer.set_requested_sample_count(&gpu, requested),
                requested
            );
            assert_eq!(renderer.requested_sample_count(), requested);
            renderer.render(
                &gpu,
                &target.view,
                target.size,
                &scene,
                &Camera::default(),
                &Hud::default(),
            );
            gpu.device.poll(wgpu::PollType::Wait).unwrap();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_smaa_survives_msaa_transitions_and_keeps_hud_after_post_process() {
        let Ok(gpu) = Gpu::new_headless() else {
            return;
        };
        let supported = super::supported_sample_counts(&gpu, OFFSCREEN_FORMAT);
        if !supported.contains(&4) {
            eprintln!("GPU SMAA smoke skipped: 4x MSAA is unsupported ({supported:?})");
            return;
        }

        let device_identity = std::ptr::addr_of!(gpu.device);
        let adapter_identity = std::ptr::addr_of!(gpu.adapter);
        let mut renderer = super::Renderer::new(&gpu, OFFSCREEN_FORMAT).unwrap();
        assert!(!renderer.smaa_enabled());
        renderer.set_smaa_enabled(&gpu, true);
        let smaa_identity = renderer.smaa.as_ref().map(|smaa| smaa as *const _);
        renderer.set_smaa_enabled(&gpu, true);
        assert_eq!(
            renderer.smaa.as_ref().map(|smaa| smaa as *const _),
            smaa_identity
        );
        let target = OffscreenTarget::new(&gpu, 32, 32);
        let mut scene = Scene::default();
        scene.draw_sky = true;
        let mut hud = Hud::default();
        hud.rect(0.0, 0.0, 4.0, 4.0, [1.0, 0.0, 1.0, 1.0]);

        let mut saw_one_x = false;
        let mut saw_msaa = false;
        for requested in [1, 2, 4, 8, 4, 1] {
            assert!(
                supported.contains(&requested),
                "expected MX250 support for {requested}x"
            );
            assert_eq!(
                renderer.set_requested_sample_count(&gpu, requested),
                requested
            );
            assert_eq!(renderer.effective_sample_count(), requested);
            assert_eq!(std::ptr::addr_of!(gpu.device), device_identity);
            assert_eq!(std::ptr::addr_of!(gpu.adapter), adapter_identity);

            for enabled in [false, true] {
                renderer.set_smaa_enabled(&gpu, enabled);
                assert_eq!(renderer.smaa_enabled(), enabled);
                assert_eq!(renderer.effective_sample_count(), requested);
                assert_eq!(renderer.requested_sample_count(), requested);
                assert_eq!(std::ptr::addr_of!(gpu.device), device_identity);
                assert_eq!(std::ptr::addr_of!(gpu.adapter), adapter_identity);

                renderer.render(
                    &gpu,
                    &target.view,
                    target.size,
                    &scene,
                    &Camera::default(),
                    &hud,
                );
                gpu.device.poll(wgpu::PollType::Wait).unwrap();
                let rgba = target.read_rgba(&gpu).unwrap();
                let hud_pixel = &rgba[..4];
                assert_eq!(
                    hud_pixel,
                    &[255, 0, 255, 255],
                    "HUD must be drawn after SMAA={enabled} at {requested}x"
                );

                let hash = rgba.iter().fold(0xcbf29ce484222325u64, |hash, &byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
                });
                eprintln!(
                    "GPU AA smoke: adapter {:?}, {}x / SMAA {}, output FNV-1a {:016x}, HUD {:?}",
                    gpu.adapter.get_info().name,
                    requested,
                    enabled,
                    hash,
                    hud_pixel
                );
                saw_one_x |= requested == 1;
                saw_msaa |= requested > 1;
            }
        }
        assert!(saw_one_x);
        assert!(saw_msaa);
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
    fn invalid_request_sanitizes_to_one_x() {
        assert_eq!(select_effective_sample_count(0, &[1, 2, 4, 8]), 1);
        assert_eq!(select_effective_sample_count(16, &[1, 2, 4, 8]), 1);
    }

    #[test]
    fn request_can_change_without_rebuilding_for_same_effective_count() {
        let transition = sample_count_transition(8, 4, &[1, 2, 4]);
        assert_eq!(transition.requested_sample_count, 8);
        assert_eq!(transition.effective_sample_count, 4);
        assert!(!transition.rebuild_pipelines);
    }

    #[test]
    fn effective_transition_requires_pipeline_rebuild() {
        let transition = sample_count_transition(4, 1, &[1, 2, 4]);
        assert_eq!(transition.requested_sample_count, 4);
        assert_eq!(transition.effective_sample_count, 4);
        assert!(transition.rebuild_pipelines);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn resize_recreates_size_targets_once_and_preserves_aa_state() {
        let Ok(gpu) = Gpu::new_headless() else {
            return;
        };
        let mut renderer = super::Renderer::new_with_config(
            &gpu,
            OFFSCREEN_FORMAT,
            RendererConfig {
                requested_sample_count: 8,
            },
        )
        .unwrap();
        renderer.set_smaa_enabled(&gpu, true);

        let device_identity = std::ptr::addr_of!(gpu.device);
        let adapter_identity = std::ptr::addr_of!(gpu.adapter);
        let requested = renderer.requested_sample_count();
        let effective = renderer.effective_sample_count();

        assert!(!renderer.resize(&gpu, (0, 0)));
        let first_plan = renderer.target_resize_plan((32, 24));
        assert!(first_plan.recreate_depth);
        assert_eq!(first_plan.recreate_msaa, effective > 1);
        assert!(first_plan.recreate_smaa);
        assert!(renderer.resize(&gpu, (32, 24)));
        assert_eq!(
            renderer.target_resize_plan((32, 24)),
            TargetResizePlan::default()
        );
        assert!(!renderer.resize(&gpu, (32, 24)));

        assert!(renderer.resize(&gpu, (64, 24)));
        assert_eq!(
            renderer.target_resize_plan((64, 24)),
            TargetResizePlan::default()
        );
        assert_eq!(renderer.requested_sample_count(), requested);
        assert_eq!(renderer.effective_sample_count(), effective);
        assert!(renderer.smaa_enabled());
        assert_eq!(std::ptr::addr_of!(gpu.device), device_identity);
        assert_eq!(std::ptr::addr_of!(gpu.adapter), adapter_identity);

        renderer.suspend();
        let restore_plan = renderer.target_resize_plan((64, 24));
        assert!(restore_plan.recreate_depth);
        assert_eq!(restore_plan.recreate_msaa, effective > 1);
        assert!(restore_plan.recreate_smaa);
        assert!(renderer.resize(&gpu, (64, 24)));
        assert_eq!(renderer.requested_sample_count(), requested);
        assert_eq!(renderer.effective_sample_count(), effective);
        assert!(renderer.smaa_enabled());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn renderer_shaders_pass_cpu_validation() {
        for (label, source) in [
            ("world.wgsl", include_str!("shaders/world.wgsl")),
            ("model.wgsl", include_str!("shaders/model.wgsl")),
            ("mtoon.wgsl", include_str!("shaders/mtoon.wgsl")),
            ("sprite.wgsl", include_str!("shaders/sprite.wgsl")),
            ("hud.wgsl", include_str!("shaders/hud.wgsl")),
        ] {
            let module = parse_str(source).unwrap_or_else(|error| panic!("{label}: {error}"));
            Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
        }

        for (label, fragment) in [
            ("smaa_edge.wgsl", include_str!("shaders/smaa_edge.wgsl")),
            (
                "smaa_weights.wgsl",
                include_str!("shaders/smaa_weights.wgsl"),
            ),
            (
                "smaa_neighborhood.wgsl",
                include_str!("shaders/smaa_neighborhood.wgsl"),
            ),
        ] {
            let source = format!("{}\n{fragment}", include_str!("shaders/smaa_common.wgsl"));
            let module = parse_str(&source).unwrap_or_else(|error| panic!("{label}: {error}"));
            Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
        }
    }

    #[test]
    fn gpu_uniform_structs_retain_wgsl_alignment() {
        assert_eq!(std::mem::size_of::<GlobalsRaw>(), 352);
        assert_eq!(std::mem::size_of::<InstanceRaw>(), 160);
        assert_eq!(std::mem::align_of::<GlobalsRaw>(), 4);
        assert_eq!(std::mem::align_of::<InstanceRaw>(), 4);
    }

    #[test]
    fn normal_matrix_preserves_tangent_orthogonality_after_nonuniform_scale() {
        let tangent = Vec3::new(1.0, 1.0, 0.0).normalize();
        let normal = Vec3::new(1.0, -1.0, 0.0).normalize();
        let model = glam::Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 0.5, 3.0),
            Quat::from_rotation_y(0.7),
            Vec3::new(4.0, 5.0, 6.0),
        );

        let world_tangent = model.transform_vector3(tangent).normalize();
        let world_normal = model_normal_matrix(model)
            .transform_vector3(normal)
            .normalize();
        assert!(world_tangent.dot(world_normal).abs() < 1e-5);

        let old_world_normal = model.transform_vector3(normal).normalize();
        assert!(world_tangent.dot(old_world_normal).abs() > 0.1);
    }

    #[test]
    fn singular_normal_matrix_fallback_stays_finite() {
        let normal = model_normal_matrix(glam::Mat4::from_scale(Vec3::new(1.0, 0.0, 2.0)));
        assert!(normal.to_cols_array().iter().all(|value| value.is_finite()));
    }

    fn screen_outline_ndc_offset(
        view_proj: Mat4,
        world_pos: Vec3,
        world_normal: Vec3,
        viewport: (u32, u32),
        width_ratio: f32,
    ) -> Vec2 {
        let clip = view_proj * world_pos.extend(1.0);
        let normal_clip = view_proj * world_normal.normalize().extend(0.0);
        let derivative = (normal_clip.truncate().truncate() * clip.w
            - clip.truncate().truncate() * normal_clip.w)
            / (clip.w * clip.w).max(1e-12);
        let aspect = viewport.0 as f32 / viewport.1 as f32;
        let height_direction = Vec2::new(derivative.x * aspect, derivative.y);
        if height_direction.length_squared() <= 1e-12 {
            return Vec2::ZERO;
        }
        let direction = height_direction.normalize();
        2.0 * width_ratio * Vec2::new(direction.x / aspect, direction.y)
    }

    fn pixel_length(ndc_offset: Vec2, viewport: (u32, u32)) -> f32 {
        Vec2::new(
            ndc_offset.x * viewport.0 as f32 * 0.5,
            ndc_offset.y * viewport.1 as f32 * 0.5,
        )
        .length()
    }

    #[test]
    fn screen_outline_is_a_viewport_height_ratio_for_distance_resolution_fov_and_shift() {
        let width_ratio = 0.0125;
        let world_normal = Vec3::new(1.0, 0.3, 0.4).normalize();
        let cases: [(Vec3, (u32, u32), f32, Vec2); 4] = [
            (Vec3::new(0.2, -0.1, -2.0), (800, 600), 40.0, Vec2::ZERO),
            (Vec3::new(0.2, -0.1, -20.0), (800, 600), 90.0, Vec2::ZERO),
            (
                Vec3::new(0.2, -0.1, -5.0),
                (450, 600),
                70.0,
                Vec2::new(0.3, -0.2),
            ),
            (
                Vec3::new(0.2, -0.1, -5.0),
                (1200, 900),
                55.0,
                Vec2::new(-0.4, 0.25),
            ),
        ];
        for (position, viewport, fov, shift) in cases {
            let aspect = viewport.0 as f32 / viewport.1 as f32;
            let mut projection =
                glam::camera::rh::proj::directx::perspective(fov.to_radians(), aspect, 0.1, 100.0);
            // Off-axis perspective terms. The derivative formulation must not
            // assume a centered projection.
            projection.z_axis.x += shift.x;
            projection.z_axis.y += shift.y;
            let offset = screen_outline_ndc_offset(
                projection,
                position,
                world_normal,
                viewport,
                width_ratio,
            );
            let measured_ratio = pixel_length(offset, viewport) / viewport.1 as f32;
            assert!(
                (measured_ratio - width_ratio).abs() < 1e-6,
                "{position:?} {viewport:?} fov {fov} shift {shift:?}: {measured_ratio}"
            );
        }

        let projection =
            glam::camera::rh::proj::directx::perspective(70.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let low = screen_outline_ndc_offset(
            projection,
            Vec3::new(0.2, -0.1, -5.0),
            world_normal,
            (600, 600),
            width_ratio,
        );
        let high = screen_outline_ndc_offset(
            projection,
            Vec3::new(0.2, -0.1, -5.0),
            world_normal,
            (1200, 1200),
            width_ratio,
        );
        assert!(
            (pixel_length(high, (1200, 1200)) / pixel_length(low, (600, 600)) - 2.0).abs() < 1e-6
        );
    }

    #[test]
    fn world_outline_is_metric_under_nonuniform_scale_and_projects_with_perspective() {
        let factor = 0.1;
        let model = Mat4::from_scale_rotation_translation(
            Vec3::new(4.0, 0.25, 2.0),
            Quat::from_rotation_y(0.4),
            Vec3::ZERO,
        );
        let world_normal = model_normal_matrix(model)
            .transform_vector3(Vec3::new(1.0, 0.5, 0.25))
            .normalize();
        assert!(((world_normal * factor).length() - factor).abs() < 1e-6);

        let projected_width = |distance: f32, fov: f32| {
            let projection =
                glam::camera::rh::proj::directx::perspective(fov.to_radians(), 1.0, 0.1, 100.0);
            let position = Vec3::new(0.0, 0.0, -distance);
            let a = projection * position.extend(1.0);
            let b = projection * (position + Vec3::X * factor).extend(1.0);
            (b.truncate().truncate() / b.w - a.truncate().truncate() / a.w).length()
        };
        assert!(projected_width(2.0, 70.0) > projected_width(10.0, 70.0) * 4.9);
        assert!(projected_width(5.0, 40.0) > projected_width(5.0, 90.0));
    }

    #[test]
    fn outline_shader_locks_vertex_lod_green_channel_and_minimal_depth_bias() {
        let shader = include_str!("shaders/mtoon.wgsl");
        assert!(shader.contains("textureSampleLevel("));
        assert!(shader.contains(").g;"));
        assert!(shader.contains("@vertex\nfn vs_outline"));
        assert!(shader.contains("@fragment\nfn fs_outline"));
        let clip_z = 0.42_f32;
        let clip_w = 2.5_f32;
        let original_ndc = clip_z / clip_w;
        let biased_ndc = (clip_z + 1e-6 * clip_w) / clip_w;
        assert!((biased_ndc - original_ndc - 1e-6).abs() < 1e-7);
    }
}
