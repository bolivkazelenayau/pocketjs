//! Isolated SMAA 1x post-process pass.
//!
//! The shader structure and lookup-table generation are adapted from
//! `iryoku/smaa` (2013), by Jorge Jimenez, Jose I. Echevarria, Belen Masia,
//! Fernando Navarro, and Diego Gutierrez. The upstream implementation and
//! supporting scripts are MIT licensed. This port keeps the required notice
//! here and generates the lookup tables at runtime instead of copying the
//! upstream binary assets into this repository.
//!
//! ```text
//! Copyright (C) 2013 Jorge Jimenez (jorge@iryoku.com)
//! Copyright (C) 2013 Jose I. Echevarria (joseignacioechevarria@gmail.com)
//! Copyright (C) 2013 Belen Masia (bmasia@unizar.es)
//! Copyright (C) 2013 Fernando Navarro (fernandn@microsoft.com)
//! Copyright (C) 2013 Diego Gutierrez (diegog@unizar.es)
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software. As clarification, there
//! is no requirement that the copyright notice and permission be included in
//! binary distributions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.
//! ```

use bytemuck::{Pod, Zeroable};

use crate::gpu::Gpu;

const ORTHO_SUBTEX_SIZE: usize = 16;
const ORTHO_SLOTS: usize = 5;
const DIAG_SUBTEX_SIZE: usize = 20;
const DIAG_SLOTS: usize = 4;
const AREA_WIDTH: u32 = (ORTHO_SUBTEX_SIZE * ORTHO_SLOTS * 2) as u32;
const AREA_HEIGHT: u32 = (ORTHO_SUBTEX_SIZE * ORTHO_SLOTS * 7) as u32;
const SEARCH_WIDTH: u32 = 64;
const SEARCH_HEIGHT: u32 = 16;

const ORTHO_SUBSAMPLE_OFFSETS: [f64; 7] = [0.0, -0.25, 0.25, -0.125, 0.125, -0.375, 0.375];
const DIAG_SUBSAMPLE_OFFSETS: [(f64, f64); 5] = [
    (0.0, 0.0),
    (0.25, -0.25),
    (-0.25, 0.25),
    (0.125, -0.125),
    (-0.125, 0.125),
];

const ORTHO_PATTERN_SLOTS: [(usize, usize); 16] = [
    (0, 0),
    (3, 0),
    (0, 3),
    (3, 3),
    (1, 0),
    (4, 0),
    (1, 3),
    (4, 3),
    (0, 1),
    (3, 1),
    (0, 4),
    (3, 4),
    (1, 1),
    (4, 1),
    (1, 4),
    (4, 4),
];

const DIAG_PATTERN_SLOTS: [(usize, usize); 16] = [
    (0, 0),
    (1, 0),
    (0, 2),
    (1, 2),
    (2, 0),
    (3, 0),
    (2, 2),
    (3, 2),
    (0, 1),
    (1, 1),
    (0, 3),
    (1, 3),
    (2, 1),
    (3, 1),
    (2, 3),
    (3, 3),
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MetricsRaw {
    texel_size: [f32; 2],
    alpha_edge: f32,
    _pad: f32,
}

struct SmaaTargets {
    _scene_texture: wgpu::Texture,
    scene_view: wgpu::TextureView,
    _scene_edge_view: wgpu::TextureView,
    _edges_texture: wgpu::Texture,
    edges_view: wgpu::TextureView,
    _weights_texture: wgpu::Texture,
    weights_view: wgpu::TextureView,
    edge_bg: wgpu::BindGroup,
    weights_bg: wgpu::BindGroup,
    neighborhood_bg: wgpu::BindGroup,
    size: (u32, u32),
}

/// A narrow canonical SMAA 1x three-pass core with Pocket3D's
/// premultiplied-alpha silhouette extension. It is deliberately owned by the
/// renderer as one unit so resize only replaces the size-dependent targets and
/// bind groups; shader pipelines and lookup textures live for the pass lifetime.
pub(crate) struct SmaaPass {
    color_format: wgpu::TextureFormat,
    metrics: wgpu::Buffer,
    linear_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,
    _area_texture: wgpu::Texture,
    area_view: wgpu::TextureView,
    _search_texture: wgpu::Texture,
    search_view: wgpu::TextureView,
    edge_bgl: wgpu::BindGroupLayout,
    weights_bgl: wgpu::BindGroupLayout,
    neighborhood_bgl: wgpu::BindGroupLayout,
    edge_pipeline: wgpu::RenderPipeline,
    weights_pipeline: wgpu::RenderPipeline,
    neighborhood_pipeline: wgpu::RenderPipeline,
    targets: Option<SmaaTargets>,
}

impl SmaaPass {
    pub(crate) fn new(gpu: &Gpu, color_format: wgpu::TextureFormat) -> Self {
        let device = &gpu.device;

        let metrics = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smaa metrics"),
            size: std::mem::size_of::<MetricsRaw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("smaa linear clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("smaa point clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (area_texture, area_view) = create_lut_texture(
            gpu,
            "smaa area lookup",
            wgpu::TextureFormat::Rgba8Unorm,
            AREA_WIDTH,
            AREA_HEIGHT,
            &build_area_texture(),
        );
        let (search_texture, search_view) = create_lut_texture(
            gpu,
            "smaa search lookup",
            wgpu::TextureFormat::R8Unorm,
            SEARCH_WIDTH,
            SEARCH_HEIGHT,
            &build_search_texture(),
        );

        let edge_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smaa edge bind group layout"),
            entries: &[texture_entry(0), sampler_entry(1), uniform_entry(2)],
        });
        let weights_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smaa weights bind group layout"),
            entries: &[
                texture_entry(0),
                sampler_entry(1),
                texture_entry(2),
                sampler_entry(3),
                texture_entry(4),
                sampler_entry(5),
                uniform_entry(6),
            ],
        });
        let neighborhood_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smaa neighborhood bind group layout"),
            entries: &[
                texture_entry(0),
                sampler_entry(1),
                texture_entry(2),
                sampler_entry(3),
                uniform_entry(4),
            ],
        });

        let common = include_str!("shaders/smaa_common.wgsl");
        let edge_shader = create_shader_module(
            device,
            "smaa edge shader",
            format!("{common}\n{}", include_str!("shaders/smaa_edge.wgsl")),
        );
        let weights_shader = create_shader_module(
            device,
            "smaa weights shader",
            format!("{common}\n{}", include_str!("shaders/smaa_weights.wgsl")),
        );
        let neighborhood_shader = create_shader_module(
            device,
            "smaa neighborhood shader",
            format!(
                "{common}\n{}",
                include_str!("shaders/smaa_neighborhood.wgsl")
            ),
        );
        let edge_pipeline = create_pipeline(
            device,
            "smaa edge pipeline",
            &edge_shader,
            &[&edge_bgl],
            wgpu::TextureFormat::Rgba8Unorm,
            "fs_edge",
        );
        let weights_pipeline = create_pipeline(
            device,
            "smaa weights pipeline",
            &weights_shader,
            &[&weights_bgl],
            wgpu::TextureFormat::Rgba8Unorm,
            "fs_weights",
        );
        let neighborhood_pipeline = create_pipeline(
            device,
            "smaa neighborhood pipeline",
            &neighborhood_shader,
            &[&neighborhood_bgl],
            color_format,
            "fs_neighborhood",
        );

        Self {
            color_format,
            metrics,
            linear_sampler,
            point_sampler,
            _area_texture: area_texture,
            area_view,
            _search_texture: search_texture,
            search_view,
            edge_bgl,
            weights_bgl,
            neighborhood_bgl,
            edge_pipeline,
            weights_pipeline,
            neighborhood_pipeline,
            targets: None,
        }
    }

    pub(crate) fn ensure_targets(&mut self, gpu: &Gpu, size: (u32, u32)) {
        if self
            .targets
            .as_ref()
            .is_some_and(|targets| targets.size == size)
        {
            return;
        }

        let device = &gpu.device;
        let non_srgb_color_format = non_srgb_variant(self.color_format);
        let scene_view_formats = if non_srgb_color_format == self.color_format {
            &[][..]
        } else {
            std::slice::from_ref(&non_srgb_color_format)
        };
        let scene_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("smaa scene color"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: scene_view_formats,
        });
        let scene_view = scene_texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Canonical SMAA luma edge detection consumes gamma-encoded values,
        // while the neighborhood pass samples the default sRGB view below so
        // the API performs exactly one decode before premultiplied filtering.
        let scene_edge_view = scene_texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(non_srgb_color_format),
            ..Default::default()
        });

        let edges_texture =
            create_target_texture(device, "smaa edges", size, wgpu::TextureFormat::Rgba8Unorm);
        let edges_view = edges_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let weights_texture = create_target_texture(
            device,
            "smaa blend weights",
            size,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let weights_view = weights_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let edge_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smaa edge bind group"),
            layout: &self.edge_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_edge_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.point_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.metrics.as_entire_binding(),
                },
            ],
        });
        let weights_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smaa weights bind group"),
            layout: &self.weights_bgl,
            entries: &[
                texture_view_entry(0, &edges_view),
                sampler_entry_resource(1, &self.linear_sampler),
                texture_view_entry(2, &self.area_view),
                sampler_entry_resource(3, &self.linear_sampler),
                texture_view_entry(4, &self.search_view),
                sampler_entry_resource(5, &self.linear_sampler),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.metrics.as_entire_binding(),
                },
            ],
        });
        let neighborhood_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smaa neighborhood bind group"),
            layout: &self.neighborhood_bgl,
            entries: &[
                texture_view_entry(0, &scene_view),
                sampler_entry_resource(1, &self.linear_sampler),
                texture_view_entry(2, &weights_view),
                sampler_entry_resource(3, &self.point_sampler),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.metrics.as_entire_binding(),
                },
            ],
        });

        self.targets = Some(SmaaTargets {
            _scene_texture: scene_texture,
            scene_view,
            _scene_edge_view: scene_edge_view,
            _edges_texture: edges_texture,
            edges_view,
            _weights_texture: weights_texture,
            weights_view,
            edge_bg,
            weights_bg,
            neighborhood_bg,
            size,
        });
    }

    pub(crate) fn scene_view(&self) -> &wgpu::TextureView {
        &self
            .targets
            .as_ref()
            .expect("SMAA targets must be ensured before use")
            .scene_view
    }

    pub(crate) fn update_metrics(&self, gpu: &Gpu, size: (u32, u32), transparent: bool) {
        gpu.queue.write_buffer(
            &self.metrics,
            0,
            bytemuck::bytes_of(&MetricsRaw {
                texel_size: [1.0 / size.0 as f32, 1.0 / size.1 as f32],
                alpha_edge: if transparent { 1.0 } else { 0.0 },
                _pad: 0.0,
            }),
        );
    }

    pub(crate) fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
    ) {
        let targets = self
            .targets
            .as_ref()
            .expect("SMAA targets must be ensured before use");

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("smaa edge detection"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.edges_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.edge_pipeline);
            pass.set_bind_group(0, &targets.edge_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("smaa blending weights"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.weights_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.weights_pipeline);
            pass.set_bind_group(0, &targets.weights_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("smaa neighborhood blending"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.neighborhood_pipeline);
            pass.set_bind_group(0, &targets.neighborhood_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn sampler_entry_resource(binding: u32, sampler: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    }
}

fn create_target_texture(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn create_lut_texture(
    gpu: &Gpu,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * format_block_bytes(format)),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn format_block_bytes(format: wgpu::TextureFormat) -> u32 {
    match format {
        wgpu::TextureFormat::R8Unorm => 1,
        wgpu::TextureFormat::Rgba8Unorm => 4,
        _ => unreachable!("SMAA LUT format must be an uncompressed 8-bit format"),
    }
}

fn non_srgb_variant(format: wgpu::TextureFormat) -> wgpu::TextureFormat {
    match format {
        wgpu::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8Unorm,
        other => other,
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    format: wgpu::TextureFormat,
    fragment_entry: &str,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts,
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fragment_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn create_shader_module(device: &wgpu::Device, label: &str, source: String) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

#[derive(Clone, Copy)]
struct Area2 {
    first: f64,
    second: f64,
}

impl Area2 {
    const ZERO: Self = Self {
        first: 0.0,
        second: 0.0,
    };

    fn add(self, other: Self) -> Self {
        Self {
            first: self.first + other.first,
            second: self.second + other.second,
        }
    }
}

fn line_area(p1: (f64, f64), p2: (f64, f64), x: f64) -> Area2 {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    let x1 = x;
    let x2 = x + 1.0;
    let y1 = p1.1 + dy * (x1 - p1.0) / dx;
    let y2 = p1.1 + dy * (x2 - p1.0) / dx;
    let inside = (x1 >= p1.0 && x1 < p2.0) || (x2 > p1.0 && x2 <= p2.0);
    if !inside {
        return Area2::ZERO;
    }

    let same_sign =
        y1.is_sign_positive() == y2.is_sign_positive() || y1.abs() < 1e-4 || y2.abs() < 1e-4;
    if same_sign {
        let area = (y1 + y2) * 0.5;
        if area < 0.0 {
            Area2 {
                first: area.abs(),
                second: 0.0,
            }
        } else {
            Area2 {
                first: 0.0,
                second: area.abs(),
            }
        }
    } else {
        let crossing_x = -p1.1 * dx / dy + p1.0;
        let fraction = crossing_x.fract();
        let area1 = if crossing_x > p1.0 {
            y1 * fraction * 0.5
        } else {
            0.0
        };
        let area2 = if crossing_x < p2.0 {
            y2 * (1.0 - fraction) * 0.5
        } else {
            0.0
        };
        let area = if area1.abs() > area2.abs() {
            area1
        } else {
            -area2
        };
        if area < 0.0 {
            Area2 {
                first: area1.abs(),
                second: area2.abs(),
            }
        } else {
            Area2 {
                first: area2.abs(),
                second: area1.abs(),
            }
        }
    }
}

fn smooth_area(distance: f64, area: Area2) -> Area2 {
    let b = Area2 {
        first: (area.first * 2.0).sqrt() * 0.5,
        second: (area.second * 2.0).sqrt() * 0.5,
    };
    let p = (distance / 32.0).clamp(0.0, 1.0);
    Area2 {
        first: b.first + (area.first - b.first) * p,
        second: b.second + (area.second - b.second) * p,
    }
}

/// The orthogonal AreaTex construction from the upstream generator. The
/// source coordinates are quadratically compressed because the shader samples
/// with sqrt(distance).
fn area_ortho(pattern: usize, left: f64, right: f64, offset: f64) -> Area2 {
    let distance = left + right + 1.0;
    let o1 = 0.5 + offset;
    let o2 = -0.5 + offset;
    let a = |p1: (f64, f64), p2: (f64, f64)| line_area(p1, p2, left);
    match pattern {
        0 => Area2::ZERO,
        1 => {
            if left <= right {
                a((0.0, o2), (distance * 0.5, 0.0))
            } else {
                Area2::ZERO
            }
        }
        2 => {
            if left >= right {
                a((distance * 0.5, 0.0), (distance, o2))
            } else {
                Area2::ZERO
            }
        }
        3 => {
            let a1 = smooth_area(distance, a((0.0, o2), (distance * 0.5, 0.0)));
            let a2 = smooth_area(distance, a((distance * 0.5, 0.0), (distance, o2)));
            a1.add(a2)
        }
        4 => {
            if left <= right {
                a((0.0, o1), (distance * 0.5, 0.0))
            } else {
                Area2::ZERO
            }
        }
        5 => Area2::ZERO,
        6 => {
            if offset.abs() > 0.0 {
                let a1 = a((0.0, o1), (distance, o2));
                let a2 = a((0.0, o1), (distance * 0.5, 0.0))
                    .add(a((distance * 0.5, 0.0), (distance, o2)));
                Area2 {
                    first: (a1.first + a2.first) * 0.5,
                    second: (a1.second + a2.second) * 0.5,
                }
            } else {
                a((0.0, o1), (distance, o2))
            }
        }
        7 => a((0.0, o1), (distance, o2)),
        8 => {
            if left >= right {
                a((distance * 0.5, 0.0), (distance, o1))
            } else {
                Area2::ZERO
            }
        }
        9 => {
            if offset.abs() > 0.0 {
                let a1 = a((0.0, o2), (distance, o1));
                let a2 = a((0.0, o2), (distance * 0.5, 0.0))
                    .add(a((distance * 0.5, 0.0), (distance, o1)));
                Area2 {
                    first: (a1.first + a2.first) * 0.5,
                    second: (a1.second + a2.second) * 0.5,
                }
            } else {
                a((0.0, o2), (distance, o1))
            }
        }
        10 => Area2::ZERO,
        11 => a((0.0, o2), (distance, o1)),
        12 => {
            let a1 = smooth_area(distance, a((0.0, o1), (distance * 0.5, 0.0)));
            let a2 = smooth_area(distance, a((distance * 0.5, 0.0), (distance, o1)));
            a1.add(a2)
        }
        13 => a((0.0, o2), (distance, o1)),
        14 => a((0.0, o1), (distance, o2)),
        15 => Area2::ZERO,
        _ => unreachable!("SMAA orthogonal pattern out of range"),
    }
}

const DIAG_PATTERN_EDGES: [(u8, u8); 16] = [
    (0, 0),
    (1, 0),
    (0, 2),
    (1, 2),
    (2, 0),
    (3, 0),
    (2, 2),
    (3, 2),
    (0, 1),
    (1, 1),
    (0, 3),
    (1, 3),
    (2, 1),
    (3, 1),
    (2, 3),
    (3, 3),
];

fn diagonal_inside(p1: (f64, f64), p2: (f64, f64), p: (f64, f64)) -> bool {
    if p1 != p2 {
        let midpoint = ((p1.0 + p2.0) * 0.5, (p1.1 + p2.1) * 0.5);
        let a = p2.1 - p1.1;
        let b = p1.0 - p2.0;
        a * (p.0 - midpoint.0) + b * (p.1 - midpoint.1) > 0.0
    } else {
        true
    }
}

fn diagonal_area1(p1: (f64, f64), p2: (f64, f64), p: (f64, f64)) -> f64 {
    const SAMPLES: usize = 30;
    let mut covered = 0.0;
    for x in 0..SAMPLES {
        for y in 0..SAMPLES {
            let offset = (
                x as f64 / (SAMPLES - 1) as f64,
                y as f64 / (SAMPLES - 1) as f64,
            );
            if diagonal_inside(p1, p2, (p.0 + offset.0, p.1 + offset.1)) {
                covered += 1.0;
            }
        }
    }
    covered / (SAMPLES * SAMPLES) as f64
}

fn diagonal_line_area(
    pattern: usize,
    p1: (f64, f64),
    p2: (f64, f64),
    left: f64,
    offset: (f64, f64),
) -> Area2 {
    let (edge1, edge2) = DIAG_PATTERN_EDGES[pattern];
    let p1 = if edge1 > 0 {
        (p1.0 + offset.0, p1.1 + offset.1)
    } else {
        p1
    };
    let p2 = if edge2 > 0 {
        (p2.0 + offset.0, p2.1 + offset.1)
    } else {
        p2
    };
    let first = diagonal_area1(p1, p2, (1.0 + left, left));
    let second = diagonal_area1(p1, p2, (1.0 + left, 1.0 + left));
    Area2 {
        first: 1.0 - first,
        second,
    }
}

/// The diagonal AreaTex construction from the upstream generator. Unlike the
/// orthogonal table, the diagonal table is generated with 30x30 brute-force
/// coverage samples per pixel, matching the official AreaTex.py script.
fn area_diag(pattern: usize, left: f64, right: f64, offset: (f64, f64)) -> Area2 {
    let distance = left + right + 1.0;
    let area = |p1: (f64, f64), p2: (f64, f64)| diagonal_line_area(pattern, p1, p2, left, offset);
    match pattern {
        0 => {
            let a1 = area((1.0, 1.0), (1.0 + distance, 1.0 + distance));
            let a2 = area((1.0, 0.0), (1.0 + distance, distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        1 => {
            let a1 = area((1.0, 0.0), (distance, distance));
            let a2 = area((1.0, 0.0), (1.0 + distance, distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        2 => {
            let a1 = area((0.0, 0.0), (1.0 + distance, distance));
            let a2 = area((1.0, 0.0), (1.0 + distance, distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        3 => area((1.0, 0.0), (1.0 + distance, distance)),
        4 => {
            let a1 = area((1.0, 1.0), (distance, distance));
            let a2 = area((1.0, 1.0), (1.0 + distance, distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        5 => {
            let a1 = area((1.0, 1.0), (distance, distance));
            let a2 = area((1.0, 0.0), (1.0 + distance, distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        6 => area((1.0, 1.0), (1.0 + distance, distance)),
        7 => {
            let a1 = area((1.0, 1.0), (1.0 + distance, distance));
            let a2 = area((1.0, 0.0), (1.0 + distance, distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        8 => {
            let a1 = area((0.0, 0.0), (1.0 + distance, 1.0 + distance));
            let a2 = area((1.0, 0.0), (1.0 + distance, 1.0 + distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        9 => area((1.0, 0.0), (1.0 + distance, 1.0 + distance)),
        10 => {
            let a1 = area((0.0, 0.0), (1.0 + distance, 1.0 + distance));
            let a2 = area((1.0, 0.0), (1.0 + distance, distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        11 => {
            let a1 = area((1.0, 0.0), (1.0 + distance, 1.0 + distance));
            let a2 = area((1.0, 0.0), (1.0 + distance, distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        12 => area((1.0, 1.0), (1.0 + distance, 1.0 + distance)),
        13 => {
            let a1 = area((1.0, 1.0), (1.0 + distance, 1.0 + distance));
            let a2 = area((1.0, 0.0), (1.0 + distance, 1.0 + distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        14 => {
            let a1 = area((1.0, 1.0), (1.0 + distance, 1.0 + distance));
            let a2 = area((1.0, 1.0), (1.0 + distance, distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        15 => {
            let a1 = area((1.0, 1.0), (1.0 + distance, 1.0 + distance));
            let a2 = area((1.0, 0.0), (1.0 + distance, distance));
            Area2 {
                first: (a1.first + a2.first) * 0.5,
                second: (a1.second + a2.second) * 0.5,
            }
        }
        _ => unreachable!("SMAA diagonal pattern out of range"),
    }
}

fn encode_area(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0) as u8
}

fn build_area_texture() -> Vec<u8> {
    let mut pixels = vec![0u8; (AREA_WIDTH * AREA_HEIGHT * 4) as usize];
    let width = AREA_WIDTH as usize;

    for (offset_index, &offset) in ORTHO_SUBSAMPLE_OFFSETS.iter().enumerate() {
        for (pattern, &(slot_x, slot_y)) in ORTHO_PATTERN_SLOTS.iter().enumerate() {
            for y in 0..ORTHO_SUBTEX_SIZE {
                for x in 0..ORTHO_SUBTEX_SIZE {
                    let area = area_ortho(pattern, (x * x) as f64, (y * y) as f64, offset);
                    let px = slot_x * ORTHO_SUBTEX_SIZE + x;
                    let py = offset_index * ORTHO_SUBTEX_SIZE * ORTHO_SLOTS
                        + slot_y * ORTHO_SUBTEX_SIZE
                        + y;
                    let dst = (py * width + px) * 4;
                    pixels[dst] = encode_area(area.first);
                    pixels[dst + 1] = encode_area(area.second);
                }
            }
        }
    }

    for (offset_index, &offset) in DIAG_SUBSAMPLE_OFFSETS.iter().enumerate() {
        for (pattern, &(slot_x, slot_y)) in DIAG_PATTERN_SLOTS.iter().enumerate() {
            for y in 0..DIAG_SUBTEX_SIZE {
                for x in 0..DIAG_SUBTEX_SIZE {
                    let area = area_diag(pattern, x as f64, y as f64, offset);
                    let px = ORTHO_SUBTEX_SIZE * ORTHO_SLOTS + slot_x * DIAG_SUBTEX_SIZE + x;
                    let py = offset_index * DIAG_SUBTEX_SIZE * DIAG_SLOTS
                        + slot_y * DIAG_SUBTEX_SIZE
                        + y;
                    let dst = (py * width + px) * 4;
                    pixels[dst] = encode_area(area.first);
                    pixels[dst + 1] = encode_area(area.second);
                }
            }
        }
    }
    pixels
}

fn bilinear_code(bits: [u8; 4]) -> usize {
    bits[0] as usize + 3 * bits[1] as usize + 7 * bits[2] as usize + 21 * bits[3] as usize
}

fn delta_left(left: [u8; 4], top: [u8; 4]) -> u8 {
    let mut distance = u8::from(top[3] == 1);
    if distance == 1 && top[2] == 1 && left[1] != 1 && left[3] != 1 {
        distance += 1;
    }
    distance
}

fn delta_right(left: [u8; 4], top: [u8; 4]) -> u8 {
    let mut distance = u8::from(top[3] == 1 && left[1] != 1 && left[3] != 1);
    if distance == 1 && top[2] == 1 && left[0] != 1 && left[2] != 1 {
        distance += 1;
    }
    distance
}

/// Exact packed SearchTex construction from the upstream SearchTex.py script.
fn build_search_texture() -> Vec<u8> {
    let mut edge_codes = [[0u8; 4]; 33];
    for mask in 0..16u8 {
        let bits = [mask & 1, (mask >> 1) & 1, (mask >> 2) & 1, (mask >> 3) & 1];
        edge_codes[bilinear_code(bits)] = bits;
    }

    let mut unpacked = vec![0u8; 66 * 33];
    for x in 0..33 {
        for y in 0..33 {
            let left = edge_codes[x];
            let top = edge_codes[y];
            if left == [0; 4] && x != 0 || top == [0; 4] && y != 0 {
                continue;
            }
            unpacked[y * 66 + x] = 127 * delta_left(left, top);
            unpacked[y * 66 + 33 + x] = 127 * delta_right(left, top);
        }
    }

    let mut packed = vec![0u8; (SEARCH_WIDTH * SEARCH_HEIGHT) as usize];
    for y in 0..SEARCH_HEIGHT as usize {
        let source_y = 32 - y;
        for x in 0..SEARCH_WIDTH as usize {
            packed[y * SEARCH_WIDTH as usize + x] = unpacked[source_y * 66 + x];
        }
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::{
        AREA_HEIGHT, AREA_WIDTH, SEARCH_HEIGHT, SEARCH_WIDTH, build_area_texture,
        build_search_texture, non_srgb_variant,
    };

    fn fnv1a64(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf29ce484222325, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
    }

    #[test]
    fn lookup_tables_match_official_generator_references() {
        // FNV-1a is used here as a small, dependency-free byte-lock. The
        // logical hashes match the official R8G8 AreaTex.h and R8 SearchTex.h
        // payloads generated by iryoku/smaa's AreaTex.py and SearchTex.py.
        // The final AreaTex hash also locks this port's Rgba8Unorm expansion
        // (the official B/A channels are represented as zeroes here). Keep the
        // dimensions in these assertions so a layout change cannot preserve
        // only a coincidental hash.
        let area = build_area_texture();
        let search = build_search_texture();
        assert_eq!(area.len(), (AREA_WIDTH * AREA_HEIGHT * 4) as usize);
        assert_eq!(search.len(), (SEARCH_WIDTH * SEARCH_HEIGHT) as usize);
        let area_logical: Vec<u8> = area
            .chunks_exact(4)
            .flat_map(|pixel| [pixel[0], pixel[1]])
            .collect();
        assert_eq!(fnv1a64(&area_logical), 0x247a58bbba65292d);
        assert_eq!(fnv1a64(&search), 0x21c1fcf0aa631065);
        assert_eq!(fnv1a64(&area), 0xf74933c6dbb9a9f5);
    }

    #[test]
    fn edge_view_uses_non_srgb_format_for_srgb_targets() {
        assert_eq!(
            non_srgb_variant(wgpu::TextureFormat::Rgba8UnormSrgb),
            wgpu::TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            non_srgb_variant(wgpu::TextureFormat::Bgra8UnormSrgb),
            wgpu::TextureFormat::Bgra8Unorm
        );
    }
}
