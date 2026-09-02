//! Presentation-boundary helpers for transparent Windows surfaces.

#[cfg(target_os = "windows")]
use crate::gpu::Gpu;

pub(crate) const LOGICAL_OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
pub(crate) const PHYSICAL_SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

#[cfg(test)]
const PACK_EPSILON: f32 = 1.0e-5;
#[cfg(test)]
const SRGB_BREAKPOINT: f32 = 0.0031308;

#[cfg(test)]
fn linear_to_srgb(value: f32) -> f32 {
    if value <= SRGB_BREAKPOINT {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert linear-premultiplied RGBA into encoded-premultiplied RGBA.
///
/// The GPU pack shader applies the same operation after sampling the sRGB
/// logical output texture. Keeping this scalar version here makes the
/// presentation math independently testable without constructing a GPU.
#[cfg(test)]
pub(crate) fn pack_pixel(input: [f32; 4]) -> [f32; 4] {
    let alpha = input[3].clamp(0.0, 1.0);
    if alpha <= PACK_EPSILON {
        return [0.0; 4];
    }

    let straight_linear = [
        (input[0] / alpha).clamp(0.0, 1.0),
        (input[1] / alpha).clamp(0.0, 1.0),
        (input[2] / alpha).clamp(0.0, 1.0),
    ];
    let encoded_straight = straight_linear.map(linear_to_srgb);

    [
        (encoded_straight[0] * alpha).clamp(0.0, alpha),
        (encoded_straight[1] * alpha).clamp(0.0, alpha),
        (encoded_straight[2] * alpha).clamp(0.0, alpha),
        alpha,
    ]
}

#[cfg(target_os = "windows")]
pub(crate) struct TransparentPresentation {
    packer: PresentationPacker,
    logical_output: LogicalOutputTarget,
    use_exact_load: bool,
}

#[cfg(target_os = "windows")]
impl TransparentPresentation {
    pub(crate) fn new(gpu: &Gpu, size: (u32, u32)) -> Self {
        assert!(
            size.0 != 0 && size.1 != 0,
            "logical output must be non-zero"
        );
        let packer = PresentationPacker::new(gpu);
        let logical_output = LogicalOutputTarget::new(gpu, size, &packer);
        Self {
            packer,
            logical_output,
            use_exact_load: true,
        }
    }

    pub(crate) fn resize(&mut self, gpu: &Gpu, size: (u32, u32)) {
        assert!(
            size.0 != 0 && size.1 != 0,
            "logical output must be non-zero"
        );
        self.logical_output = LogicalOutputTarget::new(gpu, size, &self.packer);
    }

    pub(crate) fn view(&self) -> &wgpu::TextureView {
        &self.logical_output.view
    }

    pub(crate) fn toggle(&mut self) -> bool {
        self.use_exact_load = !self.use_exact_load;
        self.use_exact_load
    }

    pub(crate) fn pack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        physical_surface: &wgpu::TextureView,
    ) {
        self.packer.pack(
            encoder,
            &self.logical_output,
            self.use_exact_load,
            physical_surface,
        );
    }
}

#[cfg(target_os = "windows")]
struct LogicalOutputTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    exact_bind_group: wgpu::BindGroup,
    filtered_bind_group: wgpu::BindGroup,
}

#[cfg(target_os = "windows")]
impl LogicalOutputTarget {
    fn new(gpu: &Gpu, size: (u32, u32), packer: &PresentationPacker) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("transparent logical output"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LOGICAL_OUTPUT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("transparent logical output sRGB view"),
            ..Default::default()
        });
        let exact_bind_group = packer.create_bind_group(gpu, &view, true);
        let filtered_bind_group = packer.create_bind_group(gpu, &view, false);
        Self {
            _texture: texture,
            view,
            exact_bind_group,
            filtered_bind_group,
        }
    }
}

#[cfg(target_os = "windows")]
struct PresentationPacker {
    exact: PresentationPackPipeline,
    filtered: PresentationPackPipeline,
}

#[cfg(target_os = "windows")]
struct PresentationPackPipeline {
    pipeline: wgpu::RenderPipeline,
    source_layout: wgpu::BindGroupLayout,
    sampler: Option<wgpu::Sampler>,
}

#[cfg(target_os = "windows")]
impl PresentationPacker {
    fn new(gpu: &Gpu) -> Self {
        Self {
            exact: PresentationPackPipeline::new(
                gpu,
                include_str!("shaders/presentation_pack.wgsl"),
                false,
            ),
            filtered: PresentationPackPipeline::new(
                gpu,
                include_str!("shaders/presentation_pack_filtered.wgsl"),
                true,
            ),
        }
    }

    fn create_bind_group(
        &self,
        gpu: &Gpu,
        source_view: &wgpu::TextureView,
        use_exact_load: bool,
    ) -> wgpu::BindGroup {
        if use_exact_load {
            self.exact.create_bind_group(gpu, source_view)
        } else {
            self.filtered.create_bind_group(gpu, source_view)
        }
    }

    fn pack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        logical_output: &LogicalOutputTarget,
        use_exact_load: bool,
        physical_surface: &wgpu::TextureView,
    ) {
        if use_exact_load {
            self.exact
                .pack(encoder, &logical_output.exact_bind_group, physical_surface);
        } else {
            self.filtered.pack(
                encoder,
                &logical_output.filtered_bind_group,
                physical_surface,
            );
        }
    }
}

#[cfg(target_os = "windows")]
impl PresentationPackPipeline {
    fn new(gpu: &Gpu, shader_source: &str, filtered: bool) -> Self {
        let device = &gpu.device;
        let mut source_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        let sampler = if filtered {
            source_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
            Some(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("transparent presentation pack sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }))
        } else {
            None
        };
        let source_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transparent presentation pack source"),
            entries: &source_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("transparent presentation pack layout"),
            bind_group_layouts: &[&source_layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("transparent presentation pack shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("transparent presentation pack"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: PHYSICAL_SURFACE_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Self {
            pipeline,
            source_layout,
            sampler,
        }
    }

    fn create_bind_group(&self, gpu: &Gpu, source_view: &wgpu::TextureView) -> wgpu::BindGroup {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(source_view),
        }];
        if let Some(sampler) = &self.sampler {
            entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transparent presentation pack source bind group"),
            layout: &self.source_layout,
            entries: &entries,
        })
    }

    fn pack(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        source_bind_group: &wgpu::BindGroup,
        physical_surface: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("transparent presentation pack pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: physical_surface,
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, source_bind_group, &[]);
        // The fullscreen triangle covers every pixel, so the non-blended
        // output overwrites the complete physical surface view.
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::{LOGICAL_OUTPUT_FORMAT, PHYSICAL_SURFACE_FORMAT, pack_pixel};
    #[cfg(not(target_arch = "wasm32"))]
    use wgpu::naga::{
        front::wgsl::parse_str,
        valid::{Capabilities, ValidationFlags, Validator},
    };

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "actual {actual} differs from expected {expected}"
        );
    }

    fn linear_to_srgb(value: f32) -> f32 {
        if value <= 0.0031308 {
            12.92 * value
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        }
    }

    #[test]
    fn opaque_input_is_ordinarily_srgb_encoded() {
        let packed = pack_pixel([0.25, 0.5, 1.0, 1.0]);
        assert_close(packed[0], linear_to_srgb(0.25));
        assert_close(packed[1], linear_to_srgb(0.5));
        assert_close(packed[2], 1.0);
        assert_eq!(packed[3], 1.0);
    }

    #[test]
    fn fractional_alpha_encodes_straight_color_before_premultiplying() {
        let alpha = 0.5;
        let straight_linear = [0.25, 0.5, 0.75];
        let packed = pack_pixel([
            straight_linear[0] * alpha,
            straight_linear[1] * alpha,
            straight_linear[2] * alpha,
            alpha,
        ]);

        for (index, color) in straight_linear.into_iter().enumerate() {
            assert_close(packed[index], alpha * linear_to_srgb(color));
        }
        assert!((packed[0] - linear_to_srgb(straight_linear[0] * alpha)).abs() > 0.01);
        assert_eq!(packed[3], alpha);
    }

    #[test]
    fn zero_alpha_is_exactly_zero() {
        assert_eq!(pack_pixel([0.8, 0.2, 1.0, 0.0]), [0.0; 4]);
    }

    #[test]
    fn packed_rgb_never_exceeds_alpha() {
        for input in [[2.0, -1.0, 0.5, 0.2], [0.1, 0.8, 1.0, 0.75]] {
            let packed = pack_pixel(input);
            assert!(packed[..3].iter().all(|&channel| channel <= packed[3]));
        }
    }

    #[test]
    fn presentation_formats_are_distinct_srgb_source_and_unorm_destination() {
        assert_eq!(LOGICAL_OUTPUT_FORMAT, wgpu::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(PHYSICAL_SURFACE_FORMAT, wgpu::TextureFormat::Bgra8Unorm);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn presentation_pack_shader_passes_cpu_validation() {
        for (label, source) in [
            (
                "presentation_pack.wgsl",
                include_str!("shaders/presentation_pack.wgsl"),
            ),
            (
                "presentation_pack_filtered.wgsl",
                include_str!("shaders/presentation_pack_filtered.wgsl"),
            ),
        ] {
            let module =
                parse_str(source).unwrap_or_else(|error| panic!("{label} must parse: {error}"));
            Validator::new(ValidationFlags::all(), Capabilities::empty())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{label} must validate: {error}"));
        }
    }
}
