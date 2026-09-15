//! Texture upload with CPU mip generation.

use crate::gpu::Gpu;

pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: (u32, u32),
}

/// Filtering is chosen by texture meaning, not by the bytes of its source image.
/// Legacy uploads continue using their established byte-space mip chain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum MipSemantic {
    Legacy,
    SrgbColor,
    LinearData,
    TangentNormal,
}

/// Upload an RGBA8 image, optionally generating a full mip chain on the CPU
/// (simple box filter — plenty for retro-density textures).
pub fn create_rgba_texture(
    gpu: &Gpu,
    label: &str,
    width: u32,
    height: u32,
    rgba: &[u8],
    srgb: bool,
    mips: bool,
) -> GpuTexture {
    create_rgba_texture_with_semantic(
        gpu,
        label,
        width,
        height,
        rgba,
        srgb,
        mips,
        MipSemantic::Legacy,
    )
}

pub(crate) fn create_rgba_texture_with_semantic(
    gpu: &Gpu,
    label: &str,
    width: u32,
    height: u32,
    rgba: &[u8],
    srgb: bool,
    mips: bool,
    semantic: MipSemantic,
) -> GpuTexture {
    assert!(
        semantic == MipSemantic::Legacy || srgb == (semantic == MipSemantic::SrgbColor),
        "{label}: texture format and mip semantic disagree"
    );
    assert_eq!(
        rgba.len(),
        (width * height * 4) as usize,
        "{label}: bad rgba size"
    );
    let mip_level_count = if mips {
        32 - width.max(height).leading_zeros()
    } else {
        1
    };
    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut level_data: Vec<u8> = rgba.to_vec();
    let (mut w, mut h) = (width, height);
    for level in 0..mip_level_count {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &level_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        if level + 1 < mip_level_count {
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            level_data = downsample_semantic(&level_data, w, h, nw, nh, semantic);
            (w, h) = (nw, nh);
        }
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    GpuTexture {
        texture,
        view,
        size: (width, height),
    }
}

fn srgb_to_linear(encoded: f32) -> f32 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

pub(crate) fn downsample_semantic(
    src: &[u8],
    w: u32,
    h: u32,
    nw: u32,
    nh: u32,
    semantic: MipSemantic,
) -> Vec<u8> {
    if semantic == MipSemantic::Legacy {
        return downsample_legacy(src, w, h, nw, nh);
    }
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = [0.0f32; 4];
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    let si = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        let value = src[si + c] as f32 / 255.0;
                        acc[c] += match semantic {
                            MipSemantic::SrgbColor if c < 3 => srgb_to_linear(value),
                            MipSemantic::TangentNormal if c < 3 => value * 2.0 - 1.0,
                            _ => value,
                        };
                    }
                }
            }
            let di = ((y * nw + x) * 4) as usize;
            let mut filtered = acc.map(|value| value * 0.25);
            if semantic == MipSemantic::TangentNormal {
                let length = (filtered[0] * filtered[0]
                    + filtered[1] * filtered[1]
                    + filtered[2] * filtered[2])
                    .sqrt();
                if length > 1e-6 {
                    for value in &mut filtered[..3] {
                        *value /= length;
                    }
                } else {
                    filtered[..3].copy_from_slice(&[0.0, 0.0, 1.0]);
                }
            }
            for c in 0..4 {
                let value = match semantic {
                    MipSemantic::SrgbColor if c < 3 => linear_to_srgb(filtered[c]),
                    MipSemantic::TangentNormal if c < 3 => filtered[c] * 0.5 + 0.5,
                    _ => filtered[c],
                };
                out[di + c] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

fn downsample_legacy(src: &[u8], w: u32, h: u32, nw: u32, nh: u32) -> Vec<u8> {
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = [0u32; 4];
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    let si = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += src[si + c] as u32;
                    }
                }
            }
            let di = ((y * nw + x) * 4) as usize;
            for c in 0..4 {
                out[di + c] = (acc[c] / 4) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod mip_tests {
    use super::{MipSemantic, downsample_semantic};

    #[test]
    fn semantic_mips_filter_color_alpha_data_and_normals() {
        let color = [
            0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255,
        ];
        let srgb = downsample_semantic(&color, 2, 2, 1, 1, MipSemantic::SrgbColor);
        let data = downsample_semantic(&color, 2, 2, 1, 1, MipSemantic::LinearData);
        assert!((srgb[0] as i32 - 188).abs() <= 1);
        assert_eq!(srgb[3], 128);
        assert_eq!(data[0], 128);
        let normals = [
            255, 128, 128, 255, 128, 255, 128, 255, 128, 128, 255, 255, 128, 128, 255, 255,
        ];
        let mip = downsample_semantic(&normals, 2, 2, 1, 1, MipSemantic::TangentNormal);
        let v: Vec<f32> = mip[..3]
            .iter()
            .map(|byte| *byte as f32 / 255.0 * 2.0 - 1.0)
            .collect();
        let length = v
            .iter()
            .map(|component| component * component)
            .sum::<f32>()
            .sqrt();
        assert!((length - 1.0).abs() < 0.02);
    }
}

/// Standard samplers shared across passes.
pub struct Samplers {
    /// Trilinear + anisotropic, repeat — world albedo and model textures.
    pub aniso_repeat: wgpu::Sampler,
    /// Bilinear clamp — lightmap pages.
    pub linear_clamp: wgpu::Sampler,
}

impl Samplers {
    pub fn new(gpu: &Gpu) -> Self {
        let aniso_repeat = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("aniso repeat"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });
        let linear_clamp = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self {
            aniso_repeat,
            linear_clamp,
        }
    }
}
