//! First-person camera. Right-handed, +Y up; yaw 0 looks down -Z.

use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub pos: Vec3,
    /// Radians around +Y. 0 = -Z, positive turns left (CCW seen from above).
    pub yaw: f32,
    /// Radians. Positive looks up. Clamped by callers to about +-89 deg.
    pub pitch: f32,
    pub fov_y: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 70f32.to_radians(),
            znear: 1.0,
            zfar: 16384.0,
        }
    }
}

impl Camera {
    /// Return the projection aspect for a physical viewport.
    ///
    /// A zero-sized viewport is possible while a window is minimized, but it
    /// is not a valid projection or render-target size.
    pub fn aspect_for_viewport(viewport: (u32, u32)) -> Option<f32> {
        (viewport.0 != 0 && viewport.1 != 0).then(|| viewport.0 as f32 / viewport.1 as f32)
    }

    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(-sy * cp, sp, -cy * cp)
    }

    /// Horizontal forward (ignores pitch), normalized.
    pub fn forward_flat(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(-sy, 0.0, -cy)
    }

    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, -sy)
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_to_mat4(self.pos, self.forward(), Vec3::Y)
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        // DirectX-style 0..1 clip depth, matching wgpu.
        glam::camera::rh::proj::directx::perspective(self.fov_y, aspect, self.znear, self.zfar)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect) * self.view()
    }

    /// Point the camera at a world position.
    pub fn look_at(&mut self, target: Vec3) {
        let d = target - self.pos;
        let flat = (d.x * d.x + d.z * d.z).sqrt();
        self.yaw = (-d.x).atan2(-d.z);
        self.pitch = d.y.atan2(flat);
    }

    /// World-space ray through a window pixel (origin, normalized direction).
    /// `cursor` is in pixels from the top-left, `viewport` the target size in
    /// the same units — cursor picking for widgets and editors.
    pub fn screen_ray(&self, cursor: glam::Vec2, viewport: (f32, f32)) -> (Vec3, Vec3) {
        let aspect = viewport.0 / viewport.1.max(1.0);
        let inv = self.view_proj(aspect).inverse();
        let ndc = glam::Vec2::new(
            cursor.x / viewport.0 * 2.0 - 1.0,
            1.0 - cursor.y / viewport.1 * 2.0,
        );
        // wgpu clip depth is 0..1; unproject the near and far plane points.
        let near = inv.project_point3(Vec3::new(ndc.x, ndc.y, 0.0));
        let far = inv.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
        (near, (far - near).normalize_or_zero())
    }
}

#[cfg(test)]
mod tests {
    use super::Camera;

    #[test]
    fn viewport_aspect_ignores_zero_dimensions() {
        assert_eq!(Camera::aspect_for_viewport((0, 600)), None);
        assert_eq!(Camera::aspect_for_viewport((450, 0)), None);
        assert_eq!(Camera::aspect_for_viewport((0, 0)), None);
    }

    #[test]
    fn viewport_aspect_changes_projection_without_changing_camera_framing() {
        let camera = Camera::default();
        let narrow = Camera::aspect_for_viewport((450, 600)).unwrap();
        let wide = Camera::aspect_for_viewport((900, 600)).unwrap();

        assert_eq!(narrow, 0.75);
        assert_eq!(wide, 1.5);
        assert_ne!(
            camera.proj(narrow).to_cols_array(),
            camera.proj(wide).to_cols_array()
        );
    }
}
