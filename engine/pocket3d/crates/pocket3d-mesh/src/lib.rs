//! Renderer-independent skin bindings shared by portable and desktop paths.
#![no_std]
extern crate alloc;

use alloc::vec::Vec;
use glam::Mat4;

/// Joint-to-node mapping and inverse bind transforms, in vertex influence order.
pub struct Skin {
    /// Node index per joint.
    pub joints: Vec<usize>,
    pub inverse_bind: Vec<Mat4>,
}

impl Skin {
    /// Evaluate joint matrices without allocating. `None` keeps the desktop
    /// object-space palette; `Some(model)` maps globals into an output space.
    /// Bindings must have equal lengths and every joint must index `globals`.
    pub fn matrices<'a>(
        &'a self,
        globals: &'a [Mat4],
        model: Option<Mat4>,
    ) -> impl ExactSizeIterator<Item = Mat4> + 'a {
        assert_eq!(self.joints.len(), self.inverse_bind.len());
        self.joints
            .iter()
            .zip(&self.inverse_bind)
            .map(move |(&node, bind)| {
                let global = globals[node];
                model.map_or(global, |model| model * global) * *bind
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use glam::{Quat, Vec3};
    use pocket3d_anim::{NodeTrs, Skeleton};

    #[test]
    fn matrices_keep_joint_and_skin_order_with_model_outside_bind() {
        let skeleton = Skeleton {
            parents: vec![usize::MAX, 0, 1],
            order: vec![0, 1, 2],
            rest: vec![
                NodeTrs {
                    translation: Vec3::new(0.5, 0.0, 0.0),
                    rotation: Quat::from_rotation_x(0.2),
                    ..NodeTrs::IDENTITY
                },
                NodeTrs {
                    translation: Vec3::Y,
                    ..NodeTrs::IDENTITY
                },
                NodeTrs {
                    translation: Vec3::Z,
                    ..NodeTrs::IDENTITY
                },
            ],
        };
        let mut globals = vec![];
        skeleton.globals_from_locals(&skeleton.rest, &mut globals);
        let skins = [
            Skin {
                joints: vec![2, 0],
                inverse_bind: vec![
                    Mat4::from_rotation_translation(Quat::from_rotation_y(0.3), Vec3::X),
                    Mat4::from_translation(Vec3::Y),
                ],
            },
            Skin {
                joints: vec![1],
                inverse_bind: vec![Mat4::from_translation(Vec3::Z)],
            },
        ];
        let palette: Vec<_> = skins
            .iter()
            .flat_map(|skin| skin.matrices(&globals, None))
            .collect();
        assert_eq!(palette.len(), 3);
        assert_eq!(palette[0], globals[2] * skins[0].inverse_bind[0]);
        assert_eq!(palette[1], globals[0] * skins[0].inverse_bind[1]);
        assert_eq!(palette[2], globals[1] * skins[1].inverse_bind[0]);

        let model =
            Mat4::from_rotation_translation(Quat::from_rotation_z(0.4), Vec3::new(3.0, -1.0, 2.0));
        for (joint, matrix) in skins[0].matrices(&globals, Some(model)).enumerate() {
            assert_eq!(
                matrix,
                (model * globals[skins[0].joints[joint]]) * skins[0].inverse_bind[joint]
            );
        }
    }

    #[test]
    #[should_panic]
    fn mismatched_bindings_are_rejected() {
        Skin {
            joints: vec![0],
            inverse_bind: vec![],
        }
        .matrices(&[Mat4::IDENTITY], None)
        .count();
    }
}
