//! Renderer-independent VRM 1.0 LookAt resolution and runtime math.
//!
//! VRM 1.0 uses a +Z-forward normalized humanoid space. The authored glTF
//! skeleton may use arbitrary rest rotations, so bone output is converted
//! from that normalized space back into each eye's authored local basis.

use glam::{EulerRot, Mat4, Quat, Vec3};
use pocket3d::anim::{NodeTrs, Skeleton};

use crate::{Vrm1Doc, Vrm1HumanBone, Vrm1LookAt, Vrm1LookAtRangeMap, Vrm1LookAtType};

const MIN_QUAT_LENGTH_SQUARED: f32 = 1.0e-12;
const MIN_DIRECTION_LENGTH_SQUARED: f32 = 1.0e-12;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vrm1LookAtAngles {
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vrm1ExpressionLookAt {
    pub look_up: f32,
    pub look_down: f32,
    pub look_left: f32,
    pub look_right: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vrm1EyeBoneOutput {
    pub node: usize,
    pub local_rotation: Quat,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Vrm1LookAtOutput {
    Bone {
        left_eye: Option<Vrm1EyeBoneOutput>,
        right_eye: Option<Vrm1EyeBoneOutput>,
    },
    Expression(Vrm1ExpressionLookAt),
}

impl Vrm1LookAtOutput {
    /// Apply only bone-mode output. Expression output remains owned by the
    /// application's expression compositor.
    pub fn apply_bone_rotations(&self, locals: &mut [NodeTrs]) {
        let Self::Bone {
            left_eye,
            right_eye,
        } = self
        else {
            return;
        };
        for eye in [left_eye, right_eye].into_iter().flatten() {
            if let Some(local) = locals.get_mut(eye.node) {
                local.rotation = eye.local_rotation;
            }
        }
    }

    pub fn expression_weights(&self) -> Vrm1ExpressionLookAt {
        match self {
            Self::Expression(weights) => *weights,
            Self::Bone { .. } => Vrm1ExpressionLookAt::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EyeRestBasis {
    node: usize,
    pre_multiply: Quat,
    post_multiply: Quat,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1LookAtRuntime {
    kind: Vrm1LookAtType,
    head: usize,
    head_chain: Vec<usize>,
    head_rest_global_rotation: Quat,
    offset_from_head_bone: Vec3,
    range_map_horizontal_inner: Vrm1LookAtRangeMap,
    range_map_horizontal_outer: Vrm1LookAtRangeMap,
    range_map_vertical_down: Vrm1LookAtRangeMap,
    range_map_vertical_up: Vrm1LookAtRangeMap,
    left_eye: Option<EyeRestBasis>,
    right_eye: Option<EyeRestBasis>,
}

impl Vrm1LookAtRuntime {
    /// Resolve and cache the authored rest bases needed by VRM 1.0 LookAt.
    /// Invalid optional eye mappings disable only those eyes. Bone mode is
    /// disabled when neither eye can be resolved; it never falls back to
    /// expression mode.
    pub fn new(document: &Vrm1Doc, skeleton: &Skeleton) -> Option<Self> {
        let look_at = document.look_at.as_ref()?;
        let head = document.humanoid.node_for(Vrm1HumanBone::Head)?;
        let (head_chain, head_rest_global_rotation) =
            rest_chain_and_global_rotation(head, skeleton)?;
        let left_eye = resolve_eye(document.humanoid.node_for(Vrm1HumanBone::LeftEye), skeleton);
        let right_eye = resolve_eye(
            document.humanoid.node_for(Vrm1HumanBone::RightEye),
            skeleton,
        );
        if look_at.kind == Vrm1LookAtType::Bone && left_eye.is_none() && right_eye.is_none() {
            log::warn!("VRM 1.0 bone LookAt has no valid eye bones; LookAt is disabled");
            return None;
        }
        Some(Self::from_parts(
            look_at,
            head,
            head_chain,
            head_rest_global_rotation,
            left_eye,
            right_eye,
        ))
    }

    fn from_parts(
        look_at: &Vrm1LookAt,
        head: usize,
        head_chain: Vec<usize>,
        head_rest_global_rotation: Quat,
        left_eye: Option<EyeRestBasis>,
        right_eye: Option<EyeRestBasis>,
    ) -> Self {
        Self {
            kind: look_at.kind,
            head,
            head_chain,
            head_rest_global_rotation,
            offset_from_head_bone: look_at.offset_from_head_bone,
            range_map_horizontal_inner: look_at.range_map_horizontal_inner,
            range_map_horizontal_outer: look_at.range_map_horizontal_outer,
            range_map_vertical_down: look_at.range_map_vertical_down,
            range_map_vertical_up: look_at.range_map_vertical_up,
            left_eye,
            right_eye,
        }
    }

    /// Evaluate a model-space target against the current animated head pose.
    /// Invalid or coincident targets deterministically produce neutral output.
    pub fn evaluate(
        &self,
        target_model: Vec3,
        locals: &[NodeTrs],
        globals: &[Mat4],
    ) -> Vrm1LookAtOutput {
        let angles = self.calculate_angles(target_model, locals, globals);
        match self.kind {
            Vrm1LookAtType::Bone => self.bone_output(angles),
            Vrm1LookAtType::Expression => Vrm1LookAtOutput::Expression(expression_output(
                angles,
                &self.range_map_horizontal_outer,
                &self.range_map_vertical_down,
                &self.range_map_vertical_up,
            )),
        }
    }

    pub fn calculate_angles(
        &self,
        target_model: Vec3,
        locals: &[NodeTrs],
        globals: &[Mat4],
    ) -> Vrm1LookAtAngles {
        let Some(head_global) = globals.get(self.head) else {
            return Vrm1LookAtAngles::default();
        };
        let origin = head_global.transform_point3(self.offset_from_head_bone);
        let Some(head_rotation) = current_global_rotation(&self.head_chain, locals) else {
            return Vrm1LookAtAngles::default();
        };
        if !target_model.is_finite() || !origin.is_finite() {
            return Vrm1LookAtAngles::default();
        }
        let look_at_rotation =
            (head_rotation * self.head_rest_global_rotation.inverse()).normalize();
        let local_target = look_at_rotation.inverse() * (target_model - origin);
        angles_from_local_target(local_target)
    }

    fn bone_output(&self, angles: Vrm1LookAtAngles) -> Vrm1LookAtOutput {
        let pitch_map = if angles.pitch_degrees >= 0.0 {
            &self.range_map_vertical_down
        } else {
            &self.range_map_vertical_up
        };
        let pitch = signed_mapped_degrees(angles.pitch_degrees, pitch_map).to_radians();
        let left_yaw_map = if angles.yaw_degrees >= 0.0 {
            &self.range_map_horizontal_outer
        } else {
            &self.range_map_horizontal_inner
        };
        let right_yaw_map = if angles.yaw_degrees >= 0.0 {
            &self.range_map_horizontal_inner
        } else {
            &self.range_map_horizontal_outer
        };
        let output_for = |eye: Option<EyeRestBasis>, yaw_map: &Vrm1LookAtRangeMap| {
            eye.map(|eye| {
                let yaw = signed_mapped_degrees(angles.yaw_degrees, yaw_map).to_radians();
                let normalized = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
                Vrm1EyeBoneOutput {
                    node: eye.node,
                    local_rotation: (eye.pre_multiply * normalized * eye.post_multiply).normalize(),
                }
            })
        };
        Vrm1LookAtOutput::Bone {
            left_eye: output_for(self.left_eye, left_yaw_map),
            right_eye: output_for(self.right_eye, right_yaw_map),
        }
    }
}

/// Map an absolute input angle through one VRM 1.0 range map. The specified
/// zero-input-maximum behavior is exact: neutral stays zero and any nonzero
/// finite input reaches `output_scale`.
pub fn map_vrm1_look_at_range(input_degrees: f32, range: &Vrm1LookAtRangeMap) -> f32 {
    if !input_degrees.is_finite()
        || !range.input_max_value.is_finite()
        || !range.output_scale.is_finite()
        || range.input_max_value < 0.0
        || range.output_scale < 0.0
    {
        return 0.0;
    }
    let input = input_degrees.abs();
    if input == 0.0 {
        return 0.0;
    }
    if range.input_max_value == 0.0 {
        return range.output_scale;
    }
    (input.min(range.input_max_value) / range.input_max_value) * range.output_scale
}

fn signed_mapped_degrees(input_degrees: f32, range: &Vrm1LookAtRangeMap) -> f32 {
    map_vrm1_look_at_range(input_degrees, range).copysign(input_degrees)
}

fn expression_output(
    angles: Vrm1LookAtAngles,
    horizontal_outer: &Vrm1LookAtRangeMap,
    vertical_down: &Vrm1LookAtRangeMap,
    vertical_up: &Vrm1LookAtRangeMap,
) -> Vrm1ExpressionLookAt {
    let mut output = Vrm1ExpressionLookAt::default();
    if angles.yaw_degrees > 0.0 {
        output.look_left = map_vrm1_look_at_range(angles.yaw_degrees, horizontal_outer);
    } else if angles.yaw_degrees < 0.0 {
        output.look_right = map_vrm1_look_at_range(angles.yaw_degrees, horizontal_outer);
    }
    if angles.pitch_degrees > 0.0 {
        output.look_down = map_vrm1_look_at_range(angles.pitch_degrees, vertical_down);
    } else if angles.pitch_degrees < 0.0 {
        output.look_up = map_vrm1_look_at_range(angles.pitch_degrees, vertical_up);
    }
    output
}

fn angles_from_local_target(local_target: Vec3) -> Vrm1LookAtAngles {
    if !local_target.is_finite() || local_target.length_squared() <= MIN_DIRECTION_LENGTH_SQUARED {
        return Vrm1LookAtAngles::default();
    }
    Vrm1LookAtAngles {
        yaw_degrees: local_target.x.atan2(local_target.z).to_degrees(),
        pitch_degrees: (-local_target.y)
            .atan2(Vec3::new(local_target.x, 0.0, local_target.z).length())
            .to_degrees(),
    }
}

fn current_global_rotation(chain: &[usize], locals: &[NodeTrs]) -> Option<Quat> {
    let mut rotation = Quat::IDENTITY;
    for &node in chain {
        let local = locals.get(node)?.rotation;
        if !valid_quat(local) {
            return None;
        }
        rotation = (rotation * local.normalize()).normalize();
    }
    Some(rotation)
}

fn resolve_eye(node: Option<usize>, skeleton: &Skeleton) -> Option<EyeRestBasis> {
    let node = node?;
    let (_, rest_global) = rest_chain_and_global_rotation(node, skeleton)?;
    let local = skeleton.rest.get(node)?.rotation;
    if !valid_quat(local) {
        return None;
    }
    let local = local.normalize();
    Some(EyeRestBasis {
        node,
        pre_multiply: (local * rest_global.inverse()).normalize(),
        post_multiply: rest_global,
    })
}

/// Returns a root-to-node chain and the node's authored global rest rotation.
fn rest_chain_and_global_rotation(node: usize, skeleton: &Skeleton) -> Option<(Vec<usize>, Quat)> {
    if skeleton.parents.len() != skeleton.rest.len() || node >= skeleton.rest.len() {
        return None;
    }
    let mut reverse_chain = Vec::new();
    let mut current = node;
    loop {
        if current >= skeleton.rest.len() || reverse_chain.len() >= skeleton.rest.len() {
            return None;
        }
        reverse_chain.push(current);
        let parent = skeleton.parents[current];
        if parent == usize::MAX {
            break;
        }
        current = parent;
    }
    reverse_chain.reverse();
    let mut rotation = Quat::IDENTITY;
    for &index in &reverse_chain {
        let local = skeleton.rest[index].rotation;
        if !valid_quat(local) {
            return None;
        }
        rotation = (rotation * local.normalize()).normalize();
    }
    Some((reverse_chain, rotation))
}

fn valid_quat(rotation: Quat) -> bool {
    rotation.is_finite() && rotation.length_squared() > MIN_QUAT_LENGTH_SQUARED
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn range(input: f32, output: f32) -> Vrm1LookAtRangeMap {
        Vrm1LookAtRangeMap {
            input_max_value: input,
            output_scale: output,
        }
    }

    fn look_at(kind: Vrm1LookAtType) -> Vrm1LookAt {
        Vrm1LookAt {
            kind,
            offset_from_head_bone: Vec3::ZERO,
            range_map_horizontal_inner: range(90.0, 10.0),
            range_map_horizontal_outer: range(90.0, 20.0),
            range_map_vertical_down: range(60.0, 12.0),
            range_map_vertical_up: range(30.0, 6.0),
        }
    }

    fn document(kind: Vrm1LookAtType, left: Option<usize>, right: Option<usize>) -> Vrm1Doc {
        let mut human_bones = BTreeMap::new();
        human_bones.insert(Vrm1HumanBone::Head, 1);
        if let Some(left) = left {
            human_bones.insert(Vrm1HumanBone::LeftEye, left);
        }
        if let Some(right) = right {
            human_bones.insert(Vrm1HumanBone::RightEye, right);
        }
        Vrm1Doc {
            meta: crate::Vrm1Meta {
                name: "test".into(),
                version: None,
                authors: vec!["test".into()],
                license_url: "test".into(),
            },
            humanoid: crate::Vrm1Humanoid { human_bones },
            materials_mtoon: crate::Vrm1ExtensionInfo::default(),
            spring_bone: crate::Vrm1ExtensionInfo::default(),
            spring_bone_semantics: None,
            node_constraint: crate::Vrm1ExtensionInfo::default(),
            node_constraint_semantics: None,
            expressions: Vec::new(),
            has_expressions: false,
            look_at: Some(look_at(kind)),
            has_first_person: false,
            node_count: 4,
            node_meshes: vec![None; 4],
            mesh_primitive_counts: Vec::new(),
        }
    }

    fn skeleton(rest: Vec<NodeTrs>, parents: Vec<usize>) -> Skeleton {
        let order = (0..rest.len()).collect();
        Skeleton {
            parents,
            rest,
            order,
        }
    }

    fn globals(skeleton: &Skeleton, locals: &[NodeTrs]) -> Vec<Mat4> {
        let mut output = Vec::new();
        skeleton.globals_from_locals(locals, &mut output);
        output
    }

    fn assert_quat(actual: Quat, expected: Quat) {
        assert!(
            actual.angle_between(expected).abs() < 1.0e-5,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn range_mapping_handles_neutral_half_clamp_zero_and_invalid_values() {
        let normal = range(90.0, 10.0);
        assert_eq!(map_vrm1_look_at_range(0.0, &normal), 0.0);
        assert!((map_vrm1_look_at_range(45.0, &normal) - 5.0).abs() < 1.0e-6);
        assert_eq!(map_vrm1_look_at_range(900.0, &normal), 10.0);
        assert_eq!(map_vrm1_look_at_range(-45.0, &normal), 5.0);
        let zero = range(0.0, 0.75);
        assert_eq!(map_vrm1_look_at_range(0.0, &zero), 0.0);
        assert_eq!(map_vrm1_look_at_range(0.01, &zero), 0.75);
        assert_eq!(map_vrm1_look_at_range(f32::NAN, &normal), 0.0);
        assert_eq!(map_vrm1_look_at_range(1.0, &range(-1.0, 2.0)), 0.0);
    }

    #[test]
    fn yaw_pitch_signs_follow_vrm1_plus_z_forward_space() {
        assert_eq!(
            angles_from_local_target(Vec3::Z),
            Vrm1LookAtAngles::default()
        );
        assert!(angles_from_local_target(Vec3::new(1.0, 0.0, 1.0)).yaw_degrees > 0.0);
        assert!(angles_from_local_target(Vec3::new(-1.0, 0.0, 1.0)).yaw_degrees < 0.0);
        assert!(angles_from_local_target(Vec3::new(0.0, -1.0, 1.0)).pitch_degrees > 0.0);
        assert!(angles_from_local_target(Vec3::new(0.0, 1.0, 1.0)).pitch_degrees < 0.0);
        assert_eq!(
            angles_from_local_target(Vec3::ZERO),
            Vrm1LookAtAngles::default()
        );
    }

    #[test]
    fn expression_output_uses_outer_horizontal_and_distinct_vertical_maps() {
        let left_down = expression_output(
            Vrm1LookAtAngles {
                yaw_degrees: 45.0,
                pitch_degrees: 30.0,
            },
            &range(90.0, 1.0),
            &range(60.0, 0.8),
            &range(30.0, 0.4),
        );
        assert_eq!(left_down.look_right, 0.0);
        assert_eq!(left_down.look_up, 0.0);
        assert!((left_down.look_left - 0.5).abs() < 1.0e-6);
        assert!((left_down.look_down - 0.4).abs() < 1.0e-6);
        let right_up = expression_output(
            Vrm1LookAtAngles {
                yaw_degrees: -90.0,
                pitch_degrees: -30.0,
            },
            &range(90.0, 1.0),
            &range(60.0, 0.8),
            &range(30.0, 0.4),
        );
        assert_eq!(right_up.look_left, 0.0);
        assert_eq!(right_up.look_down, 0.0);
        assert_eq!(right_up.look_right, 1.0);
        assert_eq!(right_up.look_up, 0.4);
    }

    #[test]
    fn bone_output_preserves_arbitrary_local_and_parent_rest_bases() {
        let parent_rest = Quat::from_rotation_z(0.31);
        let eye_rest = Quat::from_rotation_x(-0.27);
        let rest = vec![
            NodeTrs {
                rotation: parent_rest,
                ..NodeTrs::IDENTITY
            },
            NodeTrs::IDENTITY,
            NodeTrs {
                rotation: eye_rest,
                ..NodeTrs::IDENTITY
            },
        ];
        let skeleton = skeleton(rest.clone(), vec![usize::MAX, 0, 0]);
        let runtime =
            Vrm1LookAtRuntime::new(&document(Vrm1LookAtType::Bone, Some(2), None), &skeleton)
                .unwrap();
        let neutral = runtime.bone_output(Vrm1LookAtAngles::default());
        let Vrm1LookAtOutput::Bone {
            left_eye: Some(neutral),
            ..
        } = neutral
        else {
            panic!()
        };
        assert_quat(neutral.local_rotation, eye_rest);
        let output = runtime.bone_output(Vrm1LookAtAngles {
            yaw_degrees: 90.0,
            pitch_degrees: 0.0,
        });
        let Vrm1LookAtOutput::Bone {
            left_eye: Some(output),
            ..
        } = output
        else {
            panic!()
        };
        let normalized = Quat::from_rotation_y(20.0_f32.to_radians());
        let eye_global_rest = parent_rest * eye_rest;
        let expected = eye_rest * eye_global_rest.inverse() * normalized * eye_global_rest;
        assert_quat(output.local_rotation, expected.normalize());
    }

    #[test]
    fn bone_horizontal_mapping_distinguishes_inner_and_outer_per_eye() {
        let skeleton = skeleton(vec![NodeTrs::IDENTITY; 4], vec![usize::MAX, 0, 1, 1]);
        let runtime =
            Vrm1LookAtRuntime::new(&document(Vrm1LookAtType::Bone, Some(2), Some(3)), &skeleton)
                .unwrap();
        let output = runtime.bone_output(Vrm1LookAtAngles {
            yaw_degrees: 90.0,
            pitch_degrees: 0.0,
        });
        let Vrm1LookAtOutput::Bone {
            left_eye: Some(left),
            right_eye: Some(right),
        } = output
        else {
            panic!()
        };
        assert!((left.local_rotation.to_axis_angle().1.to_degrees() - 20.0).abs() < 1.0e-4);
        assert!((right.local_rotation.to_axis_angle().1.to_degrees() - 10.0).abs() < 1.0e-4);
    }

    #[test]
    fn non_identity_head_rest_rotation_still_faces_model_plus_z_at_rest() {
        let head_rest = Quat::from_rotation_x(0.37) * Quat::from_rotation_z(-0.21);
        let rest = vec![
            NodeTrs::IDENTITY,
            NodeTrs {
                rotation: head_rest,
                ..NodeTrs::IDENTITY
            },
            NodeTrs::IDENTITY,
        ];
        let skeleton = skeleton(rest.clone(), vec![usize::MAX, 0, 1]);
        let runtime =
            Vrm1LookAtRuntime::new(&document(Vrm1LookAtType::Bone, Some(2), None), &skeleton)
                .unwrap();
        let globals = globals(&skeleton, &rest);
        let origin = globals[1].w_axis.truncate();
        let angles = runtime.calculate_angles(origin + Vec3::Z, &rest, &globals);
        assert!(angles.yaw_degrees.abs() < 1.0e-5, "{angles:?}");
        assert!(angles.pitch_degrees.abs() < 1.0e-5, "{angles:?}");
    }

    #[test]
    fn animated_head_orientation_defines_current_look_at_space() {
        let rest = vec![NodeTrs::IDENTITY; 3];
        let skeleton = skeleton(rest.clone(), vec![usize::MAX, 0, 1]);
        let runtime =
            Vrm1LookAtRuntime::new(&document(Vrm1LookAtType::Bone, Some(2), None), &skeleton)
                .unwrap();
        let mut locals = rest;
        locals[1].rotation = Quat::from_rotation_y(core::f32::consts::FRAC_PI_2);
        let globals = globals(&skeleton, &locals);
        let origin = globals[1].w_axis.truncate();
        let target = origin + locals[1].rotation * Vec3::Z;
        let angles = runtime.calculate_angles(target, &locals, &globals);
        assert!(angles.yaw_degrees.abs() < 1.0e-4, "{angles:?}");
        assert!(angles.pitch_degrees.abs() < 1.0e-4, "{angles:?}");
    }

    #[test]
    fn one_missing_eye_works_both_missing_eyes_disable_bone_mode() {
        let skeleton = skeleton(vec![NodeTrs::IDENTITY; 3], vec![usize::MAX, 0, 1]);
        let one = Vrm1LookAtRuntime::new(&document(Vrm1LookAtType::Bone, Some(2), None), &skeleton)
            .unwrap();
        let output = one.bone_output(Vrm1LookAtAngles::default());
        let Vrm1LookAtOutput::Bone {
            left_eye,
            right_eye,
        } = output
        else {
            panic!()
        };
        assert!(left_eye.is_some());
        assert!(right_eye.is_none());
        assert!(
            Vrm1LookAtRuntime::new(&document(Vrm1LookAtType::Bone, None, None), &skeleton,)
                .is_none()
        );
    }

    #[test]
    fn presentation_y_pi_target_is_converted_before_vrm1_evaluation() {
        let skeleton = skeleton(vec![NodeTrs::IDENTITY; 3], vec![usize::MAX, 0, 1]);
        let runtime =
            Vrm1LookAtRuntime::new(&document(Vrm1LookAtType::Bone, Some(2), None), &skeleton)
                .unwrap();
        let locals = skeleton.rest.clone();
        let globals = globals(&skeleton, &locals);
        let presentation = Mat4::from_rotation_y(core::f32::consts::PI);
        let target_world = presentation.transform_point3(Vec3::new(1.0, 0.0, 1.0));
        let target_model = presentation.inverse().transform_point3(target_world);
        assert!(
            runtime
                .calculate_angles(target_model, &locals, &globals)
                .yaw_degrees
                > 0.0
        );
    }

    #[test]
    fn applying_bone_output_twice_does_not_accumulate() {
        let skeleton = skeleton(vec![NodeTrs::IDENTITY; 3], vec![usize::MAX, 0, 1]);
        let runtime =
            Vrm1LookAtRuntime::new(&document(Vrm1LookAtType::Bone, Some(2), None), &skeleton)
                .unwrap();
        let output = runtime.bone_output(Vrm1LookAtAngles {
            yaw_degrees: 45.0,
            pitch_degrees: 15.0,
        });
        let mut locals = skeleton.rest.clone();
        output.apply_bone_rotations(&mut locals);
        let once = locals[2].rotation;
        output.apply_bone_rotations(&mut locals);
        assert_quat(locals[2].rotation, once);
    }
}
