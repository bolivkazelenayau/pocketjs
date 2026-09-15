//! Skeletal animation: clips of TRS channels sampled onto a node hierarchy.

#![no_std]
extern crate alloc;

use alloc::{string::String, vec::Vec};
use core::fmt;
use glam::{Mat4, Quat, Vec3};

pub use glam;

#[derive(Clone, Copy, Debug)]
pub struct NodeTrs {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl NodeTrs {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Blend local poses with linear translation/scale and spherical rotation.
    /// The caller supplies a finite fraction in [0, 1].
    pub fn interpolate(self, other: Self, fraction: f32) -> Self {
        if fraction == 0.0 {
            return self;
        }
        if fraction == 1.0 {
            return other;
        }
        Self {
            translation: self.translation.lerp(other.translation, fraction),
            rotation: self.rotation.slerp(other.rotation, fraction),
            scale: self.scale.lerp(other.scale, fraction),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelPath {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Linear,
    Step,
}

/// Portable validation failures; desktop loaders add the channel and file context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChannelValidationError {
    NoKeyTimes,
    InvalidKeyTimes,
    UnsortedKeyTimes,
    ValueCountOverflow,
    ValueCountMismatch {
        values: usize,
        keys: usize,
        components: usize,
    },
    NonFiniteValues,
    MalformedRotationKey {
        key: usize,
    },
    InvalidRotationKey {
        key: usize,
    },
}

impl fmt::Display for ChannelValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NoKeyTimes => write!(f, "animation channel has no key times"),
            Self::InvalidKeyTimes => write!(f, "animation channel has invalid key times"),
            Self::UnsortedKeyTimes => write!(f, "animation channel key times are not sorted"),
            Self::ValueCountOverflow => write!(f, "animation channel value count overflows"),
            Self::ValueCountMismatch {
                values,
                keys,
                components,
            } => write!(
                f,
                "animation channel value count {values} does not match {keys} key(s) of {components} component(s)"
            ),
            Self::NonFiniteValues => write!(f, "animation channel has non-finite values"),
            Self::MalformedRotationKey { key } => {
                write!(
                    f,
                    "animation rotation key {key} has malformed quaternion values"
                )
            }
            Self::InvalidRotationKey { key } => write!(
                f,
                "animation rotation key {key} has a zero or non-finite quaternion"
            ),
        }
    }
}

impl core::error::Error for ChannelValidationError {}

type Result<T> = core::result::Result<T, ChannelValidationError>;

pub struct Channel {
    pub node: usize,
    pub path: ChannelPath,
    pub interpolation: Interpolation,
    pub times: Vec<f32>,
    /// 3 floats per key for T/S, 4 for R (xyzw).
    pub values: Vec<f32>,
}

impl Channel {
    /// Validate the invariants required by [`Self::sample`].
    ///
    /// glTF input is checked before channels are constructed, but keeping the
    /// check here makes the sampling contract explicit for other callers too.
    pub fn validate(&self) -> Result<()> {
        if self.times.is_empty() {
            return Err(ChannelValidationError::NoKeyTimes);
        }
        if self
            .times
            .iter()
            .any(|time| !time.is_finite() || *time < 0.0)
        {
            return Err(ChannelValidationError::InvalidKeyTimes);
        }
        if self.times.windows(2).any(|window| window[1] <= window[0]) {
            return Err(ChannelValidationError::UnsortedKeyTimes);
        }

        let values_per_key = match self.path {
            ChannelPath::Translation | ChannelPath::Scale => 3,
            ChannelPath::Rotation => 4,
        };
        let expected_values = self
            .times
            .len()
            .checked_mul(values_per_key)
            .ok_or(ChannelValidationError::ValueCountOverflow)?;
        if self.values.len() != expected_values {
            return Err(ChannelValidationError::ValueCountMismatch {
                values: self.values.len(),
                keys: self.times.len(),
                components: values_per_key,
            });
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(ChannelValidationError::NonFiniteValues);
        }
        Ok(())
    }

    /// Validate authored rotation values, including the middle value of each
    /// CUBICSPLINE triplet. Callers validate channel shape before sampling.
    pub fn validate_rotation_keys(&self, cubic_spline: bool) -> Result<()> {
        if self.path != ChannelPath::Rotation {
            return Ok(());
        }
        let stride = if cubic_spline { 3 } else { 1 };
        for key in 0..self.times.len() {
            let value_index = key
                .checked_mul(stride)
                .and_then(|index| index.checked_add(usize::from(cubic_spline)))
                .and_then(|index| index.checked_mul(4))
                .ok_or(ChannelValidationError::MalformedRotationKey { key })?;
            let value_end = value_index
                .checked_add(4)
                .ok_or(ChannelValidationError::MalformedRotationKey { key })?;
            let q = self
                .values
                .get(value_index..value_end)
                .ok_or(ChannelValidationError::MalformedRotationKey { key })?;
            let length_squared = q.iter().map(|value| value * value).sum::<f32>();
            if !length_squared.is_finite() || length_squared <= 1.0e-12 {
                return Err(ChannelValidationError::InvalidRotationKey { key });
            }
        }
        Ok(())
    }

    fn key_span(&self, t: f32) -> (usize, usize, f32) {
        let times = &self.times;
        if times.is_empty() {
            return (0, 0, 0.0);
        }
        if t <= times[0] {
            return (0, 0, 0.0);
        }
        let last = times.len() - 1;
        if t >= times[last] {
            return (last, last, 0.0);
        }
        let hi = times.partition_point(|&k| k <= t);
        let lo = hi - 1;
        let span = times[hi] - times[lo];
        let f = if span > 0.0 {
            (t - times[lo]) / span
        } else {
            0.0
        };
        (lo, hi, f)
    }

    fn vec3_at(&self, key: usize) -> Vec3 {
        let i = key * 3;
        Vec3::new(self.values[i], self.values[i + 1], self.values[i + 2])
    }

    fn quat_at(&self, key: usize) -> Quat {
        let i = key * 4;
        let q = Quat::from_xyzw(
            self.values[i],
            self.values[i + 1],
            self.values[i + 2],
            self.values[i + 3],
        );
        if !q.length_squared().is_finite() || q.length_squared() <= 1.0e-12 {
            Quat::IDENTITY
        } else {
            q.normalize()
        }
    }

    pub fn sample(&self, t: f32, out: &mut NodeTrs) {
        let values_per_key = match self.path {
            ChannelPath::Translation | ChannelPath::Scale => 3,
            ChannelPath::Rotation => 4,
        };
        let Some(&first_time) = self.times.first() else {
            return;
        };
        let Some(required_values) = self.times.len().checked_mul(values_per_key) else {
            return;
        };
        if !t.is_finite() || !first_time.is_finite() || self.values.len() < required_values {
            return;
        }
        let (lo, hi, f) = self.key_span(t);
        let f = if self.interpolation == Interpolation::Step {
            0.0
        } else {
            f
        };
        match self.path {
            ChannelPath::Translation => {
                out.translation = self.vec3_at(lo).lerp(self.vec3_at(hi), f);
            }
            ChannelPath::Scale => {
                out.scale = self.vec3_at(lo).lerp(self.vec3_at(hi), f);
            }
            ChannelPath::Rotation => {
                out.rotation = self.quat_at(lo).slerp(self.quat_at(hi), f);
            }
        }
    }
}

pub struct Clip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}

/// A node hierarchy with rest-pose TRS, in evaluation order (parents first).
pub struct Skeleton {
    /// Parent index per node (usize::MAX = root).
    pub parents: Vec<usize>,
    pub rest: Vec<NodeTrs>,
    /// Indices into nodes, ordered so parents precede children.
    pub order: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::{Channel, ChannelPath, Interpolation, NodeTrs};
    use alloc::vec;

    #[test]
    fn sample_ignores_nonfinite_first_timestamp() {
        let channel = Channel {
            node: 0,
            path: ChannelPath::Translation,
            interpolation: Interpolation::Linear,
            times: vec![f32::NAN],
            values: vec![1.0, 2.0, 3.0],
        };
        let mut output = NodeTrs::IDENTITY;
        channel.sample(0.0, &mut output);
        assert_eq!(output.translation, NodeTrs::IDENTITY.translation);
    }
}

impl Skeleton {
    /// Sample `clip` at `t` (wrapping if `looping`) into per-node local TRS,
    /// starting from the rest pose. The split from [`Self::globals_from_locals`]
    /// lets callers inject procedural pose edits (look-at, physics bones)
    /// between animation sampling and the hierarchy walk.
    pub fn sample_locals(
        &self,
        clip: Option<&Clip>,
        t: f32,
        looping: bool,
        locals: &mut Vec<NodeTrs>,
    ) {
        locals.clear();
        locals.extend_from_slice(&self.rest);
        let n = locals.len();
        if let Some(clip) = clip {
            let t = if clip.duration > 0.0 {
                if looping {
                    // `f32::rem_euclid` is unavailable in no_std. This is its
                    // finite-time remainder rule without changing clip math.
                    let remainder = t % clip.duration;
                    if remainder < 0.0 {
                        remainder + clip.duration
                    } else {
                        remainder
                    }
                } else {
                    t.clamp(0.0, clip.duration)
                }
            } else {
                0.0
            };
            for ch in &clip.channels {
                if ch.node < n {
                    ch.sample(t, &mut locals[ch.node]);
                }
            }
        }
    }

    /// Multiply local TRS down the hierarchy into per-node global transforms.
    pub fn globals_from_locals(&self, locals: &[NodeTrs], globals: &mut Vec<Mat4>) {
        globals.clear();
        globals.resize(locals.len(), Mat4::IDENTITY);
        for &i in &self.order {
            let local = locals[i].matrix();
            globals[i] = if self.parents[i] == usize::MAX {
                local
            } else {
                globals[self.parents[i]] * local
            };
        }
    }

    /// Sample `clip` at `t` (wrapping if `looping`) and return global
    /// transforms per node.
    pub fn global_transforms(
        &self,
        clip: Option<&Clip>,
        t: f32,
        looping: bool,
        globals: &mut Vec<Mat4>,
    ) {
        let mut locals = Vec::new();
        self.sample_locals(clip, t, looping, &mut locals);
        self.globals_from_locals(&locals, globals);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AnimState {
    pub clip: usize,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
}

impl Default for AnimState {
    fn default() -> Self {
        Self {
            clip: 0,
            time: 0.0,
            speed: 1.0,
            looping: true,
        }
    }
}

impl AnimState {
    pub fn advance(&mut self, dt: f32) {
        self.time += dt * self.speed;
    }
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use alloc::vec;

    fn channel(path: ChannelPath, values: &[f32]) -> Channel {
        Channel {
            node: 0,
            path,
            interpolation: Interpolation::Linear,
            times: vec![0.0, 1.0, 2.0],
            values: values.to_vec(),
        }
    }

    #[test]
    fn validation_reports_shape_and_value_errors_without_desktop_dependencies() {
        let mut ch = channel(
            ChannelPath::Translation,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        );
        assert_eq!(ch.validate(), Ok(()));
        ch.times.clear();
        assert_eq!(ch.validate(), Err(ChannelValidationError::NoKeyTimes));
        ch.times = vec![0.0, f32::NAN, 2.0];
        assert_eq!(ch.validate(), Err(ChannelValidationError::InvalidKeyTimes));
        ch.times = vec![0.0, 0.0, 2.0];
        assert_eq!(ch.validate(), Err(ChannelValidationError::UnsortedKeyTimes));
        ch.times = vec![0.0, 1.0, 2.0];
        ch.values.pop();
        assert!(matches!(
            ch.validate(),
            Err(ChannelValidationError::ValueCountMismatch { values: 8, .. })
        ));
        ch.values.push(f32::INFINITY);
        assert_eq!(ch.validate(), Err(ChannelValidationError::NonFiniteValues));
    }

    #[test]
    fn rotation_validation_and_sampling_protect_zero_overflow_and_malformed_keys() {
        let mut ch = channel(
            ChannelPath::Rotation,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        ch.values[4..8].fill(0.0);
        assert_eq!(
            ch.validate_rotation_keys(false),
            Err(ChannelValidationError::InvalidRotationKey { key: 1 })
        );
        let mut local = NodeTrs::IDENTITY;
        ch.sample(1.0, &mut local);
        assert_eq!(local.rotation, Quat::IDENTITY);
        ch.values[4..8].fill(f32::MAX);
        assert_eq!(
            ch.validate_rotation_keys(false),
            Err(ChannelValidationError::InvalidRotationKey { key: 1 })
        );
        ch.sample(1.0, &mut local);
        assert_eq!(local.rotation, Quat::IDENTITY);
        ch.values.truncate(3);
        assert_eq!(
            ch.validate_rotation_keys(false),
            Err(ChannelValidationError::MalformedRotationKey { key: 0 })
        );
        local.rotation = Quat::from_rotation_x(0.3);
        let prior = local.rotation;
        ch.sample(0.5, &mut local);
        assert_eq!(local.rotation, prior);
    }

    #[test]
    fn nonfinite_sampling_inputs_and_bad_nodes_leave_rest_locals_unchanged() {
        let mut ch = channel(
            ChannelPath::Translation,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        );
        let mut local = NodeTrs {
            translation: Vec3::Y,
            ..NodeTrs::IDENTITY
        };
        ch.sample(f32::NAN, &mut local);
        assert_eq!(local.translation, Vec3::Y);
        ch.times[0] = f32::INFINITY;
        ch.sample(0.5, &mut local);
        assert_eq!(local.translation, Vec3::Y);
        ch.times[0] = 0.0;
        ch.node = 9;
        let skeleton = Skeleton {
            parents: vec![usize::MAX],
            order: vec![0],
            rest: vec![local],
        };
        let clip = Clip {
            name: "bad-node".into(),
            duration: 2.0,
            channels: vec![ch],
        };
        let mut locals = vec![];
        skeleton.sample_locals(Some(&clip), 0.5, false, &mut locals);
        assert_eq!(locals[0].translation, Vec3::Y);
    }

    #[test]
    fn linear_step_endpoints_and_positive_negative_wrapping_keep_existing_math() {
        let mut ch = channel(
            ChannelPath::Translation,
            &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        );
        let mut local = NodeTrs::IDENTITY;
        ch.sample(0.5, &mut local);
        assert_eq!(local.translation, Vec3::X);
        ch.sample(1.0, &mut local);
        assert_eq!(local.translation, Vec3::X * 2.0);
        ch.sample(2.0, &mut local);
        assert_eq!(local.translation, Vec3::X * 4.0);
        ch.interpolation = Interpolation::Step;
        ch.sample(0.99, &mut local);
        assert_eq!(local.translation, Vec3::ZERO);
        ch.sample(1.0, &mut local);
        assert_eq!(local.translation, Vec3::X * 2.0);
        ch.interpolation = Interpolation::Linear;
        let skeleton = Skeleton {
            parents: vec![usize::MAX],
            order: vec![0],
            rest: vec![NodeTrs::IDENTITY],
        };
        let clip = Clip {
            name: "loop".into(),
            duration: 2.0,
            channels: vec![ch],
        };
        let mut locals = vec![];
        skeleton.sample_locals(Some(&clip), 2.5, true, &mut locals);
        assert_eq!(locals[0].translation, Vec3::X);
        skeleton.sample_locals(Some(&clip), -0.5, true, &mut locals);
        assert_eq!(locals[0].translation, Vec3::X * 3.0);
        skeleton.sample_locals(Some(&clip), 2.5, false, &mut locals);
        assert_eq!(locals[0].translation, Vec3::X * 4.0);
    }

    #[test]
    fn globals_multiply_parent_before_local() {
        let skeleton = Skeleton {
            parents: vec![usize::MAX, 0, 1],
            order: vec![0, 1, 2],
            rest: vec![
                NodeTrs {
                    rotation: Quat::from_rotation_z(0.3),
                    scale: Vec3::new(1.2, 0.7, 1.0),
                    ..NodeTrs::IDENTITY
                },
                NodeTrs {
                    translation: Vec3::X,
                    rotation: Quat::from_rotation_y(-0.4),
                    ..NodeTrs::IDENTITY
                },
                NodeTrs {
                    translation: Vec3::Y,
                    ..NodeTrs::IDENTITY
                },
            ],
        };
        let mut globals = vec![];
        skeleton.globals_from_locals(&skeleton.rest, &mut globals);
        assert_eq!(globals[1], globals[0] * skeleton.rest[1].matrix());
        assert_eq!(globals[2], globals[1] * skeleton.rest[2].matrix());
    }

    #[test]
    fn pose_interpolation_keeps_endpoints_and_uses_short_orientation() {
        let a = NodeTrs::IDENTITY;
        let b = NodeTrs {
            translation: Vec3::new(2.0, 4.0, 6.0),
            rotation: Quat::from_rotation_z(core::f32::consts::FRAC_PI_2),
            scale: Vec3::splat(3.0),
        };
        assert_eq!(a.interpolate(b, 0.0).matrix(), a.matrix());
        assert_eq!(a.interpolate(b, 1.0).matrix(), b.matrix());
        let middle = a.interpolate(b, 0.5);
        assert_eq!(middle.translation, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(middle.scale, Vec3::splat(2.0));
        let direction = middle.rotation * Vec3::X;
        assert!((direction.x - direction.y).abs() < 1.0e-6 && direction.x > 0.7);
    }
}
