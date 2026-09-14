//! Pure VRM 1.0 node-constraint rotation math.
//!
//! This module intentionally has no skeleton, hierarchy, parent-transform, or
//! frame-scheduling dependencies. Runtime code can supply the current model
//! space inputs and apply the returned local rotation to a destination node.

use glam::{Quat, Vec3};

use crate::{Vrm1AimAxis, Vrm1RollAxis};

const MIN_QUAT_LENGTH: f32 = 1.0e-6;
const MIN_VECTOR_LENGTH: f32 = 1.0e-6;
const SHORTEST_ARC_CROSS_EPSILON_SQUARED: f32 = 1.0e-12;
const SLERP_LINEAR_DOT_EPSILON: f32 = 1.0e-6;

/// Apply the VRMC_node_constraint 1.0 Rotation operation.
///
/// The source animation is interpreted as a delta from the source's authored
/// local rest rotation, then that delta is applied to the destination's
/// authored local rest rotation:
///
/// ```text
/// delta_source = inverse(source_rest) * source_current
/// target        = destination_rest * delta_source
/// output        = slerp(destination_rest, target, weight)
/// ```
///
/// The destination's current/animated rotation is deliberately not an input:
/// the destination rest rotation is the interpolation baseline. Quaternion
/// inputs are normalized before use and the interpolation follows the
/// shortest path. Invalid or zero-length quaternion inputs are treated as
/// identity so this pure operation cannot introduce non-finite output.
pub fn rotation_constraint(
    source_rest: Quat,
    source_current: Quat,
    destination_rest: Quat,
    weight: f32,
) -> Quat {
    let source_rest = normalize_or_identity(source_rest);
    let source_current = normalize_or_identity(source_current);
    let destination_rest = normalize_or_identity(destination_rest);

    let source_delta = normalize_or_identity(source_rest.inverse() * source_current);
    let target = normalize_or_identity(destination_rest * source_delta);
    let weight = if weight.is_finite() {
        weight.clamp(0.0, 1.0)
    } else {
        0.0
    };

    // `Quat::slerp` in glam is shortest-path, but explicitly normalizing the
    // endpoints and result keeps that precondition and the output invariant
    // local to this operation.
    normalize_or_identity(destination_rest.slerp(target, weight))
}

/// Normalize a quaternion only when it is finite and non-degenerate.
///
/// The largest-component scaling avoids overflow while checking very large
/// finite inputs. Returning `None` keeps checked Roll helpers from allowing
/// invalid data to produce a non-finite result.
pub fn checked_normalize_quat(quaternion: Quat) -> Option<Quat> {
    if !quaternion.is_finite() {
        return None;
    }
    let largest_component = quaternion
        .x
        .abs()
        .max(quaternion.y.abs())
        .max(quaternion.z.abs())
        .max(quaternion.w.abs());
    if !largest_component.is_finite() || largest_component <= f32::MIN_POSITIVE {
        return None;
    }
    let scaled = quaternion / largest_component;
    let length = scaled.length();
    if !length.is_finite() || length <= MIN_QUAT_LENGTH {
        return None;
    }
    let normalized = scaled / length;
    normalized.is_finite().then_some(normalized)
}

/// Normalize a vector only when it is finite and non-degenerate.
pub fn checked_normalize_vec3(vector: Vec3) -> Option<Vec3> {
    if !vector.is_finite() {
        return None;
    }
    let largest_component = vector.x.abs().max(vector.y.abs()).max(vector.z.abs());
    if !largest_component.is_finite() || largest_component <= f32::MIN_POSITIVE {
        return None;
    }
    let scaled = vector / largest_component;
    let length = scaled.length();
    if !length.is_finite() || length <= MIN_VECTOR_LENGTH {
        return None;
    }
    let normalized = scaled / length;
    normalized.is_finite().then_some(normalized)
}

/// Convert a parsed VRM Roll axis into its local-space unit vector.
pub const fn roll_axis_vector(axis: Vrm1RollAxis) -> Vec3 {
    match axis {
        Vrm1RollAxis::X => Vec3::X,
        Vrm1RollAxis::Y => Vec3::Y,
        Vrm1RollAxis::Z => Vec3::Z,
    }
}

/// Return the shortest unit quaternion rotating `from` onto `to`.
///
/// Parallel vectors use identity. For antiparallel vectors the rotation axis
/// is underdetermined, so the least-aligned Cartesian basis is selected with
/// fixed tie-breaking. This makes the difficult 180-degree case finite and
/// deterministic.
pub fn shortest_rotation(from: Vec3, to: Vec3) -> Option<Quat> {
    let from = checked_normalize_vec3(from)?;
    let to = checked_normalize_vec3(to)?;
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if !dot.is_finite() {
        return None;
    }

    let cross = from.cross(to);
    let cross_length_squared = cross.length_squared();
    if !cross_length_squared.is_finite() {
        return None;
    }
    if cross_length_squared <= SHORTEST_ARC_CROSS_EPSILON_SQUARED && dot > 0.0 {
        return Some(Quat::IDENTITY);
    }

    if cross_length_squared > SHORTEST_ARC_CROSS_EPSILON_SQUARED {
        return checked_normalize_quat(Quat::from_xyzw(cross.x, cross.y, cross.z, 1.0 + dot));
    }

    let absolute = from.abs();
    let basis = if absolute.x <= absolute.y && absolute.x <= absolute.z {
        Vec3::X
    } else if absolute.y <= absolute.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let fallback_axis = checked_normalize_vec3(from.cross(basis))?;
    checked_normalize_quat(Quat::from_xyzw(
        fallback_axis.x,
        fallback_axis.y,
        fallback_axis.z,
        0.0,
    ))
}

/// Convert a parsed VRM Aim axis into its signed local-space unit vector.
pub const fn aim_axis_vector(axis: Vrm1AimAxis) -> Vec3 {
    match axis {
        Vrm1AimAxis::PositiveX => Vec3::X,
        Vrm1AimAxis::NegativeX => Vec3::NEG_X,
        Vrm1AimAxis::PositiveY => Vec3::Y,
        Vrm1AimAxis::NegativeY => Vec3::NEG_Y,
        Vrm1AimAxis::PositiveZ => Vec3::Z,
        Vrm1AimAxis::NegativeZ => Vec3::NEG_Z,
    }
}

/// Return the unblended VRM Aim target local rotation.
///
/// The positions are current model-space node origins. `destination_parent`
/// is the current model-global rotation of the destination's parent, or the
/// identity quaternion when the destination is a root. The parent rotation is
/// supplied as an accumulated quaternion rather than extracted from a global
/// matrix, so non-uniform or negative scale in the position hierarchy cannot
/// contaminate the rotation calculation.
///
/// A coincident source and destination has no defined direction, so this
/// returns `None`; callers should retain the destination rest rotation. Invalid
/// inputs also return `None` so the public operation remains finite.
pub fn aim_target(
    source_position: Vec3,
    destination_position: Vec3,
    destination_parent: Quat,
    destination_rest: Quat,
    aim_axis: Vec3,
) -> Option<Quat> {
    let destination_parent = checked_normalize_quat(destination_parent)?;
    let destination_rest = checked_normalize_quat(destination_rest)?;
    let aim_axis = checked_normalize_vec3(aim_axis)?;
    let from = checked_normalize_vec3(destination_parent * (destination_rest * aim_axis))?;
    let to = checked_normalize_vec3(source_position - destination_position)?;
    let shortest = shortest_rotation(from, to)?;

    // Qtarget = inverse(Qp) * R * Qp * Qd0
    normalized_product(
        normalized_product(
            normalized_product(destination_parent.inverse(), shortest)?,
            destination_parent,
        )?,
        destination_rest,
    )
}

/// Apply one VRM Aim constraint, blending from the authored destination rest
/// rotation.
///
/// Aim owns the destination rotation at its evaluation stage: the current
/// destination rotation is intentionally not an input. Translation and scale
/// remain the caller's responsibility. A coincident source/destination point
/// deterministically returns the normalized destination rest rotation.
pub fn aim_constraint(
    source_position: Vec3,
    destination_position: Vec3,
    destination_parent: Quat,
    destination_rest: Quat,
    aim_axis: Vec3,
    weight: f32,
) -> Quat {
    let destination_rest = normalize_or_identity(destination_rest);
    let Some(target) = aim_target(
        source_position,
        destination_position,
        destination_parent,
        destination_rest,
        aim_axis,
    ) else {
        return destination_rest;
    };
    let weight = if weight.is_finite() {
        weight.clamp(0.0, 1.0)
    } else {
        0.0
    };
    slerp_shortest(destination_rest, target, weight).unwrap_or(destination_rest)
}

/// Apply Aim using the typed VRM signed axis enum.
pub fn apply_aim_constraint(
    source_position: Vec3,
    destination_position: Vec3,
    destination_parent: Quat,
    destination_rest: Quat,
    axis: Vrm1AimAxis,
    weight: f32,
) -> Quat {
    aim_constraint(
        source_position,
        destination_position,
        destination_parent,
        destination_rest,
        aim_axis_vector(axis),
        weight,
    )
}

/// Checked shortest-path interpolation of two quaternion rotations.
pub fn slerp_shortest(from: Quat, to: Quat, weight: f32) -> Option<Quat> {
    if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
        return None;
    }
    let from = checked_normalize_quat(from)?;
    let mut to = checked_normalize_quat(to)?;
    let mut dot = from.dot(to);
    if !dot.is_finite() {
        return None;
    }
    if dot < 0.0 {
        to = -to;
        dot = -dot;
    }
    dot = dot.clamp(-1.0, 1.0);

    let result = if weight == 0.0 {
        from
    } else if weight == 1.0 {
        to
    } else if dot > 1.0 - SLERP_LINEAR_DOT_EPSILON {
        from.lerp(to, weight)
    } else {
        let angle = dot.acos();
        let sine = angle.sin();
        if !sine.is_finite() || sine.abs() <= f32::EPSILON {
            from.lerp(to, weight)
        } else {
            let from_scale = ((1.0 - weight) * angle).sin() / sine;
            let to_scale = (weight * angle).sin() / sine;
            from * from_scale + to * to_scale
        }
    };
    checked_normalize_quat(result)
}

fn normalized_product(left: Quat, right: Quat) -> Option<Quat> {
    checked_normalize_quat(left * right)
}

/// Extract the pure Roll twist from source and destination rest bases.
///
/// The source delta is conjugated into the common authored frame and then
/// into the destination authored frame. Swing is removed by the shortest
/// rotation that maps the configured destination roll axis to its moved
/// position. No axis-angle component extraction is involved.
pub fn roll_twist(
    source_rest: Quat,
    source_current: Quat,
    destination_rest: Quat,
    roll_axis: Vec3,
) -> Option<Quat> {
    let source_rest = checked_normalize_quat(source_rest)?;
    let source_current = checked_normalize_quat(source_current)?;
    let destination_rest = checked_normalize_quat(destination_rest)?;
    let roll_axis = checked_normalize_vec3(roll_axis)?;

    let source_delta = normalized_product(source_rest.conjugate(), source_current)?;
    let source_parent_delta = normalized_product(
        normalized_product(source_rest, source_delta)?,
        source_rest.conjugate(),
    )?;
    let destination_delta = normalized_product(
        normalized_product(destination_rest.conjugate(), source_parent_delta)?,
        destination_rest,
    )?;
    let swing = shortest_rotation(roll_axis, destination_delta * roll_axis)?;
    normalized_product(swing.conjugate(), destination_delta)
}

/// Return the unblended Roll target `Qtarget = Qd0 * T`.
pub fn roll_target(
    source_rest: Quat,
    source_current: Quat,
    destination_rest: Quat,
    roll_axis: Vec3,
) -> Option<Quat> {
    let destination_rest = checked_normalize_quat(destination_rest)?;
    let twist = roll_twist(source_rest, source_current, destination_rest, roll_axis)?;
    normalized_product(destination_rest, twist)
}

/// Apply one VRM Roll constraint, blending from the authored destination rest
/// rotation. The destination's current/animated rotation is intentionally
/// not an input to this pure operation.
pub fn roll_constraint(
    source_rest: Quat,
    source_current: Quat,
    destination_rest: Quat,
    roll_axis: Vec3,
    weight: f32,
) -> Quat {
    let destination_rest = normalize_or_identity(destination_rest);
    let Some(target) = roll_target(source_rest, source_current, destination_rest, roll_axis) else {
        return destination_rest;
    };
    let weight = if weight.is_finite() {
        weight.clamp(0.0, 1.0)
    } else {
        0.0
    };
    slerp_shortest(destination_rest, target, weight).unwrap_or(destination_rest)
}

/// Apply Roll using the typed VRM axis enum.
pub fn apply_roll_constraint(
    source_rest: Quat,
    source_current: Quat,
    destination_rest: Quat,
    axis: Vrm1RollAxis,
    weight: f32,
) -> Quat {
    roll_constraint(
        source_rest,
        source_current,
        destination_rest,
        roll_axis_vector(axis),
        weight,
    )
}

fn normalize_or_identity(quaternion: Quat) -> Quat {
    checked_normalize_quat(quaternion).unwrap_or(Quat::IDENTITY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    const EPSILON: f32 = 1.0e-5;

    fn assert_same_rotation(actual: Quat, expected: Quat) {
        assert!(actual.is_finite(), "rotation is not finite: {actual:?}");
        assert!(
            expected.is_finite(),
            "expected rotation is not finite: {expected:?}"
        );
        assert!(
            (actual.length() - 1.0).abs() < EPSILON,
            "rotation is not normalized: {actual:?}"
        );
        assert!(
            actual.dot(expected).abs() > 1.0 - EPSILON,
            "{actual:?} and {expected:?} do not represent the same rotation"
        );
    }

    #[test]
    fn identity_rests_transfer_animated_source_rotation() {
        let source = Quat::from_rotation_y(0.8);

        assert_same_rotation(
            rotation_constraint(Quat::IDENTITY, source, Quat::IDENTITY, 1.0),
            source,
        );
        assert_same_rotation(
            rotation_constraint(Quat::IDENTITY, source, Quat::IDENTITY, 0.5),
            Quat::from_rotation_y(0.4),
        );
    }

    #[test]
    fn non_identity_source_rest_is_removed_before_delta_application() {
        let source_rest = Quat::from_rotation_y(0.7);
        let source_delta = Quat::from_rotation_x(0.4) * Quat::from_rotation_z(-0.2);
        let source_current = source_rest * source_delta;

        assert_same_rotation(
            rotation_constraint(source_rest, source_current, Quat::IDENTITY, 1.0),
            source_delta,
        );
    }

    #[test]
    fn arbitrary_source_and_destination_rests_use_their_authored_frames() {
        let source_rest = Quat::from_rotation_y(0.4) * Quat::from_rotation_z(-0.6);
        let source_delta = Quat::from_rotation_x(0.7) * Quat::from_rotation_y(-0.25);
        let source_current = source_rest * source_delta;
        let destination_rest = Quat::from_rotation_z(0.8) * Quat::from_rotation_x(0.3);

        let expected_delta = source_rest.inverse() * source_current;
        let expected_target = destination_rest * expected_delta;

        assert_same_rotation(
            rotation_constraint(source_rest, source_current, destination_rest, 1.0),
            expected_target,
        );
    }

    #[test]
    fn destination_animation_is_not_the_interpolation_baseline() {
        let source_current = Quat::from_rotation_x(0.6);
        let destination_rest = Quat::from_rotation_z(0.5);
        let destination_current = Quat::from_rotation_y(-1.1);
        let expected_target = destination_rest * source_current;

        // `destination_current` represents an animated destination pose. It
        // is intentionally irrelevant to the pure Rotation operation.
        let output = rotation_constraint(Quat::IDENTITY, source_current, destination_rest, 0.5);
        let expected = destination_rest.slerp(expected_target, 0.5);
        assert_same_rotation(output, expected);
        assert!(output.dot(destination_current).abs() < 1.0 - EPSILON);
    }

    #[test]
    fn weights_zero_half_and_one_blend_from_destination_rest() {
        let source_delta = Quat::from_rotation_y(1.2);
        let destination_rest = Quat::from_rotation_x(-0.3);
        let target = destination_rest * source_delta;

        assert_same_rotation(
            rotation_constraint(Quat::IDENTITY, source_delta, destination_rest, 0.0),
            destination_rest,
        );
        assert_same_rotation(
            rotation_constraint(Quat::IDENTITY, source_delta, destination_rest, 0.5),
            destination_rest.slerp(target, 0.5),
        );
        assert_same_rotation(
            rotation_constraint(Quat::IDENTITY, source_delta, destination_rest, 1.0),
            target,
        );
    }

    #[test]
    fn sign_equivalent_quaternion_inputs_produce_the_same_rotation() {
        let source_rest = Quat::from_rotation_y(0.31) * Quat::from_rotation_z(-0.27);
        let source_current = source_rest * Quat::from_rotation_x(0.83);
        let destination_rest = Quat::from_rotation_z(0.42) * Quat::from_rotation_x(-0.18);
        let expected = rotation_constraint(source_rest, source_current, destination_rest, 0.5);

        for source_rest_sign in [1.0, -1.0] {
            for source_current_sign in [1.0, -1.0] {
                for destination_rest_sign in [1.0, -1.0] {
                    let actual = rotation_constraint(
                        source_rest * source_rest_sign,
                        source_current * source_current_sign,
                        destination_rest * destination_rest_sign,
                        0.5,
                    );
                    assert_same_rotation(actual, expected);
                }
            }
        }
    }

    #[test]
    fn non_normalized_inputs_still_produce_a_normalized_rotation() {
        let source_rest = Quat::from_rotation_y(0.31) * Quat::from_rotation_z(-0.27);
        let source_current = source_rest * Quat::from_rotation_x(0.83);
        let destination_rest = Quat::from_rotation_z(0.42) * Quat::from_rotation_x(-0.18);
        let expected = rotation_constraint(source_rest, source_current, destination_rest, 0.37);

        let actual = rotation_constraint(
            source_rest * 3.0,
            source_current * 0.25,
            destination_rest * 7.0,
            0.37,
        );
        assert_same_rotation(actual, expected);
    }

    #[test]
    fn invalid_quaternion_inputs_do_not_propagate_non_finite_values() {
        let output = rotation_constraint(
            Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0),
            Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
            Quat::from_rotation_y(0.4),
            0.5,
        );

        assert!(output.is_finite());
        assert!((output.length() - 1.0).abs() < EPSILON);
    }

    fn source_current_for_parent_delta(source_rest: Quat, parent_delta: Quat) -> Quat {
        let source_rest = source_rest.normalize();
        let parent_delta = parent_delta.normalize();
        (source_rest * (source_rest.conjugate() * parent_delta * source_rest).normalize())
            .normalize()
    }

    #[test]
    fn checked_helpers_reject_invalid_values_and_normalize_valid_values() {
        let quaternion = Quat::from_rotation_x(0.37) * 3.0;
        let vector = Vec3::new(2.0, -3.0, 4.0);
        assert!((checked_normalize_quat(quaternion).unwrap().length() - 1.0).abs() < 1.0e-6);
        assert!((checked_normalize_vec3(vector).unwrap().length() - 1.0).abs() < 1.0e-6);
        assert!(checked_normalize_quat(Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)).is_none());
        assert!(checked_normalize_quat(Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0)).is_none());
        assert!(checked_normalize_vec3(Vec3::ZERO).is_none());
        assert!(checked_normalize_vec3(Vec3::new(f32::INFINITY, 0.0, 0.0)).is_none());
    }

    #[test]
    fn shortest_rotation_handles_parallel_and_antiparallel_vectors_deterministically() {
        let parallel = shortest_rotation(Vec3::X, Vec3::X).unwrap();
        assert_same_rotation(parallel, Quat::IDENTITY);

        let antiparallel = shortest_rotation(Vec3::X, -Vec3::X).unwrap();
        let repeated = shortest_rotation(Vec3::X, -Vec3::X).unwrap();
        assert!(antiparallel.is_finite());
        assert_same_rotation(antiparallel, repeated);
        assert!((antiparallel * Vec3::X + Vec3::X).length() < EPSILON);

        let arbitrary = Vec3::new(0.2, -0.8, 0.55).normalize();
        let opposite = shortest_rotation(arbitrary, -arbitrary).unwrap();
        assert!(opposite.is_finite());
        assert!((opposite * arbitrary + arbitrary).length() < EPSILON);
    }

    #[test]
    fn shortest_rotation_maps_near_antiparallel_vectors_without_nan() {
        let from = Vec3::X;
        let to = Quat::from_rotation_y(std::f32::consts::PI - 1.0e-5) * from;
        let rotation = shortest_rotation(from, to).unwrap();
        assert!(rotation.is_finite());
        assert!((rotation * from - to).length() < 4.0e-5);
    }

    #[test]
    fn small_nonzero_aim_correction_is_not_collapsed_to_identity() {
        let angle = 0.0005;
        let target_direction = Quat::from_rotation_z(angle) * Vec3::X;
        let output = aim_constraint(
            target_direction,
            Vec3::ZERO,
            Quat::IDENTITY,
            Quat::IDENTITY,
            Vec3::X,
            1.0,
        );

        assert!((output * Vec3::X - target_direction).length() < 1.0e-6);
        assert!(output.xyz().length_squared() > 0.0);
    }

    #[test]
    fn pure_positive_and_negative_twist_transfers_on_x_y_and_z() {
        for (axis, typed_axis) in [
            (Vec3::X, Vrm1RollAxis::X),
            (Vec3::Y, Vrm1RollAxis::Y),
            (Vec3::Z, Vrm1RollAxis::Z),
        ] {
            for angle in [0.73, -1.11] {
                let source_current = Quat::from_axis_angle(axis, angle);
                let target = apply_roll_constraint(
                    Quat::IDENTITY,
                    source_current,
                    Quat::IDENTITY,
                    typed_axis,
                    1.0,
                );
                assert_same_rotation(target, source_current);
            }
        }
    }

    #[test]
    fn pure_swing_transfers_zero_roll_on_all_axes() {
        for (axis, swing_axis) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)] {
            let swing = Quat::from_axis_angle(swing_axis, 0.71);
            let twist = roll_twist(Quat::IDENTITY, swing, Quat::IDENTITY, axis).unwrap();
            assert_same_rotation(twist, Quat::IDENTITY);
        }
    }

    #[test]
    fn small_pure_swing_transfers_zero_roll() {
        let source_current = Quat::from_rotation_z(0.0005);
        let output = roll_constraint(Quat::IDENTITY, source_current, Quat::IDENTITY, Vec3::X, 1.0);

        assert_same_rotation(output, Quat::IDENTITY);
    }

    #[test]
    fn near_180_degree_swing_transfers_zero_roll_and_stays_finite() {
        for (axis, swing_axis) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)] {
            let swing = Quat::from_axis_angle(swing_axis, std::f32::consts::PI - 1.0e-5);
            let twist = roll_twist(Quat::IDENTITY, swing, Quat::IDENTITY, axis).unwrap();
            assert_same_rotation(twist, Quat::IDENTITY);
        }
    }

    #[test]
    fn combined_swing_and_twist_transfers_twist_only() {
        for (axis, swing_axis) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)] {
            let swing = Quat::from_axis_angle(swing_axis, -0.91);
            let twist = Quat::from_axis_angle(axis, 0.63);
            let combined = swing * twist;
            let extracted = roll_twist(Quat::IDENTITY, combined, Quat::IDENTITY, axis).unwrap();
            assert_same_rotation(extracted, twist);
        }
    }

    #[test]
    fn arbitrary_source_rest_rotation_is_converted_into_parent_space() {
        let source_rest = Quat::from_rotation_z(0.41) * Quat::from_rotation_y(-0.28);
        let destination_delta = Quat::from_axis_angle(Vec3::X, -0.52);
        let source_current = source_current_for_parent_delta(source_rest, destination_delta);
        let extracted = roll_twist(source_rest, source_current, Quat::IDENTITY, Vec3::X).unwrap();
        assert_same_rotation(extracted, destination_delta);
    }

    #[test]
    fn arbitrary_destination_rest_rotation_is_converted_into_destination_space() {
        let destination_rest = Quat::from_rotation_x(-0.33) * Quat::from_rotation_z(0.47);
        let destination_delta = Quat::from_axis_angle(Vec3::Y, 0.58);
        let parent_delta = destination_rest * destination_delta * destination_rest.conjugate();
        let extracted =
            roll_twist(Quat::IDENTITY, parent_delta, destination_rest, Vec3::Y).unwrap();
        assert_same_rotation(extracted, destination_delta);
        assert_same_rotation(
            roll_target(Quat::IDENTITY, parent_delta, destination_rest, Vec3::Y).unwrap(),
            destination_rest * destination_delta,
        );
    }

    #[test]
    fn both_arbitrary_rest_rotations_preserve_the_destination_twist() {
        let source_rest = Quat::from_rotation_y(0.31) * Quat::from_rotation_x(-0.22);
        let destination_rest = Quat::from_rotation_z(-0.37) * Quat::from_rotation_y(0.19);
        let destination_delta = Quat::from_axis_angle(Vec3::Z, -0.66);
        let parent_delta = destination_rest * destination_delta * destination_rest.conjugate();
        let source_current = source_current_for_parent_delta(source_rest, parent_delta);
        let extracted = roll_twist(source_rest, source_current, destination_rest, Vec3::Z).unwrap();
        assert_same_rotation(extracted, destination_delta);
        assert_same_rotation(
            roll_constraint(source_rest, source_current, destination_rest, Vec3::Z, 1.0),
            destination_rest * destination_delta,
        );
    }

    #[test]
    fn animated_source_rotation_is_the_current_source_input() {
        let source_rest = Quat::from_rotation_x(0.27) * Quat::from_rotation_z(-0.18);
        let expected_twist = Quat::from_rotation_y(0.49);
        let source_animation = source_rest.conjugate() * expected_twist * source_rest;
        let source_current = source_rest * source_animation;
        let extracted = roll_twist(source_rest, source_current, Quat::IDENTITY, Vec3::Y).unwrap();
        assert_same_rotation(extracted, expected_twist);
    }

    #[test]
    fn weights_blend_from_destination_rest_using_shortest_path() {
        let destination_rest = Quat::from_rotation_z(0.4);
        let source_current = Quat::from_rotation_z(1.2);
        let target =
            roll_target(Quat::IDENTITY, source_current, destination_rest, Vec3::Z).unwrap();

        assert_same_rotation(
            roll_constraint(
                Quat::IDENTITY,
                source_current,
                destination_rest,
                Vec3::Z,
                0.0,
            ),
            destination_rest,
        );
        assert_same_rotation(
            roll_constraint(
                Quat::IDENTITY,
                source_current,
                destination_rest,
                Vec3::Z,
                1.0,
            ),
            target,
        );
        let half = roll_constraint(
            Quat::IDENTITY,
            source_current,
            destination_rest,
            Vec3::Z,
            0.5,
        );
        assert_same_rotation(half, Quat::from_rotation_z(1.0));
        assert!((half.length() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn roll_normalizes_inputs_and_keeps_invalid_inputs_finite() {
        let output = roll_constraint(
            Quat::from_rotation_x(0.2) * 3.0,
            Quat::from_rotation_x(0.8) * 2.0,
            Quat::from_rotation_x(-0.4) * 4.0,
            Vec3::new(5.0, 0.0, 0.0),
            0.5,
        );
        assert!(output.is_finite());
        assert!((output.length() - 1.0).abs() < 1.0e-6);

        let output = roll_constraint(
            Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0),
            Quat::IDENTITY,
            Quat::from_rotation_z(0.4),
            Vec3::X,
            0.5,
        );
        assert_same_rotation(output, Quat::from_rotation_z(0.4));
        let output = roll_constraint(
            Quat::IDENTITY,
            Quat::IDENTITY,
            Quat::from_rotation_z(0.4),
            Vec3::ZERO,
            0.5,
        );
        assert_same_rotation(output, Quat::from_rotation_z(0.4));
    }

    #[test]
    fn slerp_is_sign_independent_and_shortest_path() {
        let from = Quat::from_rotation_z(-0.2);
        let to = -Quat::from_rotation_z(0.8);
        let halfway = slerp_shortest(from, to, 0.5).unwrap();
        assert_same_rotation(halfway, Quat::from_rotation_z(0.3));
        assert!((halfway.length() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn aim_axis_vectors_cover_all_signed_axes() {
        for (axis, expected) in [
            (Vrm1AimAxis::PositiveX, Vec3::X),
            (Vrm1AimAxis::NegativeX, -Vec3::X),
            (Vrm1AimAxis::PositiveY, Vec3::Y),
            (Vrm1AimAxis::NegativeY, -Vec3::Y),
            (Vrm1AimAxis::PositiveZ, Vec3::Z),
            (Vrm1AimAxis::NegativeZ, -Vec3::Z),
        ] {
            assert_eq!(aim_axis_vector(axis), expected);
        }
    }

    #[test]
    fn aim_rotates_each_signed_axis_toward_the_model_space_target() {
        let steering = Quat::from_rotation_y(0.63) * Quat::from_rotation_x(-0.41);
        for axis in [
            Vrm1AimAxis::PositiveX,
            Vrm1AimAxis::NegativeX,
            Vrm1AimAxis::PositiveY,
            Vrm1AimAxis::NegativeY,
            Vrm1AimAxis::PositiveZ,
            Vrm1AimAxis::NegativeZ,
        ] {
            let local_axis = aim_axis_vector(axis);
            let desired_direction = steering * local_axis;
            let output = apply_aim_constraint(
                desired_direction,
                Vec3::ZERO,
                Quat::IDENTITY,
                Quat::IDENTITY,
                axis,
                1.0,
            );

            assert!((output * local_axis - desired_direction).length() < EPSILON);
        }
    }

    #[test]
    fn aim_preserves_arbitrary_destination_rest_and_parent_rotation() {
        let destination_parent = Quat::from_rotation_y(0.71) * Quat::from_rotation_z(-0.37);
        let destination_rest = Quat::from_rotation_x(-0.44) * Quat::from_rotation_y(0.29);
        let destination_position = Vec3::new(-0.7, 0.4, 1.2);
        let source_position = Vec3::new(1.1, -0.3, -0.6);
        let axis = Vrm1AimAxis::NegativeZ;
        let local_axis = aim_axis_vector(axis);
        let from = destination_parent * (destination_rest * local_axis);
        let to = (source_position - destination_position).normalize();
        let shortest = shortest_rotation(from, to).unwrap();
        let expected =
            (destination_parent.inverse() * shortest * destination_parent * destination_rest)
                .normalize();

        let output = apply_aim_constraint(
            source_position,
            destination_position,
            destination_parent,
            destination_rest,
            axis,
            1.0,
        );
        assert_same_rotation(output, expected);
        assert!((destination_parent * (output * local_axis) - to).length() < EPSILON);
    }

    #[test]
    fn aim_uses_current_positions_from_different_scaled_parent_hierarchies() {
        fn model_point(
            parent_position: Vec3,
            parent_rotation: Quat,
            parent_scale: Vec3,
            local_position: Vec3,
        ) -> Vec3 {
            parent_position + parent_rotation * (parent_scale * local_position)
        }

        let source_parent_rotation =
            (Quat::from_rotation_z(0.22) * Quat::from_rotation_y(-0.58)).normalize();
        let destination_parent_rotation =
            (Quat::from_rotation_x(-0.36) * Quat::from_rotation_y(0.47)).normalize();
        let source_position = model_point(
            Vec3::new(-1.4, 0.8, 0.2),
            source_parent_rotation,
            Vec3::new(-1.5, 2.0, 0.5),
            Vec3::new(0.8, -0.4, 1.1),
        );
        let destination_position = model_point(
            Vec3::new(1.1, -0.6, 0.3),
            destination_parent_rotation,
            Vec3::new(0.75, -1.2, 2.0),
            Vec3::new(-0.3, 0.5, 0.7),
        );
        let destination_rest = Quat::from_rotation_z(0.51) * Quat::from_rotation_x(-0.24);
        let axis = Vrm1AimAxis::PositiveY;
        let output = apply_aim_constraint(
            source_position,
            destination_position,
            destination_parent_rotation,
            destination_rest,
            axis,
            1.0,
        );
        let target = aim_target(
            source_position,
            destination_position,
            destination_parent_rotation,
            destination_rest,
            aim_axis_vector(axis),
        )
        .unwrap();

        assert_same_rotation(output, target);
        let direction = (source_position - destination_position).normalize();
        let model_aim = destination_parent_rotation * (output * aim_axis_vector(axis));
        assert!((model_aim - direction).length() < EPSILON);

        // Changing the animated source translation changes the Aim result;
        // the destination parent rotation remains an independently accumulated
        // quaternion and is not extracted from either scaled hierarchy.
        let animated_source_position = source_position + Vec3::new(0.9, 0.2, -0.7);
        let animated_output = apply_aim_constraint(
            animated_source_position,
            destination_position,
            destination_parent_rotation,
            destination_rest,
            axis,
            1.0,
        );
        assert!((animated_output.dot(output).abs()) < 1.0 - EPSILON);
    }

    #[test]
    fn aim_weights_zero_half_and_one_blend_from_destination_rest() {
        let destination_parent = Quat::from_rotation_y(-0.42);
        let destination_rest = Quat::from_rotation_z(0.37) * Quat::from_rotation_x(-0.28);
        let destination_position = Vec3::new(0.2, -0.4, 0.7);
        let source_position = Vec3::new(-1.1, 0.9, -0.3);
        let axis = Vrm1AimAxis::PositiveX;
        let target = aim_target(
            source_position,
            destination_position,
            destination_parent,
            destination_rest,
            aim_axis_vector(axis),
        )
        .unwrap();

        assert_same_rotation(
            apply_aim_constraint(
                source_position,
                destination_position,
                destination_parent,
                destination_rest,
                axis,
                0.0,
            ),
            destination_rest,
        );
        assert_same_rotation(
            apply_aim_constraint(
                source_position,
                destination_position,
                destination_parent,
                destination_rest,
                axis,
                1.0,
            ),
            target,
        );
        assert_same_rotation(
            apply_aim_constraint(
                source_position,
                destination_position,
                destination_parent,
                destination_rest,
                axis,
                0.5,
            ),
            slerp_shortest(destination_rest, target, 0.5).unwrap(),
        );
    }

    #[test]
    fn root_destination_uses_identity_parent_rotation() {
        let destination_rest = Quat::from_rotation_y(0.35) * Quat::from_rotation_z(-0.19);
        let axis = Vrm1AimAxis::NegativeX;
        let target_direction = Vec3::new(0.3, -0.8, 0.5).normalize();
        let output = apply_aim_constraint(
            target_direction,
            Vec3::ZERO,
            Quat::IDENTITY,
            destination_rest,
            axis,
            1.0,
        );

        assert!((output * aim_axis_vector(axis) - target_direction).length() < EPSILON);
    }

    #[test]
    fn coincident_positions_return_the_normalized_destination_rest() {
        let destination_parent = Quat::from_rotation_x(0.64) * 2.0;
        let destination_rest = Quat::from_rotation_z(-0.51) * 3.0;
        let position = Vec3::new(2.0, -1.0, 0.5);

        assert!(
            aim_target(
                position,
                position,
                destination_parent,
                destination_rest,
                Vec3::Y,
            )
            .is_none()
        );
        assert_same_rotation(
            aim_constraint(
                position,
                position,
                destination_parent,
                destination_rest,
                Vec3::Y,
                0.5,
            ),
            destination_rest,
        );
    }

    #[test]
    fn exact_opposite_direction_has_a_deterministic_finite_fallback() {
        let destination_parent = Quat::from_rotation_y(0.43) * Quat::from_rotation_z(-0.27);
        let destination_rest = Quat::from_rotation_x(-0.31) * Quat::from_rotation_y(0.52);
        let axis = Vrm1AimAxis::PositiveZ;
        let destination_position = Vec3::new(-0.2, 0.7, 1.4);
        let from = destination_parent * (destination_rest * aim_axis_vector(axis));
        let source_position = destination_position - from;

        let first = apply_aim_constraint(
            source_position,
            destination_position,
            destination_parent,
            destination_rest,
            axis,
            1.0,
        );
        let second = apply_aim_constraint(
            source_position,
            destination_position,
            destination_parent,
            destination_rest,
            axis,
            1.0,
        );

        assert_same_rotation(first, second);
        assert!((destination_parent * (first * aim_axis_vector(axis)) + from).length() < EPSILON);
    }

    #[test]
    fn invalid_aim_inputs_fall_back_to_a_finite_rest_rotation() {
        let destination_rest = Quat::from_rotation_y(0.4);
        let output = aim_constraint(
            Vec3::new(f32::NAN, 0.0, 0.0),
            Vec3::ZERO,
            Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0),
            destination_rest,
            Vec3::X,
            0.5,
        );

        assert_same_rotation(output, destination_rest);
    }
}
