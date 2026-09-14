//! pocket-vrm — VRM 0.x character semantics on the pocket3d substrate.
//!
//! Scope (engine infrastructure only; rendering, mesh loading, and app
//! behavior stay in the host):
//! - **Parsing**: the `extensions.VRM` block of a `.vrm` GLB — humanoid bone
//!   map, blend-shape expressions, spring-bone config, look-at ranges, MToon
//!   material facts — into plain-data structs ([`VrmDoc`]). The additive
//!   [`Vrm1Doc`] parser handles minimal VRM 1.0 semantic facts separately.
//! - **Spring bones**: UniVRM-style verlet simulation ([`SpringSolver`])
//!   writing local rotations into a pocket3d skeleton pose.
//! - **VRMA**: `.vrma` (VRMC_vrm_animation) loading and humanoid retargeting
//!   onto a model skeleton as a pocket3d [`pocket3d::anim::Clip`]
//!   ([`load_vrma_bytes`], [`retarget`]).
//! - **Eye look-at**: bone-type yaw/pitch onto the eye bones, limited by the
//!   model's ranges ([`apply_eye_look`]).
//!
//! Conventions verified on the VRoid `AvatarSample_A` fixture:
//! - VRM 0.x glTF data faces **-Z**: the left eye rests at negative X, the
//!   right eye at positive X, so the model's left is -X and a face-on camera
//!   sits at -Z looking toward +Z.
//! - VRM0 `materialProperties` are index-aligned with glTF `materials`
//!   (same count and order).
//! - The normalized VRM0 rest pose has identity rotations and unit scales.

pub mod glb;
pub mod lookat;
pub mod node_constraint;
pub mod parse;
pub mod spring;
pub mod vrm1;
pub mod vrm1_lookat;
pub mod vrma;

pub use lookat::apply_eye_look;
pub use node_constraint::{
    aim_axis_vector, aim_constraint, aim_target, apply_aim_constraint, apply_roll_constraint,
    checked_normalize_quat, checked_normalize_vec3, roll_axis_vector, roll_constraint, roll_target,
    roll_twist, rotation_constraint, shortest_rotation, slerp_shortest,
};
pub use parse::{
    ColliderGroup, LookAtDegreeMap, LookAtRanges, MorphBind, SphereCollider, SpringConfig,
    SpringGroup, VrmDoc, VrmExpression, VrmMaterialInfo, VrmMeta,
};
pub use spring::SpringSolver;
pub use vrm1::{
    Vrm1AimAxis, Vrm1CapsuleCollider, Vrm1Collider, Vrm1ColliderGroup, Vrm1ColliderShape, Vrm1Doc,
    Vrm1Expression, Vrm1ExpressionKind, Vrm1ExpressionOverride, Vrm1ExtensionInfo, Vrm1HumanBone,
    Vrm1Humanoid, Vrm1LookAt, Vrm1LookAtRangeMap, Vrm1LookAtType, Vrm1Meta, Vrm1MorphTargetBind,
    Vrm1NodeConstraint, Vrm1NodeConstraintKind, Vrm1NodeConstraintSet, Vrm1RollAxis,
    Vrm1SphereCollider, Vrm1Spring, Vrm1SpringBone, Vrm1SpringJoint,
};
pub use vrm1_lookat::{
    Vrm1ExpressionLookAt, Vrm1EyeBoneOutput, Vrm1LookAtAngles, Vrm1LookAtOutput, Vrm1LookAtRuntime,
    map_vrm1_look_at_range,
};
pub use vrma::{HumanoidView, VrmaDoc, load_vrma_bytes, retarget, retarget_with_humanoid};
