//! Typed VRM 1.0 semantic data, including the VRMC_springBone extension.
//!
//! Runtime objects and transforms deliberately do not live here. This module
//! validates JSON indices and hierarchy relationships before an avatar
//! candidate is committed.

use std::collections::{BTreeMap, HashSet};

use anyhow::{Context, Result, bail, ensure};
use glam::Vec3;
use serde_json::{Map, Value};

use crate::glb;
use crate::mtoon::{Vrm1MtoonMaterial, parse_mtoon_materials};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vrm1Meta {
    pub name: String,
    pub version: Option<String>,
    pub authors: Vec<String>,
    pub license_url: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Vrm1HumanBone {
    Hips,
    Spine,
    Chest,
    UpperChest,
    Neck,
    Head,
    LeftEye,
    RightEye,
    Jaw,
    LeftUpperLeg,
    LeftLowerLeg,
    LeftFoot,
    LeftToes,
    RightUpperLeg,
    RightLowerLeg,
    RightFoot,
    RightToes,
    LeftShoulder,
    LeftUpperArm,
    LeftLowerArm,
    LeftHand,
    RightShoulder,
    RightUpperArm,
    RightLowerArm,
    RightHand,
    LeftThumbMetacarpal,
    LeftThumbProximal,
    LeftThumbDistal,
    LeftIndexProximal,
    LeftIndexIntermediate,
    LeftIndexDistal,
    LeftMiddleProximal,
    LeftMiddleIntermediate,
    LeftMiddleDistal,
    LeftRingProximal,
    LeftRingIntermediate,
    LeftRingDistal,
    LeftLittleProximal,
    LeftLittleIntermediate,
    LeftLittleDistal,
    RightThumbMetacarpal,
    RightThumbProximal,
    RightThumbDistal,
    RightIndexProximal,
    RightIndexIntermediate,
    RightIndexDistal,
    RightMiddleProximal,
    RightMiddleIntermediate,
    RightMiddleDistal,
    RightRingProximal,
    RightRingIntermediate,
    RightRingDistal,
    RightLittleProximal,
    RightLittleIntermediate,
    RightLittleDistal,
}

impl Vrm1HumanBone {
    pub(crate) const REQUIRED: &'static [Self] = &[
        Self::Hips,
        Self::Spine,
        Self::Head,
        Self::LeftUpperLeg,
        Self::LeftLowerLeg,
        Self::LeftFoot,
        Self::RightUpperLeg,
        Self::RightLowerLeg,
        Self::RightFoot,
        Self::LeftUpperArm,
        Self::LeftLowerArm,
        Self::LeftHand,
        Self::RightUpperArm,
        Self::RightLowerArm,
        Self::RightHand,
    ];
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "hips" => Self::Hips,
            "spine" => Self::Spine,
            "chest" => Self::Chest,
            "upperChest" => Self::UpperChest,
            "neck" => Self::Neck,
            "head" => Self::Head,
            "leftEye" => Self::LeftEye,
            "rightEye" => Self::RightEye,
            "jaw" => Self::Jaw,
            "leftUpperLeg" => Self::LeftUpperLeg,
            "leftLowerLeg" => Self::LeftLowerLeg,
            "leftFoot" => Self::LeftFoot,
            "leftToes" => Self::LeftToes,
            "rightUpperLeg" => Self::RightUpperLeg,
            "rightLowerLeg" => Self::RightLowerLeg,
            "rightFoot" => Self::RightFoot,
            "rightToes" => Self::RightToes,
            "leftShoulder" => Self::LeftShoulder,
            "leftUpperArm" => Self::LeftUpperArm,
            "leftLowerArm" => Self::LeftLowerArm,
            "leftHand" => Self::LeftHand,
            "rightShoulder" => Self::RightShoulder,
            "rightUpperArm" => Self::RightUpperArm,
            "rightLowerArm" => Self::RightLowerArm,
            "rightHand" => Self::RightHand,
            "leftThumbMetacarpal" => Self::LeftThumbMetacarpal,
            "leftThumbProximal" => Self::LeftThumbProximal,
            "leftThumbDistal" => Self::LeftThumbDistal,
            "leftIndexProximal" => Self::LeftIndexProximal,
            "leftIndexIntermediate" => Self::LeftIndexIntermediate,
            "leftIndexDistal" => Self::LeftIndexDistal,
            "leftMiddleProximal" => Self::LeftMiddleProximal,
            "leftMiddleIntermediate" => Self::LeftMiddleIntermediate,
            "leftMiddleDistal" => Self::LeftMiddleDistal,
            "leftRingProximal" => Self::LeftRingProximal,
            "leftRingIntermediate" => Self::LeftRingIntermediate,
            "leftRingDistal" => Self::LeftRingDistal,
            "leftLittleProximal" => Self::LeftLittleProximal,
            "leftLittleIntermediate" => Self::LeftLittleIntermediate,
            "leftLittleDistal" => Self::LeftLittleDistal,
            "rightThumbMetacarpal" => Self::RightThumbMetacarpal,
            "rightThumbProximal" => Self::RightThumbProximal,
            "rightThumbDistal" => Self::RightThumbDistal,
            "rightIndexProximal" => Self::RightIndexProximal,
            "rightIndexIntermediate" => Self::RightIndexIntermediate,
            "rightIndexDistal" => Self::RightIndexDistal,
            "rightMiddleProximal" => Self::RightMiddleProximal,
            "rightMiddleIntermediate" => Self::RightMiddleIntermediate,
            "rightMiddleDistal" => Self::RightMiddleDistal,
            "rightRingProximal" => Self::RightRingProximal,
            "rightRingIntermediate" => Self::RightRingIntermediate,
            "rightRingDistal" => Self::RightRingDistal,
            "rightLittleProximal" => Self::RightLittleProximal,
            "rightLittleIntermediate" => Self::RightLittleIntermediate,
            "rightLittleDistal" => Self::RightLittleDistal,
            _ => return None,
        })
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hips => "hips",
            Self::Spine => "spine",
            Self::Chest => "chest",
            Self::UpperChest => "upperChest",
            Self::Neck => "neck",
            Self::Head => "head",
            Self::LeftEye => "leftEye",
            Self::RightEye => "rightEye",
            Self::Jaw => "jaw",
            Self::LeftUpperLeg => "leftUpperLeg",
            Self::LeftLowerLeg => "leftLowerLeg",
            Self::LeftFoot => "leftFoot",
            Self::LeftToes => "leftToes",
            Self::RightUpperLeg => "rightUpperLeg",
            Self::RightLowerLeg => "rightLowerLeg",
            Self::RightFoot => "rightFoot",
            Self::RightToes => "rightToes",
            Self::LeftShoulder => "leftShoulder",
            Self::LeftUpperArm => "leftUpperArm",
            Self::LeftLowerArm => "leftLowerArm",
            Self::LeftHand => "leftHand",
            Self::RightShoulder => "rightShoulder",
            Self::RightUpperArm => "rightUpperArm",
            Self::RightLowerArm => "rightLowerArm",
            Self::RightHand => "rightHand",
            Self::LeftThumbMetacarpal => "leftThumbMetacarpal",
            Self::LeftThumbProximal => "leftThumbProximal",
            Self::LeftThumbDistal => "leftThumbDistal",
            Self::LeftIndexProximal => "leftIndexProximal",
            Self::LeftIndexIntermediate => "leftIndexIntermediate",
            Self::LeftIndexDistal => "leftIndexDistal",
            Self::LeftMiddleProximal => "leftMiddleProximal",
            Self::LeftMiddleIntermediate => "leftMiddleIntermediate",
            Self::LeftMiddleDistal => "leftMiddleDistal",
            Self::LeftRingProximal => "leftRingProximal",
            Self::LeftRingIntermediate => "leftRingIntermediate",
            Self::LeftRingDistal => "leftRingDistal",
            Self::LeftLittleProximal => "leftLittleProximal",
            Self::LeftLittleIntermediate => "leftLittleIntermediate",
            Self::LeftLittleDistal => "leftLittleDistal",
            Self::RightThumbMetacarpal => "rightThumbMetacarpal",
            Self::RightThumbProximal => "rightThumbProximal",
            Self::RightThumbDistal => "rightThumbDistal",
            Self::RightIndexProximal => "rightIndexProximal",
            Self::RightIndexIntermediate => "rightIndexIntermediate",
            Self::RightIndexDistal => "rightIndexDistal",
            Self::RightMiddleProximal => "rightMiddleProximal",
            Self::RightMiddleIntermediate => "rightMiddleIntermediate",
            Self::RightMiddleDistal => "rightMiddleDistal",
            Self::RightRingProximal => "rightRingProximal",
            Self::RightRingIntermediate => "rightRingIntermediate",
            Self::RightRingDistal => "rightRingDistal",
            Self::RightLittleProximal => "rightLittleProximal",
            Self::RightLittleIntermediate => "rightLittleIntermediate",
            Self::RightLittleDistal => "rightLittleDistal",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vrm1Humanoid {
    pub human_bones: BTreeMap<Vrm1HumanBone, usize>,
}
impl Vrm1Humanoid {
    pub fn node_for(&self, bone: Vrm1HumanBone) -> Option<usize> {
        self.human_bones.get(&bone).copied()
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vrm1ExtensionInfo {
    pub present: bool,
    pub version: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vrm1RollAxis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vrm1AimAxis {
    PositiveX,
    NegativeX,
    PositiveY,
    NegativeY,
    PositiveZ,
    NegativeZ,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Vrm1NodeConstraintKind {
    Roll {
        source: usize,
        axis: Vrm1RollAxis,
        weight: f32,
    },
    Aim {
        source: usize,
        axis: Vrm1AimAxis,
        weight: f32,
    },
    Rotation {
        source: usize,
        weight: f32,
    },
}

impl Vrm1NodeConstraintKind {
    fn source(self) -> usize {
        match self {
            Self::Roll { source, .. }
            | Self::Aim { source, .. }
            | Self::Rotation { source, .. } => source,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vrm1NodeConstraint {
    pub destination: usize,
    pub kind: Vrm1NodeConstraintKind,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vrm1NodeConstraintSet {
    pub constraints: Vec<Vrm1NodeConstraint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vrm1ExpressionKind {
    Preset,
    Custom,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vrm1ExpressionOverride {
    None,
    Block,
    Blend,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vrm1LookAtType {
    Bone,
    Expression,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vrm1LookAtRangeMap {
    pub input_max_value: f32,
    pub output_scale: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1LookAt {
    pub kind: Vrm1LookAtType,
    pub offset_from_head_bone: Vec3,
    pub range_map_horizontal_inner: Vrm1LookAtRangeMap,
    pub range_map_horizontal_outer: Vrm1LookAtRangeMap,
    pub range_map_vertical_down: Vrm1LookAtRangeMap,
    pub range_map_vertical_up: Vrm1LookAtRangeMap,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1MorphTargetBind {
    pub node: usize,
    pub index: usize,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vrm1MaterialColorBindType {
    Color,
    EmissionColor,
    ShadeColor,
    MatcapColor,
    RimColor,
    OutlineColor,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1MaterialColorBind {
    pub material: usize,
    pub kind: Vrm1MaterialColorBindType,
    pub target_value: [f32; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1TextureTransformBind {
    pub material: usize,
    pub scale: [f32; 2],
    pub offset: [f32; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1Expression {
    pub name: String,
    pub kind: Vrm1ExpressionKind,
    pub morph_target_binds: Vec<Vrm1MorphTargetBind>,
    pub material_color_binds: Vec<Vrm1MaterialColorBind>,
    pub texture_transform_binds: Vec<Vrm1TextureTransformBind>,
    pub is_binary: bool,
    pub override_blink: Vrm1ExpressionOverride,
    pub override_look_at: Vrm1ExpressionOverride,
    pub override_mouth: Vrm1ExpressionOverride,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1SphereCollider {
    pub offset: Vec3,
    pub radius: f32,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1CapsuleCollider {
    pub offset: Vec3,
    pub tail: Vec3,
    pub radius: f32,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Vrm1ColliderShape {
    Sphere(Vrm1SphereCollider),
    Capsule(Vrm1CapsuleCollider),
}
#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1Collider {
    pub node: usize,
    pub shape: Vrm1ColliderShape,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1ColliderGroup {
    pub name: Option<String>,
    pub colliders: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1SpringJoint {
    pub node: usize,
    pub hit_radius: f32,
    pub stiffness: f32,
    pub gravity_power: f32,
    pub gravity_dir: Vec3,
    pub drag_force: f32,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1Spring {
    pub name: Option<String>,
    pub center: Option<usize>,
    pub collider_groups: Vec<usize>,
    pub joints: Vec<Vrm1SpringJoint>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1SpringBone {
    pub colliders: Vec<Vrm1Collider>,
    pub collider_groups: Vec<Vrm1ColliderGroup>,
    pub springs: Vec<Vrm1Spring>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1Doc {
    pub meta: Vrm1Meta,
    pub humanoid: Vrm1Humanoid,
    pub materials_mtoon: Vrm1ExtensionInfo,
    pub mtoon_materials: Vec<Vrm1MtoonMaterial>,
    pub spring_bone: Vrm1ExtensionInfo,
    pub spring_bone_semantics: Option<Vrm1SpringBone>,
    pub node_constraint: Vrm1ExtensionInfo,
    pub node_constraint_semantics: Option<Vrm1NodeConstraintSet>,
    pub expressions: Vec<Vrm1Expression>,
    pub has_expressions: bool,
    pub look_at: Option<Vrm1LookAt>,
    pub has_first_person: bool,
    pub node_count: usize,
    pub node_meshes: Vec<Option<usize>>,
    pub mesh_primitive_counts: Vec<usize>,
}

impl Vrm1Doc {
    /// Convert the already-validated VRM semantics into Pocket3D's authored
    /// renderer descriptors. This is the only MToon semantic handoff; the
    /// renderer does not parse the extension JSON again.
    pub fn mtoon_material_descriptors(&self) -> Vec<pocket3d::material::MtoonMaterialDescriptor> {
        self.mtoon_materials
            .iter()
            .map(Vrm1MtoonMaterial::to_pocket3d_descriptor)
            .collect()
    }

    pub fn from_glb_bytes(bytes: &[u8]) -> Result<Self> {
        let glb = glb::parse_glb(bytes).context("failed to parse VRM 1.0 GLB")?;
        let ex = glb
            .json
            .get("extensions")
            .and_then(Value::as_object)
            .context("GLB root is missing an extensions object")?;
        ensure!(
            !ex.contains_key("VRM"),
            "legacy VRM 0.x extension is not accepted by Vrm1Doc"
        );
        let vrm = ex
            .get("VRMC_vrm")
            .context("GLB root is missing required VRMC_vrm extension")?
            .as_object()
            .context("root VRMC_vrm extension must be an object")?;
        ensure!(
            required_string(vrm, "specVersion", "VRMC_vrm")? == "1.0",
            "VRMC_vrm specVersion must be \"1.0\""
        );
        let (children, node_meshes, mesh_primitive_counts, parents) = validate_nodes(&glb.json)?;
        let spring_bone = extension_info(ex, "VRMC_springBone")?;
        let spring_bone_semantics = ex
            .get("VRMC_springBone")
            .map(|v| parse_spring_bone(v, children.len(), &parents))
            .transpose()?;
        let node_constraint_required = extension_declaration_contains(
            &glb.json,
            "extensionsRequired",
            "VRMC_node_constraint",
        )?;
        let (node_constraint, node_constraint_semantics) =
            match parse_node_constraints(&glb.json, children.len()) {
                Ok(parsed) => parsed,
                Err(error) if node_constraint_required => return Err(error),
                Err(error) => {
                    log::warn!(
                        "disabling malformed optional VRMC_node_constraint extension: {error:#}"
                    );
                    (Vrm1ExtensionInfo::default(), None)
                }
            };
        let look_at = match parse_look_at(vrm) {
            Ok(look_at) => look_at,
            Err(error) => {
                log::warn!("disabling malformed VRM 1.0 LookAt metadata: {error:#}");
                None
            }
        };
        let mtoon_required = extension_declaration_contains(
            &glb.json,
            "extensionsRequired",
            "VRMC_materials_mtoon",
        )?;
        let mtoon_used =
            extension_declaration_contains(&glb.json, "extensionsUsed", "VRMC_materials_mtoon")?;
        ensure!(
            !mtoon_required || mtoon_used,
            "glTF extensionsRequired contains VRMC_materials_mtoon, but extensionsUsed does not"
        );
        let mtoon_materials = match parse_mtoon_materials(&glb.json).and_then(|materials| {
            ensure!(
                materials.is_empty() || mtoon_used,
                "VRMC_materials_mtoon is present on a material but is missing from glTF extensionsUsed"
            );
            ensure!(
                !mtoon_required || !materials.is_empty(),
                "glTF extensionsRequired contains VRMC_materials_mtoon, but no material declares the extension"
            );
            for material in &materials {
                if let Some((role, tex_coord)) = material.unsupported_texture_coordinate(1) {
                    bail!(
                        "material {} {role} selects TEXCOORD_{tex_coord}, but native MToon supports only TEXCOORD_0 and TEXCOORD_1",
                        material.material_index
                    );
                }
            }
            Ok(materials)
        }) {
            Ok(materials) => materials,
            Err(error) if mtoon_required => return Err(error),
            Err(error) => {
                log::warn!(
                    "disabling malformed or unsupported optional VRMC_materials_mtoon extension: {error:#}"
                );
                Vec::new()
            }
        };
        let materials_mtoon = if mtoon_materials.is_empty() {
            Vrm1ExtensionInfo::default()
        } else {
            Vrm1ExtensionInfo {
                present: true,
                version: Some("1.0".to_owned()),
            }
        };
        Ok(Self {
            meta: parse_meta(vrm)?,
            humanoid: parse_humanoid(vrm, children.len())?,
            materials_mtoon,
            mtoon_materials,
            spring_bone,
            spring_bone_semantics,
            node_constraint,
            node_constraint_semantics,
            expressions: parse_expressions(
                vrm,
                glb.json
                    .get("materials")
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len),
            )?,
            has_expressions: vrm.contains_key("expressions"),
            look_at,
            has_first_person: optional_object(vrm, "firstPerson")?,
            node_count: children.len(),
            node_meshes,
            mesh_primitive_counts,
        })
    }
}

fn parse_spring_bone(
    value: &Value,
    node_count: usize,
    parents: &[Option<usize>],
) -> Result<Vrm1SpringBone> {
    let root = value
        .as_object()
        .context("VRMC_springBone extension must be an object")?;
    ensure!(
        required_string(root, "specVersion", "VRMC_springBone")? == "1.0",
        "VRMC_springBone.specVersion must be \"1.0\""
    );
    let colliders = match root.get("colliders") {
        None => Vec::new(),
        Some(v) => {
            let a = v
                .as_array()
                .context("VRMC_springBone.colliders must be an array")?;
            ensure!(!a.is_empty(), "VRMC_springBone.colliders must not be empty");
            a.iter()
                .enumerate()
                .map(|(i, v)| parse_collider(v, i, node_count))
                .collect::<Result<Vec<_>>>()?
        }
    };
    let collider_groups = match root.get("colliderGroups") {
        None => Vec::new(),
        Some(v) => {
            let a = v
                .as_array()
                .context("VRMC_springBone.colliderGroups must be an array")?;
            ensure!(
                !a.is_empty(),
                "VRMC_springBone.colliderGroups must not be empty"
            );
            a.iter()
                .enumerate()
                .map(|(i, v)| {
                    let o = v
                        .as_object()
                        .with_context(|| format!("colliderGroups[{i}] must be an object"))?;
                    let a = o
                        .get("colliders")
                        .with_context(|| format!("colliderGroups[{i}] is missing colliders"))?
                        .as_array()
                        .with_context(|| {
                            format!("colliderGroups[{i}].colliders must be an array")
                        })?;
                    ensure!(
                        !a.is_empty(),
                        "colliderGroups[{i}].colliders must not be empty"
                    );
                    let refs = a
                        .iter()
                        .enumerate()
                        .map(|(j, v)| {
                            let r = index_value(v, &format!("colliderGroups[{i}].colliders[{j}]"))?;
                            ensure!(
                                r < colliders.len(),
                                "colliderGroups[{i}] references collider {r} out of range"
                            );
                            Ok(r)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Vrm1ColliderGroup {
                        name: optional_string(o, "name", &format!("colliderGroups[{i}]"))?,
                        colliders: refs,
                    })
                })
                .collect::<Result<Vec<_>>>()?
        }
    };
    let springs = match root.get("springs") {
        None => Vec::new(),
        Some(v) => {
            let a = v
                .as_array()
                .context("VRMC_springBone.springs must be an array")?;
            ensure!(!a.is_empty(), "VRMC_springBone.springs must not be empty");
            a.iter()
                .enumerate()
                .map(|(i, v)| parse_spring(v, i, node_count, collider_groups.len()))
                .collect::<Result<Vec<_>>>()?
        }
    };
    validate_spring_relationships(&springs, parents)?;
    Ok(Vrm1SpringBone {
        colliders,
        collider_groups,
        springs,
    })
}
fn parse_collider(value: &Value, index: usize, node_count: usize) -> Result<Vrm1Collider> {
    let o = value
        .as_object()
        .with_context(|| format!("colliders[{index}] must be an object"))?;
    let node = index_value(
        o.get("node")
            .with_context(|| format!("colliders[{index}] is missing node"))?,
        &format!("colliders[{index}].node"),
    )?;
    ensure!(
        node < node_count,
        "colliders[{index}].node {node} is out of range"
    );
    let shape = o
        .get("shape")
        .with_context(|| format!("colliders[{index}] is missing shape"))?
        .as_object()
        .with_context(|| format!("colliders[{index}].shape must be an object"))?;
    let shape_count =
        usize::from(shape.contains_key("sphere")) + usize::from(shape.contains_key("capsule"));
    ensure!(
        shape_count == 1,
        "colliders[{index}].shape must contain exactly one sphere or capsule"
    );
    let shape = if let Some(v) = shape.get("sphere") {
        let s = v
            .as_object()
            .with_context(|| format!("colliders[{index}].shape.sphere must be an object"))?;
        Vrm1ColliderShape::Sphere(Vrm1SphereCollider {
            offset: optional_vec3(s, "offset", &format!("colliders[{index}].sphere"))?
                .unwrap_or(Vec3::ZERO),
            radius: nonnegative(
                optional_f32(s, "radius", &format!("colliders[{index}].sphere"))?.unwrap_or(0.0),
                &format!("colliders[{index}].sphere.radius"),
            )?,
        })
    } else if let Some(v) = shape.get("capsule") {
        let s = v
            .as_object()
            .with_context(|| format!("colliders[{index}].shape.capsule must be an object"))?;
        Vrm1ColliderShape::Capsule(Vrm1CapsuleCollider {
            offset: optional_vec3(s, "offset", &format!("colliders[{index}].capsule"))?
                .unwrap_or(Vec3::ZERO),
            tail: optional_vec3(s, "tail", &format!("colliders[{index}].capsule"))?
                .unwrap_or(Vec3::ZERO),
            radius: nonnegative(
                optional_f32(s, "radius", &format!("colliders[{index}].capsule"))?.unwrap_or(0.0),
                &format!("colliders[{index}].capsule.radius"),
            )?,
        })
    } else {
        bail!("colliders[{index}].shape must contain sphere or capsule")
    };
    Ok(Vrm1Collider { node, shape })
}
fn parse_spring(
    value: &Value,
    index: usize,
    node_count: usize,
    group_count: usize,
) -> Result<Vrm1Spring> {
    let o = value
        .as_object()
        .with_context(|| format!("springs[{index}] must be an object"))?;
    let a = o
        .get("joints")
        .with_context(|| format!("springs[{index}] is missing joints"))?
        .as_array()
        .with_context(|| format!("springs[{index}].joints must be an array"))?;
    ensure!(!a.is_empty(), "springs[{index}].joints must not be empty");
    let joints = a
        .iter()
        .enumerate()
        .map(|(j, v)| {
            let x = v
                .as_object()
                .with_context(|| format!("springs[{index}].joints[{j}] must be an object"))?;
            let c = format!("springs[{index}].joints[{j}]");
            let node = index_value(
                x.get("node")
                    .with_context(|| format!("{c} is missing node"))?,
                &format!("{c}.node"),
            )?;
            ensure!(node < node_count, "{c}.node {node} is out of range");
            Ok(Vrm1SpringJoint {
                node,
                hit_radius: nonnegative(
                    optional_f32(x, "hitRadius", &c)?.unwrap_or(0.0),
                    &format!("{c}.hitRadius"),
                )?,
                stiffness: nonnegative(
                    optional_f32(x, "stiffness", &c)?.unwrap_or(1.0),
                    &format!("{c}.stiffness"),
                )?,
                gravity_power: nonnegative(
                    optional_f32(x, "gravityPower", &c)?.unwrap_or(0.0),
                    &format!("{c}.gravityPower"),
                )?,
                gravity_dir: optional_vec3(x, "gravityDir", &c)?
                    .unwrap_or(Vec3::new(0.0, -1.0, 0.0)),
                drag_force: bounded(
                    optional_f32(x, "dragForce", &c)?.unwrap_or(0.5),
                    0.0,
                    1.0,
                    &format!("{c}.dragForce"),
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let groups = match o.get("colliderGroups") {
        None => Vec::new(),
        Some(v) => {
            let a = v
                .as_array()
                .with_context(|| format!("springs[{index}].colliderGroups must be an array"))?;
            ensure!(
                !a.is_empty(),
                "springs[{index}].colliderGroups must not be empty"
            );
            a.iter()
                .enumerate()
                .map(|(j, v)| {
                    let r = index_value(v, &format!("springs[{index}].colliderGroups[{j}]"))?;
                    ensure!(
                        r < group_count,
                        "springs[{index}] references collider group {r} out of range"
                    );
                    Ok(r)
                })
                .collect::<Result<Vec<_>>>()?
        }
    };
    let center = optional_index(o, "center", &format!("springs[{index}]"))?;
    if let Some(n) = center {
        ensure!(
            n < node_count,
            "springs[{index}].center {n} is out of range"
        )
    }
    Ok(Vrm1Spring {
        name: optional_string(o, "name", &format!("springs[{index}]"))?,
        center,
        collider_groups: groups,
        joints,
    })
}
fn validate_spring_relationships(springs: &[Vrm1Spring], parents: &[Option<usize>]) -> Result<()> {
    let mut owners = vec![None; parents.len()];
    for (si, s) in springs.iter().enumerate() {
        for p in s.joints.windows(2) {
            ensure!(
                is_ancestor(p[0].node, p[1].node, parents),
                "spring {si} joint {} must be an ancestor of joint {}",
                p[0].node,
                p[1].node
            );
            let mut node = p[1].node;
            loop {
                if let Some(owner) = owners[node] {
                    ensure!(
                        owner == si,
                        "spring joint node {node} is owned by more than one spring"
                    )
                } else {
                    owners[node] = Some(si);
                }
                if node == p[0].node {
                    break;
                }
                node = parents[node].context("spring joint ancestry is malformed")?;
            }
        }
        if s.joints.len() == 1 {
            let node = s.joints[0].node;
            ensure!(
                owners[node].is_none(),
                "spring joint node {node} is owned by more than one spring"
            );
            owners[node] = Some(si);
        }
        if let Some(c) = s.center {
            ensure!(
                c == s.joints[0].node || is_ancestor(c, s.joints[0].node, parents),
                "spring {si} center must be the first joint or one of its ancestors"
            )
        }
    }
    for (si, s) in springs.iter().enumerate() {
        if let Some(c) = s.center {
            for (oi, o) in springs.iter().enumerate() {
                if si != oi {
                    for j in &o.joints {
                        ensure!(
                            !(c == j.node || is_ancestor(j.node, c, parents)),
                            "spring {si} center is a joint or descendant of another spring"
                        )
                    }
                }
            }
        }
    }
    Ok(())
}
fn is_ancestor(a: usize, mut n: usize, p: &[Option<usize>]) -> bool {
    while let Some(x) = p[n] {
        if x == a {
            return true;
        }
        n = x
    }
    false
}

fn parse_expressions(
    vrm: &Map<String, Value>,
    material_count: usize,
) -> Result<Vec<Vrm1Expression>> {
    let Some(v) = vrm.get("expressions") else {
        return Ok(Vec::new());
    };
    let g = v
        .as_object()
        .context("VRMC_vrm.expressions must be an object when present")?;
    let mut out = Vec::new();
    for (group, kind) in [
        ("preset", Vrm1ExpressionKind::Preset),
        ("custom", Vrm1ExpressionKind::Custom),
    ] {
        let Some(v) = g.get(group) else { continue };
        let o = v.as_object().with_context(|| {
            format!("VRMC_vrm.expressions.{group} must be an object when present")
        })?;
        for (name, v) in o {
            let x = v.as_object().with_context(|| {
                format!("VRMC_vrm.expressions.{group}.{name} must be an object")
            })?;
            if kind == Vrm1ExpressionKind::Preset {
                ensure!(
                    is_preset(name),
                    "unknown VRM 1.0 preset expression {name:?}"
                )
            } else {
                ensure!(
                    !is_preset(name),
                    "custom VRM 1.0 expression name {name:?} conflicts with a preset"
                )
            }
            let morph_target_binds = match x.get("morphTargetBinds") {
                None => Vec::new(),
                Some(v) => {
                    let a = v.as_array().with_context(|| {
                        format!("expression {name}.morphTargetBinds must be an array")
                    })?;
                    a.iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let b = v.as_object().with_context(|| {
                                format!("expression {name}.morphTargetBinds[{i}] must be an object")
                            })?;
                            Ok(Vrm1MorphTargetBind {
                                node: index_value(
                                    b.get("node").with_context(|| {
                                        format!("expression {name} morph bind is missing node")
                                    })?,
                                    "morph bind.node",
                                )?,
                                index: index_value(
                                    b.get("index").with_context(|| {
                                        format!("expression {name} morph bind is missing index")
                                    })?,
                                    "morph bind.index",
                                )?,
                                weight: required_f32(
                                    b.get("weight").with_context(|| {
                                        format!("expression {name} morph bind is missing weight")
                                    })?,
                                    "morph bind.weight",
                                )?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?
                }
            };
            let material_color_binds = match x.get("materialColorBinds") {
                None => Vec::new(),
                Some(value) => value
                    .as_array()
                    .with_context(|| {
                        format!("expression {name}.materialColorBinds must be an array")
                    })?
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        let context = format!("expression {name}.materialColorBinds[{index}]");
                        let bind = value
                            .as_object()
                            .with_context(|| format!("{context} must be an object"))?;
                        let material = index_value(
                            bind.get("material")
                                .with_context(|| format!("{context} is missing material"))?,
                            &format!("{context}.material"),
                        )?;
                        ensure!(
                            material < material_count,
                            "{context}.material index {material} is out of range for {material_count} materials"
                        );
                        let kind = match bind
                            .get("type")
                            .and_then(Value::as_str)
                            .with_context(|| format!("{context}.type must be a string"))?
                        {
                            "color" => Vrm1MaterialColorBindType::Color,
                            "emissionColor" => Vrm1MaterialColorBindType::EmissionColor,
                            "shadeColor" => Vrm1MaterialColorBindType::ShadeColor,
                            "matcapColor" => Vrm1MaterialColorBindType::MatcapColor,
                            "rimColor" => Vrm1MaterialColorBindType::RimColor,
                            "outlineColor" => Vrm1MaterialColorBindType::OutlineColor,
                            value => bail!("{context}.type has unknown value {value:?}"),
                        };
                        let target_value = required_f32_array::<4>(
                            bind.get("targetValue")
                                .with_context(|| format!("{context} is missing targetValue"))?,
                            &format!("{context}.targetValue"),
                        )?;
                        Ok(Vrm1MaterialColorBind {
                            material,
                            kind,
                            target_value,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            };
            let texture_transform_binds = match x.get("textureTransformBinds") {
                None => Vec::new(),
                Some(value) => value
                    .as_array()
                    .with_context(|| {
                        format!("expression {name}.textureTransformBinds must be an array")
                    })?
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        let context = format!("expression {name}.textureTransformBinds[{index}]");
                        let bind = value
                            .as_object()
                            .with_context(|| format!("{context} must be an object"))?;
                        let material = index_value(
                            bind.get("material")
                                .with_context(|| format!("{context} is missing material"))?,
                            &format!("{context}.material"),
                        )?;
                        ensure!(
                            material < material_count,
                            "{context}.material index {material} is out of range for {material_count} materials"
                        );
                        let scale = optional_f32_array::<2>(bind, "scale", &context)?
                            .unwrap_or([1.0, 1.0]);
                        let offset = optional_f32_array::<2>(bind, "offset", &context)?
                            .unwrap_or([0.0, 0.0]);
                        Ok(Vrm1TextureTransformBind {
                            material,
                            scale,
                            offset,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            };
            out.push(Vrm1Expression {
                name: name.clone(),
                kind,
                morph_target_binds,
                material_color_binds,
                texture_transform_binds,
                is_binary: optional_bool(x, "isBinary", name)?.unwrap_or(false),
                override_blink: parse_override(x, "overrideBlink", name)?,
                override_look_at: parse_override(x, "overrideLookAt", name)?,
                override_mouth: parse_override(x, "overrideMouth", name)?,
            })
        }
    }
    Ok(out)
}

fn parse_look_at(vrm: &Map<String, Value>) -> Result<Option<Vrm1LookAt>> {
    let Some(value) = vrm.get("lookAt") else {
        return Ok(None);
    };
    let look_at = value
        .as_object()
        .context("VRMC_vrm.lookAt must be an object")?;
    let kind = match required_string(look_at, "type", "VRMC_vrm.lookAt")?.as_str() {
        "bone" => Vrm1LookAtType::Bone,
        "expression" => Vrm1LookAtType::Expression,
        value => bail!("VRMC_vrm.lookAt.type has unknown value {value:?}"),
    };
    Ok(Some(Vrm1LookAt {
        kind,
        // The specification recommends an implementation fallback when the
        // offset is absent. Model-space zero is deterministic and keeps the
        // origin at the authored head node.
        offset_from_head_bone: optional_vec3(look_at, "offsetFromHeadBone", "VRMC_vrm.lookAt")?
            .unwrap_or(Vec3::ZERO),
        range_map_horizontal_inner: parse_look_at_range_map(look_at, "rangeMapHorizontalInner")?,
        range_map_horizontal_outer: parse_look_at_range_map(look_at, "rangeMapHorizontalOuter")?,
        range_map_vertical_down: parse_look_at_range_map(look_at, "rangeMapVerticalDown")?,
        range_map_vertical_up: parse_look_at_range_map(look_at, "rangeMapVerticalUp")?,
    }))
}

fn parse_look_at_range_map(look_at: &Map<String, Value>, key: &str) -> Result<Vrm1LookAtRangeMap> {
    let context = format!("VRMC_vrm.lookAt.{key}");
    let range = look_at
        .get(key)
        .with_context(|| format!("VRMC_vrm.lookAt is missing required {key}"))?
        .as_object()
        .with_context(|| format!("{context} must be an object"))?;
    let input_max_value = number(
        range
            .get("inputMaxValue")
            .with_context(|| format!("{context} is missing inputMaxValue"))?,
        &format!("{context}.inputMaxValue"),
    )?;
    let output_scale = number(
        range
            .get("outputScale")
            .with_context(|| format!("{context} is missing outputScale"))?,
        &format!("{context}.outputScale"),
    )?;
    ensure!(
        (0.0..=180.0).contains(&input_max_value),
        "{context}.inputMaxValue must be in [0, 180]"
    );
    ensure!(
        output_scale >= 0.0,
        "{context}.outputScale must be non-negative"
    );
    Ok(Vrm1LookAtRangeMap {
        input_max_value,
        output_scale,
    })
}

fn is_preset(s: &str) -> bool {
    matches!(
        s,
        "neutral"
            | "happy"
            | "angry"
            | "sad"
            | "relaxed"
            | "surprised"
            | "aa"
            | "ih"
            | "ou"
            | "ee"
            | "oh"
            | "blink"
            | "blinkLeft"
            | "blinkRight"
            | "lookUp"
            | "lookDown"
            | "lookLeft"
            | "lookRight"
    )
}
fn parse_override(o: &Map<String, Value>, key: &str, name: &str) -> Result<Vrm1ExpressionOverride> {
    match o.get(key) {
        None => Ok(Vrm1ExpressionOverride::None),
        Some(v) => match v
            .as_str()
            .with_context(|| format!("expression {name}.{key} must be one of none, block, blend"))?
        {
            "none" => Ok(Vrm1ExpressionOverride::None),
            "block" => Ok(Vrm1ExpressionOverride::Block),
            "blend" => Ok(Vrm1ExpressionOverride::Blend),
            v => bail!("expression {name}.{key} must be one of none, block, blend, got {v:?}"),
        },
    }
}
fn parse_meta(v: &Map<String, Value>) -> Result<Vrm1Meta> {
    let m = v
        .get("meta")
        .context("VRMC_vrm is missing required meta section")?
        .as_object()
        .context("VRMC_vrm.meta must be an object")?;
    let a = m
        .get("authors")
        .context("VRMC_vrm.meta is missing required authors")?
        .as_array()
        .context("VRMC_vrm.meta.authors must be an array")?;
    ensure!(
        !a.is_empty(),
        "VRMC_vrm.meta.authors must contain at least one author"
    );
    let authors = a
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let s = v
                .as_str()
                .with_context(|| format!("VRMC_vrm.meta.authors[{i}] must be a string"))?;
            ensure!(
                !s.trim().is_empty(),
                "VRMC_vrm.meta.authors[{i}] must not be empty"
            );
            Ok(s.to_owned())
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Vrm1Meta {
        name: required_string(m, "name", "VRMC_vrm.meta")?,
        version: optional_string(m, "version", "VRMC_vrm.meta")?,
        authors,
        license_url: required_string(m, "licenseUrl", "VRMC_vrm.meta")?,
    })
}
fn parse_humanoid(v: &Map<String, Value>, n: usize) -> Result<Vrm1Humanoid> {
    let h = v
        .get("humanoid")
        .context("VRMC_vrm is missing required humanoid section")?
        .get("humanBones")
        .context("VRMC_vrm.humanoid is missing required humanBones")?
        .as_object()
        .context("VRMC_vrm.humanoid.humanBones must be an object")?;
    let mut out = BTreeMap::new();
    let mut used = HashSet::new();
    for (name, v) in h {
        let b = Vrm1HumanBone::parse(name)
            .with_context(|| format!("unknown VRM 1.0 human bone {name}"))?;
        let o = v
            .as_object()
            .with_context(|| format!("human bone {name} must be an object"))?;
        let node = index_value(
            o.get("node")
                .with_context(|| format!("human bone {name} is missing node"))?,
            &format!("human bone {name}.node"),
        )?;
        ensure!(
            node < n,
            "human bone {name}.node {node} is out of range for {n} nodes"
        );
        ensure!(
            used.insert(node),
            "multiple human bones map to glTF node {node}"
        );
        ensure!(out.insert(b, node).is_none(), "duplicate human bone {name}")
    }
    for b in Vrm1HumanBone::REQUIRED {
        ensure!(
            out.contains_key(b),
            "VRMC_vrm.humanoid is missing required human bone {}",
            b.as_str()
        )
    }
    Ok(Vrm1Humanoid { human_bones: out })
}
fn extension_info(e: &Map<String, Value>, name: &str) -> Result<Vrm1ExtensionInfo> {
    match e.get(name) {
        None => Ok(Vrm1ExtensionInfo::default()),
        Some(v) => required_extension_info(v, name),
    }
}
fn required_extension_info(v: &Value, name: &str) -> Result<Vrm1ExtensionInfo> {
    let o = v
        .as_object()
        .with_context(|| format!("{name} extension must be an object"))?;
    let version = required_string(o, "specVersion", name)?;
    ensure!(
        version == "1.0",
        "{name}.specVersion must be \"1.0\", got \"{version}\""
    );
    Ok(Vrm1ExtensionInfo {
        present: true,
        version: Some(version),
    })
}
fn parse_node_constraints(
    root: &Value,
    node_count: usize,
) -> Result<(Vrm1ExtensionInfo, Option<Vrm1NodeConstraintSet>)> {
    const NAME: &str = "VRMC_node_constraint";
    let extensions_used = extension_declaration_contains(root, "extensionsUsed", NAME)?;
    let extensions_required = extension_declaration_contains(root, "extensionsRequired", NAME)?;
    ensure!(
        !extensions_required || extensions_used,
        "glTF extensionsRequired contains {NAME}, but extensionsUsed does not"
    );

    let nodes = root
        .get("nodes")
        .and_then(Value::as_array)
        .context("GLB is missing required glTF nodes array")?;
    let mut info: Option<Vrm1ExtensionInfo> = None;
    let mut constraints = Vec::new();

    for (destination, node) in nodes.iter().enumerate() {
        let node = node
            .as_object()
            .with_context(|| format!("glTF node {destination} must be an object"))?;
        let Some(extensions) = node.get("extensions") else {
            continue;
        };
        let extensions = extensions
            .as_object()
            .with_context(|| format!("glTF node {destination}.extensions must be an object"))?;
        let Some(value) = extensions.get(NAME) else {
            continue;
        };

        let extension_context = format!("glTF node {destination}.extensions.{NAME}");
        let extension_info = required_extension_info(value, &extension_context)?;
        if let Some(previous) = &info {
            ensure!(
                previous.version == extension_info.version,
                "{NAME} has conflicting specVersion values"
            );
        }
        info = Some(extension_info);

        let extension = value
            .as_object()
            .with_context(|| format!("{extension_context} extension must be an object"))?;
        let constraint_context = format!("{extension_context}.constraint");
        let constraint = extension
            .get("constraint")
            .with_context(|| format!("{extension_context} is missing constraint"))?
            .as_object()
            .with_context(|| format!("{constraint_context} must be an object"))?;
        let kind = parse_node_constraint_kind(constraint, &constraint_context)?;
        let source = kind.source();
        ensure!(
            source < node_count,
            "{extension_context} source {source} is out of range for {node_count} nodes"
        );
        ensure!(
            source != destination,
            "{extension_context} source must not equal destination node {destination}"
        );
        constraints.push(Vrm1NodeConstraint { destination, kind });
    }

    let Some(info) = info else {
        ensure!(
            !extensions_required,
            "glTF extensionsRequired contains {NAME}, but no node declares the extension"
        );
        return Ok((Vrm1ExtensionInfo::default(), None));
    };
    ensure!(
        extensions_used,
        "{NAME} is present on a node but is missing from glTF extensionsUsed"
    );
    Ok((info, Some(Vrm1NodeConstraintSet { constraints })))
}

fn parse_node_constraint_kind(
    extension: &Map<String, Value>,
    context: &str,
) -> Result<Vrm1NodeConstraintKind> {
    let variant_count = ["roll", "aim", "rotation"]
        .into_iter()
        .filter(|key| extension.contains_key(*key))
        .count();
    ensure!(
        variant_count == 1,
        "{context} must contain exactly one of roll, aim, or rotation; found {variant_count}"
    );

    if let Some(value) = extension.get("roll") {
        let roll = value
            .as_object()
            .with_context(|| format!("{context}.roll must be an object"))?;
        let source = constraint_source(roll, &format!("{context}.roll"))?;
        let axis = parse_roll_axis(roll, &format!("{context}.roll"))?;
        let weight = constraint_weight(roll, &format!("{context}.roll"))?;
        return Ok(Vrm1NodeConstraintKind::Roll {
            source,
            axis,
            weight,
        });
    }

    if let Some(value) = extension.get("aim") {
        let aim = value
            .as_object()
            .with_context(|| format!("{context}.aim must be an object"))?;
        let source = constraint_source(aim, &format!("{context}.aim"))?;
        let axis = parse_aim_axis(aim, &format!("{context}.aim"))?;
        let weight = constraint_weight(aim, &format!("{context}.aim"))?;
        return Ok(Vrm1NodeConstraintKind::Aim {
            source,
            axis,
            weight,
        });
    }

    let rotation = extension
        .get("rotation")
        .expect("exactly one node constraint variant was counted");
    let rotation = rotation
        .as_object()
        .with_context(|| format!("{context}.rotation must be an object"))?;
    Ok(Vrm1NodeConstraintKind::Rotation {
        source: constraint_source(rotation, &format!("{context}.rotation"))?,
        weight: constraint_weight(rotation, &format!("{context}.rotation"))?,
    })
}

fn constraint_source(extension: &Map<String, Value>, context: &str) -> Result<usize> {
    index_value(
        extension
            .get("source")
            .with_context(|| format!("{context} is missing source"))?,
        &format!("{context}.source"),
    )
}

fn constraint_weight(extension: &Map<String, Value>, context: &str) -> Result<f32> {
    let weight = extension
        .get("weight")
        .map(|value| number(value, &format!("{context}.weight")))
        .transpose()?
        .unwrap_or(1.0);
    bounded(weight, 0.0, 1.0, &format!("{context}.weight"))
}

fn parse_roll_axis(extension: &Map<String, Value>, context: &str) -> Result<Vrm1RollAxis> {
    let axis = required_string(extension, "rollAxis", context)?;
    match axis.as_str() {
        "X" => Ok(Vrm1RollAxis::X),
        "Y" => Ok(Vrm1RollAxis::Y),
        "Z" => Ok(Vrm1RollAxis::Z),
        _ => bail!("{context}.rollAxis must be one of X, Y, or Z, got {axis:?}"),
    }
}

fn parse_aim_axis(extension: &Map<String, Value>, context: &str) -> Result<Vrm1AimAxis> {
    let axis = required_string(extension, "aimAxis", context)?;
    match axis.as_str() {
        "PositiveX" => Ok(Vrm1AimAxis::PositiveX),
        "NegativeX" => Ok(Vrm1AimAxis::NegativeX),
        "PositiveY" => Ok(Vrm1AimAxis::PositiveY),
        "NegativeY" => Ok(Vrm1AimAxis::NegativeY),
        "PositiveZ" => Ok(Vrm1AimAxis::PositiveZ),
        "NegativeZ" => Ok(Vrm1AimAxis::NegativeZ),
        _ => bail!(
            "{context}.aimAxis must be one of PositiveX, NegativeX, PositiveY, NegativeY, PositiveZ, or NegativeZ, got {axis:?}"
        ),
    }
}

fn extension_declaration_contains(root: &Value, key: &str, name: &str) -> Result<bool> {
    let Some(value) = root.get(key) else {
        return Ok(false);
    };
    let declarations = value
        .as_array()
        .with_context(|| format!("glTF {key} must be an array"))?;
    let mut found = false;
    for (index, value) in declarations.iter().enumerate() {
        let declaration = value
            .as_str()
            .with_context(|| format!("glTF {key}[{index}] must be an extension name string"))?;
        found |= declaration == name;
    }
    Ok(found)
}

fn optional_object(v: &Map<String, Value>, name: &str) -> Result<bool> {
    let Some(x) = v.get(name) else {
        return Ok(false);
    };
    ensure!(
        x.is_object(),
        "VRMC_vrm.{name} must be an object when present"
    );
    Ok(true)
}
fn validate_nodes(
    root: &Value,
) -> Result<(
    Vec<Vec<usize>>,
    Vec<Option<usize>>,
    Vec<usize>,
    Vec<Option<usize>>,
)> {
    let a = root
        .get("nodes")
        .context("GLB is missing required glTF nodes array")?
        .as_array()
        .context("glTF nodes must be an array")?;
    let n = a.len();
    let meshes = match root.get("meshes") {
        None => Vec::new(),
        Some(v) => v
            .as_array()
            .context("glTF meshes must be an array")?
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_object()
                    .with_context(|| format!("glTF mesh {i} must be an object"))?
                    .get("primitives")
                    .with_context(|| format!("glTF mesh {i} is missing primitives"))?
                    .as_array()
                    .with_context(|| format!("glTF mesh {i}.primitives must be an array"))
                    .map(Vec::len)
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let mesh_count = root.get("meshes").map(|_| meshes.len());
    let mut c = vec![Vec::new(); n];
    let mut nm = vec![None; n];
    let mut p = vec![None; n];
    for (i, v) in a.iter().enumerate() {
        let o = v
            .as_object()
            .with_context(|| format!("glTF node {i} must be an object"))?;
        if let Some(v) = o.get("mesh") {
            let m = index_value(v, &format!("glTF node {i}.mesh"))?;
            ensure!(
                Some(m) < mesh_count,
                "glTF node {i}.mesh {m} is out of range for meshes"
            );
            nm[i] = Some(m)
        }
        if let Some(v) = o.get("children") {
            let a = v
                .as_array()
                .with_context(|| format!("glTF node {i}.children must be an array"))?;
            let mut seen = HashSet::new();
            for v in a {
                let x = index_value(v, &format!("glTF node {i}.children"))?;
                ensure!(
                    x < n,
                    "glTF node {i}.children references node {x}, but node count is {n}"
                );
                ensure!(
                    seen.insert(x),
                    "glTF node {i}.children contains duplicate node {x}"
                );
                ensure!(
                    p[x].replace(i).is_none(),
                    "glTF node {x} has more than one parent"
                );
                c[i].push(x)
            }
        }
    }
    let mut state = vec![0u8; n];
    for start in 0..n {
        if state[start] != 0 {
            continue;
        }
        state[start] = 1;
        let mut stack = vec![(start, 0)];
        while let Some((node, ci)) = stack.last_mut() {
            if *ci == c[*node].len() {
                state[*node] = 2;
                stack.pop();
                continue;
            }
            let x = c[*node][*ci];
            *ci += 1;
            match state[x] {
                0 => {
                    state[x] = 1;
                    stack.push((x, 0))
                }
                1 => bail!("glTF node hierarchy contains a cycle through node {x}"),
                2 => {}
                _ => unreachable!(),
            }
        }
    }
    Ok((c, nm, meshes, p))
}
fn required_string(o: &Map<String, Value>, key: &str, context: &str) -> Result<String> {
    let v = o
        .get(key)
        .with_context(|| format!("{context} is missing required {key}"))?
        .as_str()
        .with_context(|| format!("{context}.{key} must be a string"))?;
    ensure!(!v.trim().is_empty(), "{context}.{key} must not be empty");
    Ok(v.to_owned())
}
fn optional_string(o: &Map<String, Value>, key: &str, context: &str) -> Result<Option<String>> {
    match o.get(key) {
        None => Ok(None),
        Some(v) => Ok(Some(
            v.as_str()
                .with_context(|| format!("{context}.{key} must be a string"))?
                .to_owned(),
        )),
    }
}
fn index_value(v: &Value, context: &str) -> Result<usize> {
    let n = v
        .as_u64()
        .with_context(|| format!("{context} must be an integer"))?;
    usize::try_from(n).with_context(|| format!("{context} does not fit in usize"))
}
fn optional_index(o: &Map<String, Value>, key: &str, context: &str) -> Result<Option<usize>> {
    o.get(key)
        .map(|v| index_value(v, &format!("{context}.{key}")))
        .transpose()
}
fn number(v: &Value, context: &str) -> Result<f32> {
    let n = v
        .as_f64()
        .with_context(|| format!("{context} must be a number"))? as f32;
    ensure!(
        n.is_finite(),
        "{context} must remain finite after f32 conversion"
    );
    Ok(n)
}
fn required_f32_array<const N: usize>(v: &Value, context: &str) -> Result<[f32; N]> {
    let values = v
        .as_array()
        .with_context(|| format!("{context} must be an array"))?;
    ensure!(
        values.len() == N,
        "{context} must contain exactly {N} numbers"
    );
    let mut result = [0.0; N];
    for (index, value) in values.iter().enumerate() {
        result[index] = number(value, &format!("{context}[{index}]"))?;
    }
    Ok(result)
}
fn optional_f32_array<const N: usize>(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<Option<[f32; N]>> {
    object
        .get(key)
        .map(|value| required_f32_array(value, &format!("{context}.{key}")))
        .transpose()
}
fn required_f32(v: &Value, context: &str) -> Result<f32> {
    v.as_f64()
        .with_context(|| format!("{context} must be a number"))
        .map(|n| n as f32)
}
fn optional_f32(o: &Map<String, Value>, key: &str, context: &str) -> Result<Option<f32>> {
    o.get(key)
        .map(|v| number(v, &format!("{context}.{key}")))
        .transpose()
}
fn optional_bool(o: &Map<String, Value>, key: &str, context: &str) -> Result<Option<bool>> {
    match o.get(key) {
        None => Ok(None),
        Some(v) => Ok(Some(v.as_bool().with_context(|| {
            format!("expression {context}.{key} must be a boolean")
        })?)),
    }
}
fn optional_vec3(o: &Map<String, Value>, key: &str, context: &str) -> Result<Option<Vec3>> {
    let Some(v) = o.get(key) else { return Ok(None) };
    let a = v
        .as_array()
        .with_context(|| format!("{context}.{key} must be an array"))?;
    ensure!(
        a.len() == 3,
        "{context}.{key} must contain exactly three numbers"
    );
    Ok(Some(Vec3::new(
        number(&a[0], &format!("{context}.{key}[0]"))?,
        number(&a[1], &format!("{context}.{key}[1]"))?,
        number(&a[2], &format!("{context}.{key}[2]"))?,
    )))
}
fn nonnegative(v: f32, c: &str) -> Result<f32> {
    ensure!(v >= 0.0, "{c} must be non-negative");
    Ok(v)
}
fn bounded(v: f32, lo: f32, hi: f32, c: &str) -> Result<f32> {
    ensure!((lo..=hi).contains(&v), "{c} must be in [{lo}, {hi}]");
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn glb(value: Value) -> Vec<u8> {
        let mut json_bytes = serde_json::to_vec(&value).unwrap();
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }
        let total = 12 + 8 + json_bytes.len();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"glTF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&(total as u32).to_le_bytes());
        bytes.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
        bytes.extend_from_slice(&json_bytes);
        bytes
    }

    fn base() -> Value {
        let mut bones = Map::new();
        for (i, bone) in Vrm1HumanBone::REQUIRED.iter().enumerate() {
            bones.insert(bone.as_str().to_owned(), json!({"node": i}));
        }
        let nodes = (0..Vrm1HumanBone::REQUIRED.len())
            .map(|i| {
                if i + 1 < Vrm1HumanBone::REQUIRED.len() {
                    json!({"children":[i+1]})
                } else {
                    json!({})
                }
            })
            .collect::<Vec<_>>();
        json!({"asset":{"version":"2.0"},"nodes":nodes,"extensions":{"VRMC_vrm":{"specVersion":"1.0","meta":{"name":"test","authors":["test"],"licenseUrl":"test"},"humanoid":{"humanBones":bones}}}})
    }

    fn valid_document() -> Value {
        let mut value = base();
        value["materials"] = json!([
            {"extensions":{"VRMC_materials_mtoon":{"specVersion":"1.0"}}}
        ]);
        value["nodes"][0]["extensions"] = json!({"VRMC_node_constraint":{
            "specVersion":"1.0",
            "constraint":{"rotation":{"source":1}}
        }});
        value["extensionsUsed"] = json!(["VRMC_materials_mtoon", "VRMC_node_constraint"]);
        value["extensions"]["VRMC_vrm"]["meta"]["version"] = json!("1.0");
        value["extensions"]["VRMC_vrm"]["expressions"] = json!({});
        value["extensions"]["VRMC_vrm"]["lookAt"] = look_at_json("bone");
        value["extensions"]["VRMC_vrm"]["firstPerson"] = json!({});
        value["extensions"]["VRMC_springBone"] = json!({"specVersion":"1.0"});
        value["extensions"]["X_unknown_optional"] = json!({"anything":true});
        value
    }

    fn look_at_json(kind: &str) -> Value {
        json!({
            "type": kind,
            "offsetFromHeadBone": [0.1, 0.2, 0.3],
            "rangeMapHorizontalInner": { "inputMaxValue": 10.0, "outputScale": 1.0 },
            "rangeMapHorizontalOuter": { "inputMaxValue": 20.0, "outputScale": 2.0 },
            "rangeMapVerticalDown": { "inputMaxValue": 30.0, "outputScale": 3.0 },
            "rangeMapVerticalUp": { "inputMaxValue": 40.0, "outputScale": 4.0 }
        })
    }

    fn parse(value: Value) -> Result<Vrm1Doc> {
        Vrm1Doc::from_glb_bytes(&glb(value))
    }

    fn node_constraint_document(extension: Value) -> Value {
        let mut value = base();
        value["extensionsUsed"] = json!(["VRMC_node_constraint"]);
        value["extensionsRequired"] = json!(["VRMC_node_constraint"]);
        let mut extension = extension
            .as_object()
            .expect("test constraint extension must be an object")
            .clone();
        let spec_version = extension.remove("specVersion");
        let mut node_extension = serde_json::Map::new();
        if let Some(spec_version) = spec_version {
            node_extension.insert("specVersion".to_owned(), spec_version);
        }
        node_extension.insert("constraint".to_owned(), Value::Object(extension));
        value["nodes"][0]["extensions"] = json!({
            "VRMC_node_constraint": Value::Object(node_extension)
        });
        value
    }

    #[test]
    fn node_constraint_absence_has_no_typed_semantics() {
        let doc = parse(base()).unwrap();
        assert!(!doc.node_constraint.present);
        assert_eq!(doc.node_constraint_semantics, None);
    }

    #[test]
    fn valid_node_constraint_variants_are_typed() {
        let roll = parse(node_constraint_document(json!({
            "specVersion": "1.0",
            "roll": {"source": 1, "rollAxis": "Y", "weight": 0.25}
        })))
        .unwrap();
        assert_eq!(
            roll.node_constraint_semantics
                .as_ref()
                .unwrap()
                .constraints
                .len(),
            1
        );
        assert_eq!(
            roll.node_constraint_semantics.as_ref().unwrap().constraints[0],
            Vrm1NodeConstraint {
                destination: 0,
                kind: Vrm1NodeConstraintKind::Roll {
                    source: 1,
                    axis: Vrm1RollAxis::Y,
                    weight: 0.25,
                },
            }
        );

        let aim = parse(node_constraint_document(json!({
            "specVersion": "1.0",
            "aim": {"source": 1, "aimAxis": "NegativeZ", "weight": 0.5}
        })))
        .unwrap();
        assert!(matches!(
            aim.node_constraint_semantics.as_ref().unwrap().constraints[0].kind,
            Vrm1NodeConstraintKind::Aim {
                source: 1,
                axis: Vrm1AimAxis::NegativeZ,
                weight
            } if weight == 0.5
        ));

        let rotation = parse(node_constraint_document(json!({
            "specVersion": "1.0",
            "rotation": {"source": 1, "weight": 0.75}
        })))
        .unwrap();
        assert!(matches!(
            rotation
                .node_constraint_semantics
                .as_ref()
                .unwrap()
                .constraints[0]
                .kind,
            Vrm1NodeConstraintKind::Rotation {
                source: 1,
                weight
            } if weight == 0.75
        ));
    }

    #[test]
    fn node_constraint_weight_defaults_to_one_for_every_variant() {
        for variant in [
            json!({"roll": {"source": 1, "rollAxis": "X"}}),
            json!({"aim": {"source": 1, "aimAxis": "PositiveX"}}),
            json!({"rotation": {"source": 1}}),
        ] {
            let mut extension = variant;
            extension["specVersion"] = json!("1.0");
            let doc = parse(node_constraint_document(extension)).unwrap();
            let kind = doc.node_constraint_semantics.unwrap().constraints[0].kind;
            let weight = match kind {
                Vrm1NodeConstraintKind::Roll { weight, .. }
                | Vrm1NodeConstraintKind::Aim { weight, .. }
                | Vrm1NodeConstraintKind::Rotation { weight, .. } => weight,
            };
            assert_eq!(weight, 1.0);
        }
    }

    #[test]
    fn every_node_constraint_axis_is_typed() {
        for (axis, expected) in [
            ("X", Vrm1RollAxis::X),
            ("Y", Vrm1RollAxis::Y),
            ("Z", Vrm1RollAxis::Z),
        ] {
            let doc = parse(node_constraint_document(json!({
                "specVersion": "1.0",
                "roll": {"source": 1, "rollAxis": axis}
            })))
            .unwrap();
            assert!(matches!(
                doc.node_constraint_semantics.unwrap().constraints[0].kind,
                Vrm1NodeConstraintKind::Roll { axis, .. } if axis == expected
            ));
        }

        for (axis, expected) in [
            ("PositiveX", Vrm1AimAxis::PositiveX),
            ("NegativeX", Vrm1AimAxis::NegativeX),
            ("PositiveY", Vrm1AimAxis::PositiveY),
            ("NegativeY", Vrm1AimAxis::NegativeY),
            ("PositiveZ", Vrm1AimAxis::PositiveZ),
            ("NegativeZ", Vrm1AimAxis::NegativeZ),
        ] {
            let doc = parse(node_constraint_document(json!({
                "specVersion": "1.0",
                "aim": {"source": 1, "aimAxis": axis}
            })))
            .unwrap();
            assert!(matches!(
                doc.node_constraint_semantics.unwrap().constraints[0].kind,
                Vrm1NodeConstraintKind::Aim { axis, .. } if axis == expected
            ));
        }
    }

    #[test]
    fn node_constraint_requires_exactly_one_variant() {
        let zero = node_constraint_document(json!({"specVersion": "1.0"}));
        let error = parse(zero).unwrap_err().to_string();
        assert!(
            error.contains("exactly one") && error.contains("found 0"),
            "{error}"
        );

        let multiple = node_constraint_document(json!({
            "specVersion": "1.0",
            "roll": {"source": 1, "rollAxis": "X"},
            "rotation": {"source": 1}
        }));
        let error = parse(multiple).unwrap_err().to_string();
        assert!(
            error.contains("exactly one") && error.contains("found 2"),
            "{error}"
        );
    }

    #[test]
    fn node_constraint_requires_the_normative_constraint_object() {
        let mut value = base();
        value["extensionsUsed"] = json!(["VRMC_node_constraint"]);
        value["extensionsRequired"] = json!(["VRMC_node_constraint"]);
        value["nodes"][0]["extensions"] = json!({
            "VRMC_node_constraint": {
                "specVersion": "1.0",
                "rotation": {"source": 1}
            }
        });
        let error = parse(value).unwrap_err().to_string();
        assert!(error.contains("missing constraint"), "{error}");
    }

    #[test]
    fn node_constraint_indices_and_axes_are_validated() {
        let missing_source = node_constraint_document(json!({
            "specVersion": "1.0",
            "roll": {"rollAxis": "X"}
        }));
        let error = parse(missing_source).unwrap_err().to_string();
        assert!(error.contains("missing source"), "{error}");

        for source in [json!(999), json!(-1), json!("one")] {
            let value = node_constraint_document(json!({
                "specVersion": "1.0",
                "rotation": {"source": source}
            }));
            assert!(parse(value).is_err(), "invalid source {source}");
        }

        let self_reference = node_constraint_document(json!({
            "specVersion": "1.0",
            "rotation": {"source": 0}
        }));
        let error = parse(self_reference).unwrap_err().to_string();
        assert!(error.contains("must not equal destination"), "{error}");

        for (variant, axis_key) in [("roll", "rollAxis"), ("aim", "aimAxis")] {
            let mut body = serde_json::Map::new();
            body.insert("source".to_owned(), json!(1));
            body.insert(axis_key.to_owned(), json!("NotAnAxis"));
            let mut extension = serde_json::Map::new();
            extension.insert("specVersion".to_owned(), json!("1.0"));
            extension.insert(variant.to_owned(), Value::Object(body));
            let error = parse(node_constraint_document(Value::Object(extension)))
                .unwrap_err()
                .to_string();
            assert!(error.contains("Axis") || error.contains("axis"), "{error}");
        }
    }

    #[test]
    fn node_constraint_weights_are_finite_and_bounded() {
        for weight in [json!(-0.01), json!(1.01), json!(3.5e38_f64), json!("one")] {
            let value = node_constraint_document(json!({
                "specVersion": "1.0",
                "rotation": {"source": 1, "weight": weight}
            }));
            assert!(parse(value).is_err(), "invalid weight {weight}");
        }
    }

    #[test]
    fn node_constraint_spec_version_and_declarations_are_consistent() {
        for spec_version in [Some(json!("0.9")), None] {
            let mut extension = json!({"rotation": {"source": 1}});
            if let Some(spec_version) = spec_version {
                extension["specVersion"] = spec_version;
            }
            let error = parse(node_constraint_document(extension))
                .unwrap_err()
                .to_string();
            assert!(error.contains("VRMC_node_constraint") && error.contains("specVersion"));
        }

        let mut undeclared = node_constraint_document(json!({
            "specVersion": "1.0",
            "rotation": {"source": 1}
        }));
        undeclared.as_object_mut().unwrap().remove("extensionsUsed");
        let error = parse(undeclared).unwrap_err().to_string();
        assert!(
            error.contains("extensionsRequired") && error.contains("extensionsUsed"),
            "{error}"
        );

        let mut required_without_used = base();
        required_without_used["extensionsRequired"] = json!(["VRMC_node_constraint"]);
        let error = parse(required_without_used).unwrap_err().to_string();
        assert!(error.contains("extensionsUsed"), "{error}");
    }

    #[test]
    fn malformed_optional_node_constraint_disables_the_entire_extension() {
        let mut value = node_constraint_document(json!({
            "specVersion": "1.0",
            "rotation": {}
        }));
        value.as_object_mut().unwrap().remove("extensionsRequired");
        let document = parse(value).expect("malformed optional constraint must not reject VRM");
        assert!(!document.node_constraint.present);
        assert!(document.node_constraint_semantics.is_none());
    }

    #[test]
    fn required_node_constraint_without_a_payload_is_rejected() {
        let mut value = base();
        value["extensionsUsed"] = json!(["VRMC_node_constraint"]);
        value["extensionsRequired"] = json!(["VRMC_node_constraint"]);
        let error = parse(value).unwrap_err().to_string();
        assert!(error.contains("no node declares"), "{error}");
    }

    #[test]
    fn spring_defaults_and_shape_order_are_typed() {
        assert!(parse(base()).unwrap().spring_bone_semantics.is_none());
        let mut value = base();
        value["extensions"]["VRMC_springBone"] = json!({
            "specVersion":"1.0",
            "colliders":[
                {"node":0,"shape":{"sphere":{},"extras":{"tag":true}}},
                {"node":0,"shape":{"capsule":{"tail":[1,2,3],"extensions":{}}}}
            ],
            "colliderGroups":[{"name":"body","colliders":[0,1,0]}],
            "springs":[{"center":0,"colliderGroups":[0],"joints":[{"node":1},{"node":3}]}]
        });
        let doc = parse(value).unwrap();
        let spring = doc.spring_bone_semantics.unwrap();
        assert_eq!(spring.collider_groups[0].colliders, [0, 1, 0]);
        assert_eq!(
            spring.colliders[0],
            Vrm1Collider {
                node: 0,
                shape: Vrm1ColliderShape::Sphere(Vrm1SphereCollider {
                    offset: Vec3::ZERO,
                    radius: 0.0
                })
            }
        );
        assert_eq!(spring.springs[0].joints[0].hit_radius, 0.0);
        assert_eq!(spring.springs[0].joints[0].stiffness, 1.0);
        assert_eq!(spring.springs[0].joints[0].gravity_power, 0.0);
        assert_eq!(
            spring.springs[0].joints[0].gravity_dir,
            Vec3::new(0.0, -1.0, 0.0)
        );
        assert_eq!(spring.springs[0].joints[0].drag_force, 0.5);
    }

    #[test]
    fn malformed_spring_shapes_and_values_fail_without_member_dropping() {
        let mut value = base();
        value["extensions"]["VRMC_springBone"] = json!({"specVersion":"1.0","colliders":[{"node":0,"shape":{"sphere":{},"capsule":{}}}]});
        assert!(parse(value).is_err());
        let mut value = base();
        value["extensions"]["VRMC_springBone"] =
            json!({"specVersion":"1.0","springs":[{"joints":[{"node":1,"gravityDir":[0,0]}]}]});
        assert!(parse(value).is_err());
        let mut value = base();
        value["extensions"]["VRMC_springBone"] =
            json!({"specVersion":"1.0","springs":[{"joints":[{"node":1,"dragForce":1.1}]}]});
        assert!(parse(value).is_err());
        let mut value = base();
        value["extensions"]["VRMC_springBone"] = json!({"specVersion":"1.0","colliders":[{"node":0,"shape":{"sphere":{"radius":3.5e38}}}]});
        assert!(parse(value).is_err());
        for field in ["colliders", "colliderGroups", "springs"] {
            let mut value = base();
            value["extensions"]["VRMC_springBone"] = json!({"specVersion":"1.0",field:[]});
            assert!(parse(value).is_err(), "empty root array {field} must fail");
        }
        let mut value = base();
        value["extensions"]["VRMC_springBone"] =
            json!({"specVersion":"1.0","springs":[{"joints":[{"node":1}],"colliderGroups":[]}]});
        assert!(parse(value).is_err());
    }

    #[test]
    fn spring_references_ancestry_and_joint_ownership_are_strict() {
        let mut value = base();
        value["extensions"]["VRMC_springBone"] =
            json!({"specVersion":"1.0","springs":[{"joints":[{"node":1},{"node":3}]}]});
        assert_eq!(
            parse(value).unwrap().spring_bone_semantics.unwrap().springs[0]
                .joints
                .len(),
            2
        );
        let mut value = base();
        value["extensions"]["VRMC_springBone"] = json!({"specVersion":"1.0","springs":[{"joints":[{"node":1},{"node":3}]},{"joints":[{"node":2},{"node":4}]}]});
        assert!(parse(value).is_err());
        let mut value = base();
        value["extensions"]["VRMC_springBone"] = json!({"specVersion":"1.0","springs":[{"joints":[{"node":1},{"node":3}]},{"joints":[{"node":1}]}]});
        assert!(parse(value).is_err());
        let mut value = base();
        value["extensions"]["VRMC_springBone"] =
            json!({"specVersion":"1.0","springs":[{"joints":[{"node":3},{"node":1}]}]});
        assert!(parse(value).is_err());
    }

    #[test]
    fn one_joint_spring_is_valid_noop_data() {
        let mut value = base();
        value["extensions"]["VRMC_springBone"] =
            json!({"specVersion":"1.0","springs":[{"joints":[{"node":1}]}]});
        assert_eq!(
            parse(value).unwrap().spring_bone_semantics.unwrap().springs[0]
                .joints
                .len(),
            1
        );
    }

    #[test]
    fn valid_minimal_vrm1_retains_semantic_facts() {
        let doc = parse(valid_document()).unwrap();
        assert_eq!(doc.meta.name, "test");
        assert_eq!(doc.meta.version.as_deref(), Some("1.0"));
        assert_eq!(doc.meta.authors, ["test"]);
        assert_eq!(doc.humanoid.node_for(Vrm1HumanBone::Hips), Some(0));
        assert_eq!(doc.node_count, Vrm1HumanBone::REQUIRED.len());
        assert!(doc.has_expressions);
        let look_at = doc.look_at.as_ref().unwrap();
        assert_eq!(look_at.kind, Vrm1LookAtType::Bone);
        assert_eq!(look_at.offset_from_head_bone, Vec3::new(0.1, 0.2, 0.3));
        assert_eq!(look_at.range_map_horizontal_inner.input_max_value, 10.0);
        assert_eq!(look_at.range_map_horizontal_outer.output_scale, 2.0);
        assert_eq!(look_at.range_map_vertical_down.input_max_value, 30.0);
        assert_eq!(look_at.range_map_vertical_up.output_scale, 4.0);
        assert!(doc.has_first_person);
        assert!(doc.materials_mtoon.present);
        assert_eq!(doc.materials_mtoon.version.as_deref(), Some("1.0"));
        assert!(doc.spring_bone.present);
        assert_eq!(doc.spring_bone.version.as_deref(), Some("1.0"));
        assert!(doc.node_constraint.present);
        assert_eq!(doc.node_constraint.version.as_deref(), Some("1.0"));
        assert_eq!(doc.node_meshes.len(), doc.node_count);
        assert!(doc.node_meshes.iter().all(Option::is_none));
    }

    #[test]
    fn look_at_absence_and_valid_expression_are_typed() {
        assert!(parse(base()).unwrap().look_at.is_none());
        let mut value = base();
        value["extensions"]["VRMC_vrm"]["lookAt"] = look_at_json("expression");
        let look_at = parse(value).unwrap().look_at.unwrap();
        assert_eq!(look_at.kind, Vrm1LookAtType::Expression);
        assert_eq!(look_at.offset_from_head_bone, Vec3::new(0.1, 0.2, 0.3));
    }

    #[test]
    fn look_at_without_offset_uses_head_origin() {
        let mut value = base();
        let mut look_at = look_at_json("bone");
        look_at
            .as_object_mut()
            .unwrap()
            .remove("offsetFromHeadBone");
        value["extensions"]["VRMC_vrm"]["lookAt"] = look_at;
        assert_eq!(
            parse(value).unwrap().look_at.unwrap().offset_from_head_bone,
            Vec3::ZERO
        );
    }

    #[test]
    fn malformed_or_partial_look_at_is_disabled_without_rejecting_avatar() {
        let mut malformed_enum = base();
        malformed_enum["extensions"]["VRMC_vrm"]["lookAt"] = look_at_json("bones");
        assert!(parse(malformed_enum).unwrap().look_at.is_none());

        let mut partial = base();
        let mut look_at = look_at_json("bone");
        look_at
            .as_object_mut()
            .unwrap()
            .remove("rangeMapVerticalUp");
        partial["extensions"]["VRMC_vrm"]["lookAt"] = look_at;
        assert!(parse(partial).unwrap().look_at.is_none());

        let mut overflow = base();
        let mut look_at = look_at_json("bone");
        look_at["rangeMapHorizontalInner"]["outputScale"] = json!(3.5e38_f64);
        overflow["extensions"]["VRMC_vrm"]["lookAt"] = look_at;
        assert!(parse(overflow).unwrap().look_at.is_none());

        let mut wrong_numeric_type = base();
        let mut look_at = look_at_json("bone");
        look_at["offsetFromHeadBone"] = json!([0.0, "bad", 0.0]);
        wrong_numeric_type["extensions"]["VRMC_vrm"]["lookAt"] = look_at;
        assert!(parse(wrong_numeric_type).unwrap().look_at.is_none());
    }

    #[test]
    fn expressions_retain_all_typed_bind_families_and_defaults() {
        let mut value = valid_document();
        value["extensions"]["VRMC_vrm"]["expressions"] = json!({
            "preset": {
                "blink": {
                    "morphTargetBinds":[{"node":2,"index":3,"weight":0.75}],
                    "isBinary":true,
                    "overrideBlink":"block",
                    "overrideLookAt":"blend",
                    "overrideMouth":"none",
                    "materialColorBinds":[
                        {"material":0,"type":"color","targetValue":[0.1,0.2,0.3,0.4]},
                        {"material":0,"type":"emissionColor","targetValue":[1.0,2.0,3.0,4.0]},
                        {"material":0,"type":"shadeColor","targetValue":[0.2,0.3,0.4,0.5]},
                        {"material":0,"type":"matcapColor","targetValue":[0.3,0.4,0.5,0.6]},
                        {"material":0,"type":"rimColor","targetValue":[0.4,0.5,0.6,0.7]},
                        {"material":0,"type":"outlineColor","targetValue":[0.5,0.6,0.7,0.8]}
                    ],
                    "textureTransformBinds":[
                        {"material":0},
                        {"material":0,"scale":[2.0,3.0],"offset":[0.25,-0.5]}
                    ]
                }
            },
            "custom": {
                "MyFace": {
                    "morphTargetBinds":[{"node":3,"index":1,"weight":1.0}]
                },
                "materialOnly": {
                    "materialColorBinds":[
                        {"material":0,"type":"color","targetValue":[0.6,0.7,0.8,0.9]}
                    ]
                },
                "textureOnly": {
                    "textureTransformBinds":[
                        {"material":0,"scale":[0.5,0.75],"offset":[0.1,0.2]}
                    ]
                }
            }
        });
        let doc = parse(value).unwrap();
        assert_eq!(doc.expressions.len(), 4);
        let blink = &doc.expressions[0];
        assert_eq!(blink.name, "blink");
        assert_eq!(blink.kind, Vrm1ExpressionKind::Preset);
        assert_eq!(blink.morph_target_binds[0].node, 2);
        assert_eq!(blink.morph_target_binds[0].index, 3);
        assert_eq!(blink.morph_target_binds[0].weight, 0.75);
        assert!(blink.is_binary);
        assert_eq!(blink.override_blink, Vrm1ExpressionOverride::Block);
        assert_eq!(blink.override_look_at, Vrm1ExpressionOverride::Blend);
        assert_eq!(blink.override_mouth, Vrm1ExpressionOverride::None);
        assert_eq!(blink.material_color_binds.len(), 6);
        assert_eq!(
            blink
                .material_color_binds
                .iter()
                .map(|bind| bind.kind)
                .collect::<Vec<_>>(),
            vec![
                Vrm1MaterialColorBindType::Color,
                Vrm1MaterialColorBindType::EmissionColor,
                Vrm1MaterialColorBindType::ShadeColor,
                Vrm1MaterialColorBindType::MatcapColor,
                Vrm1MaterialColorBindType::RimColor,
                Vrm1MaterialColorBindType::OutlineColor,
            ]
        );
        assert_eq!(blink.material_color_binds[0].material, 0);
        assert_eq!(
            blink.material_color_binds[0].target_value,
            [0.1, 0.2, 0.3, 0.4]
        );
        assert_eq!(blink.texture_transform_binds[0].scale, [1.0, 1.0]);
        assert_eq!(blink.texture_transform_binds[0].offset, [0.0, 0.0]);
        assert_eq!(blink.texture_transform_binds[1].scale, [2.0, 3.0]);
        assert_eq!(blink.texture_transform_binds[1].offset, [0.25, -0.5]);
        let face = doc
            .expressions
            .iter()
            .find(|expression| expression.name == "MyFace")
            .unwrap();
        assert_eq!(face.kind, Vrm1ExpressionKind::Custom);
        assert!(!face.is_binary);
        assert_eq!(face.morph_target_binds.len(), 1);
        assert!(face.material_color_binds.is_empty());
        assert!(face.texture_transform_binds.is_empty());
        let material_only = doc
            .expressions
            .iter()
            .find(|expression| expression.name == "materialOnly")
            .unwrap();
        assert!(material_only.morph_target_binds.is_empty());
        assert_eq!(material_only.material_color_binds.len(), 1);
        assert!(material_only.texture_transform_binds.is_empty());
        let texture_only = doc
            .expressions
            .iter()
            .find(|expression| expression.name == "textureOnly")
            .unwrap();
        assert!(texture_only.morph_target_binds.is_empty());
        assert!(texture_only.material_color_binds.is_empty());
        assert_eq!(texture_only.texture_transform_binds.len(), 1);
    }

    #[test]
    fn expression_material_binds_reject_malformed_values() {
        let expression = |bind: Value, key: &str| {
            let mut value = valid_document();
            value["extensions"]["VRMC_vrm"]["expressions"] = json!({
                "custom": {"materialOnly": {(key): [bind]}}
            });
            value
        };

        assert!(
            parse(expression(
                json!({"material":1,"type":"color","targetValue":[1.0,1.0,1.0,1.0]}),
                "materialColorBinds"
            ))
            .is_err()
        );
        assert!(
            parse(expression(
                json!({"material":0,"type":"invalid","targetValue":[1.0,1.0,1.0,1.0]}),
                "materialColorBinds"
            ))
            .is_err()
        );
        assert!(
            parse(expression(
                json!({"material":0,"type":"color","targetValue":[1.0,1.0,1.0]}),
                "materialColorBinds"
            ))
            .is_err()
        );
        assert!(
            parse(expression(
                json!({"material":0,"type":"color","targetValue":[3.5e38_f64,1.0,1.0,1.0]}),
                "materialColorBinds"
            ))
            .is_err()
        );
        assert!(
            parse(expression(
                json!({"material":0,"scale":[1.0,3.5e38_f64]}),
                "textureTransformBinds"
            ))
            .is_err()
        );
        assert!(
            parse(expression(
                json!({"material":0,"offset":[0.0]}),
                "textureTransformBinds"
            ))
            .is_err()
        );
    }

    #[test]
    fn node_mesh_mapping_is_retained_and_checked() {
        let mut value = valid_document();
        value["meshes"] = json!([{"primitives":[]}]);
        value["nodes"][0]["mesh"] = json!(0);
        value["nodes"][1]["mesh"] = json!(0);
        let doc = parse(value).unwrap();
        assert_eq!(doc.node_meshes[0], Some(0));
        assert_eq!(doc.node_meshes[1], Some(0));
        assert!(doc.node_meshes[2].is_none());

        let mut out_of_range = valid_document();
        out_of_range["meshes"] = json!([{"primitives":[]}]);
        out_of_range["nodes"][0]["mesh"] = json!(1);
        let error = parse(out_of_range).unwrap_err().to_string();
        assert!(error.contains("node 0.mesh 1 is out of range"), "{error}");
    }

    #[test]
    fn malformed_expression_structure_is_rejected() {
        let mut non_object = valid_document();
        non_object["extensions"]["VRMC_vrm"]["expressions"] = json!({"preset":[]});
        let error = parse(non_object).unwrap_err().to_string();
        assert!(
            error.contains("expressions.preset must be an object"),
            "{error}"
        );

        let mut malformed_bind = valid_document();
        malformed_bind["extensions"]["VRMC_vrm"]["expressions"] = json!({
            "custom":{"face":{"morphTargetBinds":[{"node":0,"index":"bad","weight":1.0}]}}
        });
        let error = parse(malformed_bind).unwrap_err().to_string();
        assert!(
            error.contains("morph bind.index must be an integer"),
            "{error}"
        );

        let mut malformed_override = valid_document();
        malformed_override["extensions"]["VRMC_vrm"]["expressions"] =
            json!({"custom":{"face":{"overrideBlink":"invalid"}}});
        let error = parse(malformed_override).unwrap_err().to_string();
        assert!(error.contains("overrideBlink must be one of"), "{error}");
    }

    #[test]
    fn custom_expression_cannot_use_a_canonical_preset_name() {
        let mut value = valid_document();
        value["extensions"]["VRMC_vrm"]["expressions"] = json!({"custom":{"blink":{}}});
        let error = parse(value).unwrap_err().to_string();
        assert!(error.contains("conflicts with a preset"), "{error}");
    }

    #[test]
    fn meta_version_is_optional_and_empty_is_valid() {
        let mut value = valid_document();
        value["extensions"]["VRMC_vrm"]["meta"]
            .as_object_mut()
            .unwrap()
            .remove("version");
        assert_eq!(parse(value).unwrap().meta.version, None);

        let mut value = valid_document();
        value["extensions"]["VRMC_vrm"]["meta"]["version"] = json!("");
        assert_eq!(parse(value).unwrap().meta.version.as_deref(), Some(""));
    }

    #[test]
    fn material_and_node_extensions_use_standard_locations() {
        let mut value = valid_document();
        value["materials"][0]["extensions"]
            .as_object_mut()
            .unwrap()
            .remove("VRMC_materials_mtoon");
        value["nodes"][0]["extensions"]
            .as_object_mut()
            .unwrap()
            .remove("VRMC_node_constraint");
        value["extensions"].as_object_mut().unwrap().insert(
            "VRMC_materials_mtoon".to_owned(),
            json!({"specVersion":"1.0"}),
        );
        value["extensions"].as_object_mut().unwrap().insert(
            "VRMC_node_constraint".to_owned(),
            json!({"specVersion":"1.0"}),
        );
        let doc = parse(value).unwrap();
        assert!(!doc.materials_mtoon.present);
        assert!(!doc.node_constraint.present);
    }

    #[test]
    fn optional_future_mtoon_falls_back_but_required_future_mtoon_is_rejected() {
        let mut mtoon = valid_document();
        mtoon["materials"][0]["extensions"]["VRMC_materials_mtoon"]["specVersion"] = json!("0.9");
        let document = parse(mtoon.clone()).unwrap();
        assert!(!document.materials_mtoon.present);
        assert!(document.mtoon_materials.is_empty());

        mtoon["extensionsRequired"] = json!(["VRMC_materials_mtoon"]);
        let error = parse(mtoon).unwrap_err().to_string();
        assert!(
            error.contains("VRMC_materials_mtoon") && error.contains("1.0"),
            "{error}"
        );
    }

    #[test]
    fn required_mtoon_without_unlit_is_native_and_declaration_consistent() {
        let mut value = valid_document();
        value["extensionsRequired"] = json!(["VRMC_materials_mtoon"]);
        assert!(
            value["materials"][0]["extensions"]
                .get("KHR_materials_unlit")
                .is_none()
        );
        let document = parse(value).unwrap();
        assert!(document.materials_mtoon.present);
        assert_eq!(document.mtoon_materials.len(), 1);

        let mut missing_used = valid_document();
        missing_used["extensionsRequired"] = json!(["VRMC_materials_mtoon"]);
        missing_used["extensionsUsed"] = json!(["VRMC_node_constraint"]);
        let error = parse(missing_used).unwrap_err().to_string();
        assert!(error.contains("extensionsUsed"), "{error}");

        let mut missing_payload = valid_document();
        missing_payload["extensionsRequired"] = json!(["VRMC_materials_mtoon"]);
        missing_payload["materials"][0]["extensions"]
            .as_object_mut()
            .unwrap()
            .remove("VRMC_materials_mtoon");
        let error = parse(missing_payload).unwrap_err().to_string();
        assert!(error.contains("no material declares"), "{error}");
    }

    #[test]
    fn optional_malformed_or_unsupported_mtoon_falls_back_but_required_rejects() {
        let mut malformed = valid_document();
        malformed["materials"][0]["extensions"]["VRMC_materials_mtoon"]["transparentWithZWrite"] =
            json!("invalid");
        let document = parse(malformed.clone()).unwrap();
        assert!(document.mtoon_materials.is_empty());
        malformed["extensionsRequired"] = json!(["VRMC_materials_mtoon"]);
        let error = parse(malformed).unwrap_err().to_string();
        assert!(error.contains("transparentWithZWrite"), "{error}");

        let mut unsupported = valid_document();
        unsupported["images"] = json!([{"uri":"unused.png"}]);
        unsupported["textures"] = json!([{"source":0}]);
        unsupported["materials"][0]["extensions"]["VRMC_materials_mtoon"]["shadeMultiplyTexture"] =
            json!({"index":0,"texCoord":2});
        let document = parse(unsupported.clone()).unwrap();
        assert!(document.mtoon_materials.is_empty());
        unsupported["extensionsRequired"] = json!(["VRMC_materials_mtoon"]);
        let error = parse(unsupported).unwrap_err().to_string();
        assert!(error.contains("TEXCOORD_2"), "{error}");
    }

    #[test]
    fn supported_required_texture_transform_override_remains_native() {
        let mut value = valid_document();
        value["extensionsUsed"] = json!([
            "VRMC_materials_mtoon",
            "VRMC_node_constraint",
            "KHR_texture_transform"
        ]);
        value["extensionsRequired"] = json!(["VRMC_materials_mtoon", "KHR_texture_transform"]);
        value["images"] = json!([{"uri":"unused.png"}]);
        value["textures"] = json!([{"source":0}]);
        value["materials"][0]["extensions"]["VRMC_materials_mtoon"]["shadeMultiplyTexture"] = json!({
            "index":0,
            "texCoord":0,
            "extensions":{"KHR_texture_transform":{"texCoord":1}}
        });
        let document = parse(value).unwrap();
        assert_eq!(
            document.mtoon_materials[0]
                .shade_multiply_texture
                .as_ref()
                .unwrap()
                .effective_tex_coord(),
            1
        );
    }

    #[test]
    fn recognized_non_material_extension_versions_are_required_and_validated() {
        let mut node_constraint = valid_document();
        node_constraint["extensionsRequired"] = json!(["VRMC_node_constraint"]);
        node_constraint["nodes"][0]["extensions"]["VRMC_node_constraint"]
            .as_object_mut()
            .unwrap()
            .remove("specVersion");
        let error = parse(node_constraint).unwrap_err().to_string();
        assert!(
            error.contains("VRMC_node_constraint") && error.contains("specVersion"),
            "{error}"
        );

        let mut spring_bone = valid_document();
        spring_bone["extensions"]["VRMC_springBone"]["specVersion"] = json!("0.9");
        let error = parse(spring_bone).unwrap_err().to_string();
        assert!(
            error.contains("VRMC_springBone") && error.contains("1.0"),
            "{error}"
        );
    }

    #[test]
    fn missing_vrmc_extension_is_rejected() {
        let mut value = valid_document();
        value["extensions"]
            .as_object_mut()
            .unwrap()
            .remove("VRMC_vrm");
        let error = parse(value).unwrap_err().to_string();
        assert!(error.contains("missing required VRMC_vrm"), "{error}");
    }

    #[test]
    fn wrong_or_missing_vrm_spec_version_is_rejected() {
        let mut wrong = valid_document();
        wrong["extensions"]["VRMC_vrm"]["specVersion"] = json!("0.99");
        let error = parse(wrong).unwrap_err().to_string();
        assert!(error.contains("specVersion must be \"1.0\""), "{error}");

        let mut missing = valid_document();
        missing["extensions"]["VRMC_vrm"]
            .as_object_mut()
            .unwrap()
            .remove("specVersion");
        let error = parse(missing).unwrap_err().to_string();
        assert!(error.contains("missing required specVersion"), "{error}");
    }

    #[test]
    fn legacy_vrm0_is_rejected() {
        let mut value = valid_document();
        value["extensions"]
            .as_object_mut()
            .unwrap()
            .insert("VRM".to_owned(), json!({"meta":{},"humanoid":{}}));
        let error = parse(value).unwrap_err().to_string();
        assert!(error.contains("legacy VRM 0.x"), "{error}");
    }

    #[test]
    fn required_meta_and_humanoid_sections_are_validated() {
        let mut missing_meta = valid_document();
        missing_meta["extensions"]["VRMC_vrm"]
            .as_object_mut()
            .unwrap()
            .remove("meta");
        let error = parse(missing_meta).unwrap_err().to_string();
        assert!(error.contains("missing required meta"), "{error}");

        let mut missing_humanoid = valid_document();
        missing_humanoid["extensions"]["VRMC_vrm"]
            .as_object_mut()
            .unwrap()
            .remove("humanoid");
        let error = parse(missing_humanoid).unwrap_err().to_string();
        assert!(error.contains("missing required humanoid"), "{error}");
    }

    #[test]
    fn required_bone_and_node_indices_are_validated() {
        let mut missing_bone = valid_document();
        missing_bone["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]
            .as_object_mut()
            .unwrap()
            .remove("head");
        let error = parse(missing_bone).unwrap_err().to_string();
        assert!(
            error.contains("missing required human bone head"),
            "{error}"
        );

        let mut out_of_range = valid_document();
        out_of_range["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]["head"]["node"] =
            json!(999);
        let error = parse(out_of_range).unwrap_err().to_string();
        assert!(error.contains("out of range"), "{error}");
    }

    #[test]
    fn duplicate_bone_mapping_is_rejected() {
        let mut value = valid_document();
        value["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]["head"]["node"] = json!(0);
        let error = parse(value).unwrap_err().to_string();
        assert!(error.contains("multiple human bones map"), "{error}");
    }

    #[test]
    fn cyclic_hierarchy_is_rejected() {
        let mut value = valid_document();
        value["nodes"][Vrm1HumanBone::REQUIRED.len() - 1]["children"] = json!([0]);
        let error = parse(value).unwrap_err().to_string();
        assert!(error.contains("contains a cycle"), "{error}");
    }

    #[test]
    fn unknown_optional_extensions_are_tolerated() {
        let doc = parse(valid_document()).unwrap();
        assert_eq!(doc.materials_mtoon.version.as_deref(), Some("1.0"));
    }
}
