//! Minimal VRM 1.0 semantic facts parsed from a self-contained GLB.
//!
//! This module intentionally does not load meshes, apply presentation
//! transforms, or implement any VRM runtime systems. It only validates and
//! retains the semantic information needed by a future host-side loader.

use std::collections::{BTreeMap, HashSet};

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};

use crate::glb;

/// The metadata fields retained from VRMC_vrm.meta.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vrm1Meta {
    pub name: String,
    /// Optional in the VRM 1.0 schema.
    pub version: Option<String>,
    pub authors: Vec<String>,
    pub license_url: String,
}

/// A typed VRM 1.0 humanoid bone name.
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
    const REQUIRED: &'static [Self] = &[
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

    /// The spelling used as a key in humanoid.humanBones.
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

/// Validated VRM 1.0 human-bone to glTF-node mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vrm1Humanoid {
    pub human_bones: BTreeMap<Vrm1HumanBone, usize>,
}

impl Vrm1Humanoid {
    pub fn node_for(&self, bone: Vrm1HumanBone) -> Option<usize> {
        self.human_bones.get(&bone).copied()
    }
}

/// Presence and declared version of a recognized VRM 1.0 extension.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vrm1ExtensionInfo {
    pub present: bool,
    pub version: Option<String>,
}

/// Whether a VRM 1.0 expression came from the preset or custom expression
/// namespace.  The namespaces remain distinct in the parser even though the
/// host addresses both with the expression's authored name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vrm1ExpressionKind {
    Preset,
    Custom,
}

/// VRM 1.0's blink/eye/mouth override mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vrm1ExpressionOverride {
    None,
    Block,
    Blend,
}

/// A typed morph target bind from a VRM 1.0 expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1MorphTargetBind {
    pub node: usize,
    pub index: usize,
    pub weight: f32,
}

/// The expression semantics retained from VRMC_vrm.  Material and texture
/// binds are intentionally represented only by presence: this first runtime
/// slice is morph-only, but their presence still affects validation and
/// diagnostics in the host.
#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1Expression {
    pub name: String,
    pub kind: Vrm1ExpressionKind,
    pub morph_target_binds: Vec<Vrm1MorphTargetBind>,
    pub is_binary: bool,
    pub override_blink: Vrm1ExpressionOverride,
    pub override_look_at: Vrm1ExpressionOverride,
    pub override_mouth: Vrm1ExpressionOverride,
    pub has_material_color_binds: bool,
    pub has_texture_transform_binds: bool,
}

/// Minimal, structurally validated VRM 1.0 semantic facts.
///
/// This is not full humanoid conformance validation: ancestry, positive bone
/// scales, and runtime retargeting rules remain outside this API.
#[derive(Clone, Debug, PartialEq)]
pub struct Vrm1Doc {
    pub meta: Vrm1Meta,
    pub humanoid: Vrm1Humanoid,
    pub materials_mtoon: Vrm1ExtensionInfo,
    pub spring_bone: Vrm1ExtensionInfo,
    pub node_constraint: Vrm1ExtensionInfo,
    pub expressions: Vec<Vrm1Expression>,
    pub has_expressions: bool,
    pub has_look_at: bool,
    pub has_first_person: bool,
    pub node_count: usize,
    /// glTF mesh index for each glTF node, when the node has a mesh.
    pub node_meshes: Vec<Option<usize>>,
    /// Primitive count for each glTF mesh.  The parent uses this to reject a
    /// morph bind when even one primitive of the referenced mesh cannot carry
    /// the target.
    pub mesh_primitive_counts: Vec<usize>,
}

impl Vrm1Doc {
    /// Parse a VRM 1.0 semantic document from a self-contained GLB byte slice.
    ///
    /// This method intentionally does not read files or invoke a mesh
    /// importer. The caller retains ownership of the original bytes.
    pub fn from_glb_bytes(bytes: &[u8]) -> Result<Self> {
        let glb = glb::parse_glb(bytes).context("failed to parse VRM 1.0 GLB")?;
        let extensions = glb
            .json
            .get("extensions")
            .and_then(Value::as_object)
            .context("GLB root is missing an extensions object")?;

        if extensions.contains_key("VRM") {
            bail!("legacy VRM 0.x extension is not accepted by Vrm1Doc");
        }

        let vrm = extensions
            .get("VRMC_vrm")
            .context("GLB root is missing required VRMC_vrm extension")?;
        let vrm = vrm
            .as_object()
            .context("root VRMC_vrm extension must be an object")?;
        let spec_version = required_string(vrm, "specVersion", "VRMC_vrm")?;
        ensure!(
            spec_version == "1.0",
            "VRMC_vrm specVersion must be \"1.0\", got \"{spec_version}\""
        );

        let (node_children, node_meshes, mesh_primitive_counts) = validate_nodes(&glb.json)?;
        let meta = parse_meta(vrm)?;
        let humanoid = parse_humanoid(vrm, node_children.len())?;
        let materials_mtoon =
            extension_info_in_array(&glb.json, "materials", "VRMC_materials_mtoon")?;
        let node_constraint = extension_info_in_array(&glb.json, "nodes", "VRMC_node_constraint")?;

        let has_expressions = vrm.contains_key("expressions");
        let expressions = parse_expressions(vrm)?;

        Ok(Self {
            meta,
            humanoid,
            materials_mtoon,
            spring_bone: extension_info(extensions, "VRMC_springBone")?,
            node_constraint,
            expressions,
            has_expressions,
            has_look_at: optional_object(vrm, "lookAt")?,
            has_first_person: optional_object(vrm, "firstPerson")?,
            node_count: node_children.len(),
            node_meshes,
            mesh_primitive_counts,
        })
    }
}

fn parse_expressions(vrm: &Map<String, Value>) -> Result<Vec<Vrm1Expression>> {
    let Some(value) = vrm.get("expressions") else {
        return Ok(Vec::new());
    };
    let expressions = value
        .as_object()
        .context("VRMC_vrm.expressions must be an object when present")?;
    let mut parsed = Vec::new();
    parse_expression_group(
        expressions,
        "preset",
        Vrm1ExpressionKind::Preset,
        &mut parsed,
    )?;
    parse_expression_group(
        expressions,
        "custom",
        Vrm1ExpressionKind::Custom,
        &mut parsed,
    )?;
    Ok(parsed)
}

fn parse_expression_group(
    expressions: &Map<String, Value>,
    group_name: &str,
    kind: Vrm1ExpressionKind,
    output: &mut Vec<Vrm1Expression>,
) -> Result<()> {
    let Some(value) = expressions.get(group_name) else {
        return Ok(());
    };
    let group = value.as_object().with_context(|| {
        format!("VRMC_vrm.expressions.{group_name} must be an object when present")
    })?;
    for (name, value) in group {
        let expression = value.as_object().with_context(|| {
            format!("VRMC_vrm.expressions.{group_name}.{name} must be an object")
        })?;
        output.push(parse_expression(name, kind, expression)?);
    }
    Ok(())
}

fn parse_expression(
    name: &str,
    kind: Vrm1ExpressionKind,
    expression: &Map<String, Value>,
) -> Result<Vrm1Expression> {
    if kind == Vrm1ExpressionKind::Preset {
        ensure!(
            is_canonical_preset_name(name),
            "unknown VRM 1.0 preset expression {name:?}"
        );
    } else {
        ensure!(
            !is_canonical_preset_name(name),
            "custom VRM 1.0 expression name {name:?} conflicts with a preset"
        );
    }
    let morph_target_binds = match expression.get("morphTargetBinds") {
        None => Vec::new(),
        Some(value) => {
            let binds = value
                .as_array()
                .with_context(|| format!("expression {name}.morphTargetBinds must be an array"))?;
            let mut parsed = Vec::with_capacity(binds.len());
            for (bind_index, value) in binds.iter().enumerate() {
                let bind = value.as_object().with_context(|| {
                    format!("expression {name}.morphTargetBinds[{bind_index}] must be an object")
                })?;
                let node = required_usize(bind, "node", &format!("expression {name} morph bind"))?;
                let index =
                    required_usize(bind, "index", &format!("expression {name} morph bind"))?;
                let weight =
                    required_f32(bind, "weight", &format!("expression {name} morph bind"))?;
                parsed.push(Vrm1MorphTargetBind {
                    node,
                    index,
                    weight,
                });
            }
            parsed
        }
    };

    Ok(Vrm1Expression {
        name: name.to_owned(),
        kind,
        morph_target_binds,
        is_binary: optional_bool(expression, "isBinary", name)?.unwrap_or(false),
        override_blink: parse_override(expression, "overrideBlink", name)?,
        override_look_at: parse_override(expression, "overrideLookAt", name)?,
        override_mouth: parse_override(expression, "overrideMouth", name)?,
        has_material_color_binds: optional_array_presence(expression, "materialColorBinds", name)?,
        has_texture_transform_binds: optional_array_presence(
            expression,
            "textureTransformBinds",
            name,
        )?,
    })
}

fn is_canonical_preset_name(name: &str) -> bool {
    matches!(
        name,
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

fn required_usize(object: &Map<String, Value>, key: &str, context: &str) -> Result<usize> {
    let value = object
        .get(key)
        .with_context(|| format!("{context} is missing required {key}"))?
        .as_u64()
        .with_context(|| format!("{context}.{key} must be an integer"))?;
    usize::try_from(value).with_context(|| format!("{context}.{key} does not fit in usize"))
}

fn required_f32(object: &Map<String, Value>, key: &str, context: &str) -> Result<f32> {
    let value = object
        .get(key)
        .with_context(|| format!("{context} is missing required {key}"))?
        .as_f64()
        .with_context(|| format!("{context}.{key} must be a number"))?;
    Ok(value as f32)
}

fn optional_bool(object: &Map<String, Value>, key: &str, context: &str) -> Result<Option<bool>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    value
        .as_bool()
        .map(Some)
        .with_context(|| format!("expression {context}.{key} must be a boolean"))
}

fn parse_override(
    object: &Map<String, Value>,
    key: &str,
    expression_name: &str,
) -> Result<Vrm1ExpressionOverride> {
    let Some(value) = object.get(key) else {
        return Ok(Vrm1ExpressionOverride::None);
    };
    let value = value.as_str().with_context(|| {
        format!("expression {expression_name}.{key} must be one of none, block, blend")
    })?;
    match value {
        "none" => Ok(Vrm1ExpressionOverride::None),
        "block" => Ok(Vrm1ExpressionOverride::Block),
        "blend" => Ok(Vrm1ExpressionOverride::Blend),
        _ => bail!(
            "expression {expression_name}.{key} must be one of none, block, blend, got {value:?}"
        ),
    }
}

fn optional_array_presence(
    object: &Map<String, Value>,
    key: &str,
    expression_name: &str,
) -> Result<bool> {
    let Some(value) = object.get(key) else {
        return Ok(false);
    };
    ensure!(
        value.is_array(),
        "expression {expression_name}.{key} must be an array when present"
    );
    Ok(true)
}

fn required_string(object: &Map<String, Value>, key: &str, context: &str) -> Result<String> {
    let value = object
        .get(key)
        .with_context(|| format!("{context} is missing required {key}"))?;
    let value = value
        .as_str()
        .with_context(|| format!("{context}.{key} must be a string"))?;
    ensure!(
        !value.trim().is_empty(),
        "{context}.{key} must not be empty"
    );
    Ok(value.to_owned())
}

fn parse_meta(vrm: &Map<String, Value>) -> Result<Vrm1Meta> {
    let meta = vrm
        .get("meta")
        .context("VRMC_vrm is missing required meta section")?
        .as_object()
        .context("VRMC_vrm.meta must be an object")?;
    let authors_value = meta
        .get("authors")
        .context("VRMC_vrm.meta is missing required authors")?;
    let authors_array = authors_value
        .as_array()
        .context("VRMC_vrm.meta.authors must be an array")?;
    ensure!(
        !authors_array.is_empty(),
        "VRMC_vrm.meta.authors must contain at least one author"
    );
    let mut authors = Vec::with_capacity(authors_array.len());
    for (index, author) in authors_array.iter().enumerate() {
        let author = author
            .as_str()
            .with_context(|| format!("VRMC_vrm.meta.authors[{index}] must be a string"))?;
        ensure!(
            !author.trim().is_empty(),
            "VRMC_vrm.meta.authors[{index}] must not be empty"
        );
        authors.push(author.to_owned());
    }
    Ok(Vrm1Meta {
        name: required_string(meta, "name", "VRMC_vrm.meta")?,
        version: optional_string(meta, "version", "VRMC_vrm.meta")?,
        authors,
        license_url: required_string(meta, "licenseUrl", "VRMC_vrm.meta")?,
    })
}

fn optional_string(
    object: &Map<String, Value>,
    key: &str,
    context: &str,
) -> Result<Option<String>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let value = value
        .as_str()
        .with_context(|| format!("{context}.{key} must be a string"))?;
    Ok(Some(value.to_owned()))
}

fn parse_humanoid(vrm: &Map<String, Value>, node_count: usize) -> Result<Vrm1Humanoid> {
    let human_bones = vrm
        .get("humanoid")
        .context("VRMC_vrm is missing required humanoid section")?
        .get("humanBones")
        .context("VRMC_vrm.humanoid is missing required humanBones")?
        .as_object()
        .context("VRMC_vrm.humanoid.humanBones must be an object")?;

    let mut mappings = BTreeMap::new();
    let mut used_nodes = HashSet::new();
    for (name, entry) in human_bones {
        let bone = Vrm1HumanBone::parse(name)
            .with_context(|| format!("unknown VRM 1.0 human bone {name}"))?;
        let entry = entry
            .as_object()
            .with_context(|| format!("VRMC_vrm.humanoid.humanBones.{name} must be an object"))?;
        let node = entry
            .get("node")
            .with_context(|| format!("human bone {name} is missing node"))?
            .as_u64()
            .with_context(|| format!("human bone {name}.node must be an integer"))?;
        let node = usize::try_from(node)
            .with_context(|| format!("human bone {name}.node does not fit in usize"))?;
        ensure!(
            node < node_count,
            "human bone {name}.node {node} is out of range for {node_count} nodes"
        );
        ensure!(
            used_nodes.insert(node),
            "multiple human bones map to glTF node {node}"
        );
        ensure!(
            mappings.insert(bone, node).is_none(),
            "duplicate human bone {name}"
        );
    }

    for &bone in Vrm1HumanBone::REQUIRED {
        ensure!(
            mappings.contains_key(&bone),
            "VRMC_vrm.humanoid is missing required human bone {}",
            bone.as_str()
        );
    }
    Ok(Vrm1Humanoid {
        human_bones: mappings,
    })
}

fn extension_info(extensions: &Map<String, Value>, name: &str) -> Result<Vrm1ExtensionInfo> {
    let Some(value) = extensions.get(name) else {
        return Ok(Vrm1ExtensionInfo::default());
    };
    required_extension_info(value, name)
}

fn required_extension_info(value: &Value, name: &str) -> Result<Vrm1ExtensionInfo> {
    let object = value
        .as_object()
        .with_context(|| format!("{name} extension must be an object"))?;
    let version = required_string(object, "specVersion", name)?;
    ensure!(
        version == "1.0",
        "{name}.specVersion must be \"1.0\", got \"{version}\""
    );
    Ok(Vrm1ExtensionInfo {
        present: true,
        version: Some(version),
    })
}

fn extension_info_in_array(
    root: &Value,
    array_name: &str,
    extension_name: &str,
) -> Result<Vrm1ExtensionInfo> {
    let Some(items) = root.get(array_name) else {
        return Ok(Vrm1ExtensionInfo::default());
    };
    let items = items
        .as_array()
        .with_context(|| format!("glTF {array_name} must be an array"))?;
    let mut found: Option<Vrm1ExtensionInfo> = None;
    for (index, item) in items.iter().enumerate() {
        let item = item
            .as_object()
            .with_context(|| format!("glTF {array_name}[{index}] must be an object"))?;
        let Some(item_extensions) = item.get("extensions") else {
            continue;
        };
        let item_extensions = item_extensions
            .as_object()
            .with_context(|| format!("glTF {array_name}[{index}].extensions must be an object"))?;
        let Some(value) = item_extensions.get(extension_name) else {
            continue;
        };
        let context = format!("{array_name}[{index}].extensions.{extension_name}");
        let info = required_extension_info(value, &context)?;
        if let Some(previous) = &found {
            ensure!(
                previous.version == info.version,
                "{extension_name} has conflicting specVersion values"
            );
        }
        found = Some(info);
    }
    Ok(found.unwrap_or_default())
}

fn optional_object(vrm: &Map<String, Value>, name: &str) -> Result<bool> {
    let Some(value) = vrm.get(name) else {
        return Ok(false);
    };
    ensure!(
        value.is_object(),
        "VRMC_vrm.{name} must be an object when present"
    );
    Ok(true)
}

fn validate_nodes(json: &Value) -> Result<(Vec<Vec<usize>>, Vec<Option<usize>>, Vec<usize>)> {
    let nodes = json
        .get("nodes")
        .context("GLB is missing required glTF nodes array")?
        .as_array()
        .context("glTF nodes must be an array")?;
    let count = nodes.len();
    let mesh_primitive_counts = match json.get("meshes") {
        None => Vec::new(),
        Some(value) => {
            let meshes = value.as_array().context("glTF meshes must be an array")?;
            let mut counts = Vec::with_capacity(meshes.len());
            for (index, mesh) in meshes.iter().enumerate() {
                let mesh = mesh
                    .as_object()
                    .with_context(|| format!("glTF mesh {index} must be an object"))?;
                let primitives = mesh
                    .get("primitives")
                    .with_context(|| format!("glTF mesh {index} is missing primitives"))?
                    .as_array()
                    .with_context(|| format!("glTF mesh {index}.primitives must be an array"))?;
                counts.push(primitives.len());
            }
            counts
        }
    };
    let mesh_count = json.get("meshes").map(|_| mesh_primitive_counts.len());
    let mut children = Vec::with_capacity(count);
    let mut node_meshes = Vec::with_capacity(count);
    let mut parent = vec![None; count];
    for (index, node) in nodes.iter().enumerate() {
        let node = node
            .as_object()
            .with_context(|| format!("glTF node {index} must be an object"))?;
        let mesh = match node.get("mesh") {
            None => None,
            Some(value) => {
                let mesh = value
                    .as_u64()
                    .with_context(|| format!("glTF node {index}.mesh must be an integer"))?;
                let mesh = usize::try_from(mesh)
                    .with_context(|| format!("glTF node {index}.mesh index is too large"))?;
                let mesh_count = mesh_count.with_context(|| {
                    format!("glTF node {index}.mesh is present but meshes is missing")
                })?;
                ensure!(
                    mesh < mesh_count,
                    "glTF node {index}.mesh {mesh} is out of range for {mesh_count} meshes"
                );
                Some(mesh)
            }
        };
        node_meshes.push(mesh);
        let mut node_children = Vec::new();
        let mut seen = HashSet::new();
        if let Some(values) = node.get("children") {
            let values = values
                .as_array()
                .with_context(|| format!("glTF node {index}.children must be an array"))?;
            for value in values {
                let child = value.as_u64().with_context(|| {
                    format!("glTF node {index}.children contains a non-integer")
                })?;
                let child = usize::try_from(child)
                    .with_context(|| format!("glTF node {index}.children index is too large"))?;
                ensure!(
                    child < count,
                    "glTF node {index}.children references node {child}, but node count is {count}"
                );
                ensure!(
                    parent[child].replace(index).is_none(),
                    "glTF node {child} has more than one parent"
                );
                ensure!(
                    seen.insert(child),
                    "glTF node {index}.children contains duplicate node {child}"
                );
                node_children.push(child);
            }
        }
        children.push(node_children);
    }

    // Iterative DFS keeps malformed deep input from overflowing the Rust call
    // stack and detects back-edges before any consumer can walk the hierarchy.
    let mut state = vec![0_u8; count];
    for start in 0..count {
        if state[start] != 0 {
            continue;
        }
        state[start] = 1;
        let mut stack = vec![(start, 0_usize)];
        while let Some((node, child_index)) = stack.last_mut() {
            if *child_index == children[*node].len() {
                state[*node] = 2;
                stack.pop();
                continue;
            }
            let child = children[*node][*child_index];
            *child_index += 1;
            match state[child] {
                0 => {
                    state[child] = 1;
                    stack.push((child, 0));
                }
                1 => bail!("glTF node hierarchy contains a cycle through node {child}"),
                2 => {}
                _ => unreachable!(),
            }
        }
    }
    Ok((children, node_meshes, mesh_primitive_counts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn glb(json_value: Value) -> Vec<u8> {
        let mut json_bytes = serde_json::to_vec(&json_value).unwrap();
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }
        let total_len = 12 + 8 + json_bytes.len();
        let mut bytes = Vec::with_capacity(total_len);
        bytes.extend_from_slice(b"glTF");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&(total_len as u32).to_le_bytes());
        bytes.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
        bytes.extend_from_slice(&json_bytes);
        bytes
    }

    fn valid_document() -> Value {
        let mut bones = Map::new();
        for (index, bone) in Vrm1HumanBone::REQUIRED.iter().enumerate() {
            bones.insert(bone.as_str().to_owned(), json!({ "node": index }));
        }
        let mut nodes = Vec::new();
        for index in 0..Vrm1HumanBone::REQUIRED.len() {
            let mut node = json!({});
            if index + 1 < Vrm1HumanBone::REQUIRED.len() {
                node["children"] = json!([index + 1]);
            }
            nodes.push(node);
        }
        nodes[0]["extensions"] = json!({ "VRMC_node_constraint": { "specVersion": "1.0" } });
        let materials =
            json!([{ "extensions": { "VRMC_materials_mtoon": { "specVersion": "1.0" } } }]);
        json!({
            "asset": { "version": "2.0" },
            "nodes": nodes,
            "materials": materials,
            "extensions": {
                "VRMC_vrm": {
                    "specVersion": "1.0",
                    "meta": {
                        "name": "Generated",
                        "version": "1.0",
                        "authors": ["PocketJS"],
                        "licenseUrl": "https://example.invalid/license"
                    },
                    "humanoid": { "humanBones": bones },
                    "expressions": {},
                    "lookAt": {},
                    "firstPerson": {}
                },
                "VRMC_springBone": { "specVersion": "1.0" },
                "X_unknown_optional": { "anything": true }
            }
        })
    }

    fn parse(value: Value) -> Result<Vrm1Doc> {
        Vrm1Doc::from_glb_bytes(&glb(value))
    }

    #[test]
    fn valid_minimal_vrm1_retains_semantic_facts() {
        let doc = parse(valid_document()).unwrap();
        assert_eq!(doc.meta.name, "Generated");
        assert_eq!(doc.meta.version.as_deref(), Some("1.0"));
        assert_eq!(doc.meta.authors, ["PocketJS"]);
        assert_eq!(doc.humanoid.node_for(Vrm1HumanBone::Hips), Some(0));
        assert_eq!(doc.node_count, Vrm1HumanBone::REQUIRED.len());
        assert!(doc.has_expressions);
        assert!(doc.has_look_at);
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
    fn expressions_retain_preset_custom_and_unsupported_semantics() {
        let mut value = valid_document();
        value["extensions"]["VRMC_vrm"]["expressions"] = json!({
            "preset": {
                "blink": {
                    "morphTargetBinds": [{ "node": 2, "index": 3, "weight": 0.75 }],
                    "isBinary": true,
                    "overrideBlink": "block",
                    "overrideLookAt": "blend",
                    "overrideMouth": "none",
                    "materialColorBinds": [],
                    "textureTransformBinds": []
                }
            },
            "custom": {
                "MyFace": {
                    "morphTargetBinds": [{ "node": 3, "index": 1, "weight": 1.0 }]
                }
            }
        });
        let doc = parse(value).unwrap();
        assert_eq!(doc.expressions.len(), 2);
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
        assert!(blink.has_material_color_binds);
        assert!(blink.has_texture_transform_binds);
        assert_eq!(doc.expressions[1].kind, Vrm1ExpressionKind::Custom);
        assert!(!doc.expressions[1].is_binary);
    }

    #[test]
    fn node_mesh_mapping_is_retained_and_checked() {
        let mut value = valid_document();
        value["meshes"] = json!([{ "primitives": [] }]);
        value["nodes"][0]["mesh"] = json!(0);
        value["nodes"][1]["mesh"] = json!(0);
        let doc = parse(value).unwrap();
        assert_eq!(doc.node_meshes[0], Some(0));
        assert_eq!(doc.node_meshes[1], Some(0));
        assert!(doc.node_meshes[2].is_none());

        let mut out_of_range = valid_document();
        out_of_range["meshes"] = json!([{ "primitives": [] }]);
        out_of_range["nodes"][0]["mesh"] = json!(1);
        let error = parse(out_of_range).unwrap_err().to_string();
        assert!(error.contains("node 0.mesh 1 is out of range"), "{error}");
    }

    #[test]
    fn malformed_expression_structure_is_rejected() {
        let mut non_object = valid_document();
        non_object["extensions"]["VRMC_vrm"]["expressions"] = json!({
            "preset": []
        });
        let error = parse(non_object).unwrap_err().to_string();
        assert!(
            error.contains("expressions.preset must be an object"),
            "{error}"
        );

        let mut malformed_bind = valid_document();
        malformed_bind["extensions"]["VRMC_vrm"]["expressions"] = json!({
            "custom": {
                "face": {
                    "morphTargetBinds": [{ "node": 0, "index": "bad", "weight": 1.0 }]
                }
            }
        });
        let error = parse(malformed_bind).unwrap_err().to_string();
        assert!(
            error.contains("morph bind.index must be an integer"),
            "{error}"
        );

        let mut malformed_override = valid_document();
        malformed_override["extensions"]["VRMC_vrm"]["expressions"] = json!({
            "custom": { "face": { "overrideBlink": "invalid" } }
        });
        let error = parse(malformed_override).unwrap_err().to_string();
        assert!(error.contains("overrideBlink must be one of"), "{error}");
    }

    #[test]
    fn custom_expression_cannot_use_a_canonical_preset_name() {
        let mut value = valid_document();
        value["extensions"]["VRMC_vrm"]["expressions"] = json!({
            "custom": { "blink": {} }
        });
        let error = parse(value).unwrap_err().to_string();
        assert!(error.contains("conflicts with a preset"), "{error}");
    }

    #[test]
    fn meta_version_is_optional() {
        let mut value = valid_document();
        value["extensions"]["VRMC_vrm"]["meta"]
            .as_object_mut()
            .unwrap()
            .remove("version");
        let doc = parse(value).unwrap();
        assert_eq!(doc.meta.version, None);
    }

    #[test]
    fn empty_meta_version_is_valid() {
        let mut value = valid_document();
        value["extensions"]["VRMC_vrm"]["meta"]["version"] = json!("");
        let doc = parse(value).unwrap();
        assert_eq!(doc.meta.version.as_deref(), Some(""));
    }

    #[test]
    fn material_and_node_extensions_use_standard_locations() {
        let doc = parse(valid_document()).unwrap();
        assert!(doc.materials_mtoon.present);
        assert!(doc.node_constraint.present);

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
            json!({ "specVersion": "1.0" }),
        );
        value["extensions"].as_object_mut().unwrap().insert(
            "VRMC_node_constraint".to_owned(),
            json!({ "specVersion": "1.0" }),
        );
        let doc = parse(value).unwrap();
        assert!(!doc.materials_mtoon.present);
        assert!(!doc.node_constraint.present);
    }

    #[test]
    fn recognized_extension_versions_are_required_and_validated() {
        let mut mtoon = valid_document();
        mtoon["materials"][0]["extensions"]["VRMC_materials_mtoon"]["specVersion"] = json!("0.9");
        let error = parse(mtoon).unwrap_err().to_string();
        assert!(
            error.contains("VRMC_materials_mtoon") && error.contains("1.0"),
            "{error}"
        );

        let mut node_constraint = valid_document();
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
    fn wrong_or_missing_spec_version_is_rejected() {
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
            .insert("VRM".to_owned(), json!({ "meta": {}, "humanoid": {} }));
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
