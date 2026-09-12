//! `.vrma` (VRMC_vrm_animation) loading and retargeting onto a VRM model.
//!
//! VRM0 keeps the legacy normalized-rig conversion for compatibility. VRM1
//! humanoid rotations are normalized through the source authored rest basis
//! and written through the target authored rest basis. Hips translation is
//! converted through the source and target rest parent transforms.
//!
//! VRM0 hips heights retain the legacy channel-unit calculation. VRM1 hips
//! heights and animated positions use full authored parent TRS transforms,
//! including scale, before converting into the target hips-parent space.

use anyhow::{Context, Result, bail, ensure};
use glam::{Mat4, Quat, Vec3};
use pocket3d::anim::{Channel, ChannelPath, Clip, Interpolation, NodeTrs, Skeleton};
use serde_json::Value;

use crate::glb::{self, GltfNodes};
use crate::vrm1::Vrm1Humanoid;

/// A parsed `.vrma` document: glTF animation 0 decoded into pocket3d
/// [`Channel`]s (node = vrma node index) plus the humanoid map and the rest
/// hierarchy needed for hips-height scaling.
pub struct VrmaDoc {
    pub name: String,
    pub duration: f32,
    /// VRM humanoid bone name → vrma node index.
    pub humanoid: Vec<(String, usize)>,
    /// Translation/rotation channels of glTF animation 0.
    pub channels: Vec<Channel>,
    /// Rest hierarchy of the vrma's own nodes.
    pub nodes: GltfNodes,
}

impl VrmaDoc {
    /// Look up a humanoid bone's vrma node index.
    pub fn humanoid_node(&self, bone: &str) -> Option<usize> {
        self.humanoid
            .iter()
            .find(|(name, _)| name == bone)
            .map(|&(_, node)| node)
    }
}

/// Non-owning view over either VRM humanoid mapping representation.
///
/// The parsers remain deliberately separate. This view only joins the small
/// amount of semantic information the retargeter needs: bone lookup and the
/// authored convention of the target.
#[derive(Clone, Copy, Debug)]
pub enum HumanoidView<'a> {
    Vrm0(&'a [(String, usize)]),
    Vrm1(&'a Vrm1Humanoid),
}

impl HumanoidView<'_> {
    fn node_for(self, bone: &str) -> Option<usize> {
        match self {
            Self::Vrm0(humanoid) => humanoid
                .iter()
                .find(|(name, _)| name == bone)
                .map(|&(_, node)| node),
            Self::Vrm1(humanoid) => humanoid
                .human_bones
                .iter()
                .find(|(name, _)| name.as_str() == bone)
                .map(|(_, &node)| node),
        }
    }

    fn visit_mappings(self, mut visit: impl FnMut(&str, usize) -> Result<()>) -> Result<()> {
        match self {
            Self::Vrm0(humanoid) => {
                for (name, node) in humanoid {
                    visit(name, *node)?;
                }
            }
            Self::Vrm1(humanoid) => {
                for (name, node) in &humanoid.human_bones {
                    visit(name.as_str(), *node)?;
                }
            }
        }
        Ok(())
    }

    fn is_vrm0(self) -> bool {
        matches!(self, Self::Vrm0(_))
    }
}

/// Parse a `.vrma` GLB: glTF animation 0 plus `VRMC_vrm_animation`.
/// Sampler interpolation LINEAR/STEP is kept; CUBICSPLINE keeps the middle
/// values and degrades to LINEAR. Sparse accessors are rejected.
pub fn load_vrma_bytes(bytes: &[u8]) -> Result<VrmaDoc> {
    let glb = glb::parse_glb(bytes)?;
    let ext = glb
        .json
        .get("extensions")
        .and_then(|e| e.get("VRMC_vrm_animation"))
        .context("not a .vrma: missing extensions.VRMC_vrm_animation")?;
    let spec_version = ext
        .get("specVersion")
        .and_then(Value::as_str)
        .context("VRMC_vrm_animation.specVersion must be a string")?;
    ensure!(
        spec_version == "1.0",
        "VRMC_vrm_animation.specVersion must be \"1.0\", got \"{spec_version}\""
    );

    // humanoid.humanBones is an object: { boneName: { node } }. serde_json
    // objects iterate in sorted key order, so this Vec is deterministic.
    let mut humanoid = Vec::new();
    if let Some(bones) = ext
        .get("humanoid")
        .and_then(|h| h.get("humanBones"))
        .and_then(Value::as_object)
    {
        for (name, v) in bones {
            if let Some(node) = v.get("node").and_then(Value::as_u64) {
                humanoid.push((name.clone(), node as usize));
            }
        }
    }

    let anim = glb
        .json
        .get("animations")
        .and_then(|a| a.get(0))
        .context(".vrma has no animations")?;
    let name = anim
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("vrma")
        .to_string();
    let samplers = anim
        .get("samplers")
        .and_then(Value::as_array)
        .context("animation has no samplers")?;

    let mut channels = Vec::new();
    let mut duration = 0.0f32;
    for ch in anim
        .get("channels")
        .and_then(Value::as_array)
        .context("animation has no channels")?
    {
        let target = ch.get("target").context("channel has no target")?;
        let Some(node) = target.get("node").and_then(Value::as_u64) else {
            continue; // targets without a node are legal glTF; skip
        };
        let path = match target.get("path").and_then(Value::as_str) {
            Some("translation") => ChannelPath::Translation,
            Some("rotation") => ChannelPath::Rotation,
            // "scale"/"weights" (and unknown paths) are out of scope.
            _ => continue,
        };
        let si = ch
            .get("sampler")
            .and_then(Value::as_u64)
            .context("channel has no sampler")? as usize;
        let sampler = samplers.get(si).context("sampler index out of range")?;
        let input = sampler
            .get("input")
            .and_then(Value::as_u64)
            .context("sampler input")?;
        let output = sampler
            .get("output")
            .and_then(Value::as_u64)
            .context("sampler output")?;
        let interp = sampler
            .get("interpolation")
            .and_then(Value::as_str)
            .unwrap_or("LINEAR");

        let (times, tc) = glb::read_f32_accessor(&glb.json, glb.bin, input as usize)?;
        ensure!(tc == 1, "sampler input must be SCALAR");
        let (mut values, comps) = glb::read_f32_accessor(&glb.json, glb.bin, output as usize)?;
        let expected = match path {
            ChannelPath::Translation => 3,
            ChannelPath::Rotation => 4,
            ChannelPath::Scale => unreachable!(),
        };
        ensure!(
            comps == expected,
            "sampler output has {comps} components, expected {expected}"
        );
        let interpolation = match interp {
            "STEP" => Interpolation::Step,
            "CUBICSPLINE" => {
                // Keys are (in-tangent, value, out-tangent); keep the values.
                ensure!(
                    values.len() == times.len() * comps * 3,
                    "CUBICSPLINE output count mismatch"
                );
                values = (0..times.len())
                    .flat_map(|k| {
                        let base = (k * 3 + 1) * comps;
                        values[base..base + comps].to_vec()
                    })
                    .collect();
                Interpolation::Linear
            }
            _ => Interpolation::Linear,
        };
        ensure!(
            values.len() == times.len() * comps,
            "sampler output count mismatch"
        );
        duration = duration.max(times.last().copied().unwrap_or(0.0));
        channels.push(Channel {
            node: node as usize,
            path,
            interpolation,
            times,
            values,
        });
    }

    Ok(VrmaDoc {
        name,
        duration,
        humanoid,
        channels,
        nodes: GltfNodes::parse(&glb.json)?,
    })
}

/// Retarget a VRMA clip using the legacy VRM0 behavior.
///
/// This compatibility wrapper intentionally keeps the old public API for
/// callers that already have a VRM0 `humanBones` array.
pub fn retarget(
    vrma: &VrmaDoc,
    humanoid: &[(String, usize)],
    model_skeleton: &Skeleton,
) -> Result<Clip> {
    retarget_with_humanoid(vrma, HumanoidView::Vrm0(humanoid), model_skeleton)
}

/// Retarget a VRMA clip onto either a VRM0 or VRM1 humanoid target.
pub fn retarget_with_humanoid(
    vrma: &VrmaDoc,
    target: HumanoidView<'_>,
    model_skeleton: &Skeleton,
) -> Result<Clip> {
    validate_vrma(vrma)?;
    validate_skeleton(
        &model_skeleton.parents,
        &model_skeleton.rest,
        "model skeleton",
    )?;
    validate_target_mappings(target, model_skeleton)?;

    if target.is_vrm0() {
        retarget_vrm0(vrma, target, model_skeleton)
    } else {
        retarget_vrm1(vrma, target, model_skeleton)
    }
}

fn retarget_vrm0(
    vrma: &VrmaDoc,
    target: HumanoidView<'_>,
    model_skeleton: &Skeleton,
) -> Result<Clip> {
    let model_hips = target
        .node_for("hips")
        .context("model humanoid has no hips")?;
    ensure!(
        model_hips < model_skeleton.rest.len(),
        "model hips node out of range"
    );
    let vrma_hips = vrma.humanoid_node("hips");

    let model_h = channel_units_height(model_hips, &model_skeleton.parents, &model_skeleton.rest)?;
    ensure!(
        model_h > 1e-4,
        "model hips rest height is not positive ({model_h})"
    );

    // vrma node → bone name (first mapping wins; maps are tiny).
    let bone_of = |node: usize| -> Option<&str> {
        vrma.humanoid
            .iter()
            .find(|&&(_, n)| n == node)
            .map(|(name, _)| name.as_str())
    };

    let mut channels = Vec::new();
    for ch in &vrma.channels {
        let Some(bone) = bone_of(ch.node) else {
            continue; // non-humanoid channel: drop
        };
        let Some(model_node) = target.node_for(bone) else {
            continue; // model lacks this bone: drop
        };
        match ch.path {
            ChannelPath::Rotation => {
                // VRMC_vrm_animation poses live in the VRM 1.0 humanoid
                // space (character faces +Z); VRM 0.x rigs face -Z. Conjugate
                // every rotation by the 180° yaw between the spaces —
                // R_y(π) q R_y(π)⁻¹ — which for quaternions is (-x, y, -z, w).
                let values: Vec<f32> = ch
                    .values
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|q| [-q[0], q[1], -q[2], q[3]])
                    .collect();
                channels.push(Channel {
                    node: model_node,
                    path: ChannelPath::Rotation,
                    interpolation: ch.interpolation,
                    times: ch.times.clone(),
                    values,
                });
            }
            ChannelPath::Translation if bone == "hips" => {
                let hips_node = vrma_hips.context("vrma humanoid has no hips")?;
                ensure!(
                    hips_node < vrma.nodes.rest.len(),
                    "vrma hips node out of range"
                );
                let vrma_h =
                    channel_units_height(hips_node, &vrma.nodes.parents, &vrma.nodes.rest)?;
                ensure!(
                    vrma_h > 1e-4,
                    "vrma hips rest height is not positive ({vrma_h})"
                );
                let scale = model_h / vrma_h;
                let mut values: Vec<f32> = ch.values.iter().map(|v| v * scale).collect();
                // Re-anchor: keep the loop centered over the origin in X/Z.
                // The same +Z→-Z yaw flip as rotations applies: negate X/Z.
                if values.len() >= 3 {
                    let (x0, z0) = (values[0], values[2]);
                    for key in values.as_chunks_mut::<3>().0 {
                        key[0] = -(key[0] - x0);
                        key[2] = -(key[2] - z0);
                    }
                }
                channels.push(Channel {
                    node: model_node,
                    path: ChannelPath::Translation,
                    interpolation: ch.interpolation,
                    times: ch.times.clone(),
                    values,
                });
            }
            _ => {} // non-hips translation, scale: drop
        }
    }
    if channels.is_empty() {
        bail!("retarget produced no channels (no humanoid overlap?)");
    }
    Ok(Clip {
        name: vrma.name.clone(),
        duration: vrma.duration,
        channels,
    })
}

fn retarget_vrm1(
    vrma: &VrmaDoc,
    target: HumanoidView<'_>,
    model_skeleton: &Skeleton,
) -> Result<Clip> {
    let source_hips = vrma.humanoid_node("hips");
    let target_hips = target.node_for("hips");

    let hips_conversion = if let (Some(source_hips), Some(target_hips)) = (source_hips, target_hips)
    {
        Some(prepare_vrm1_hips_conversion(
            vrma,
            source_hips,
            target_hips,
            model_skeleton,
        )?)
    } else {
        None
    };

    let bone_of = |node: usize| -> Option<&str> {
        vrma.humanoid
            .iter()
            .find(|&&(_, mapped_node)| mapped_node == node)
            .map(|(name, _)| name.as_str())
    };

    let mut channels = Vec::new();
    for ch in &vrma.channels {
        let Some(bone) = bone_of(ch.node) else {
            continue;
        };
        let Some(model_node) = target.node_for(bone) else {
            continue;
        };
        match ch.path {
            ChannelPath::Rotation => {
                let source_local = normalized_rest_rotation(
                    vrma.nodes.rest[ch.node].rotation,
                    "source local rest rotation",
                )?;
                let source_world =
                    rest_global_rotation(ch.node, &vrma.nodes.parents, &vrma.nodes.rest, "source")?;
                let target_local = normalized_rest_rotation(
                    model_skeleton.rest[model_node].rotation,
                    "target local rest rotation",
                )?;
                let target_world = rest_global_rotation(
                    model_node,
                    &model_skeleton.parents,
                    &model_skeleton.rest,
                    "target",
                )?;

                let mut values = Vec::with_capacity(ch.values.len());
                for q in ch.values.as_chunks::<4>().0 {
                    let animated = normalized_animation_rotation(*q)?;
                    let normalized =
                        (source_world * source_local.inverse() * animated * source_world.inverse())
                            .normalize();
                    let converted =
                        (target_local * target_world.inverse() * normalized * target_world)
                            .normalize();
                    ensure!(
                        converted.is_finite(),
                        "VRM1 retarget produced a non-finite rotation for {bone}"
                    );
                    values.extend_from_slice(&converted.to_array());
                }
                channels.push(Channel {
                    node: model_node,
                    path: ChannelPath::Rotation,
                    interpolation: ch.interpolation,
                    times: ch.times.clone(),
                    values,
                });
            }
            ChannelPath::Translation if bone == "hips" => {
                let conversion = hips_conversion
                    .as_ref()
                    .context("VRM1 hips translation requires source and target hips")?;
                let values = convert_vrm1_hips_translation(ch, conversion)?;
                channels.push(Channel {
                    node: model_node,
                    path: ChannelPath::Translation,
                    interpolation: ch.interpolation,
                    times: ch.times.clone(),
                    values,
                });
            }
            _ => {} // non-hips translation, scale: drop
        }
    }
    if channels.is_empty() {
        bail!("retarget produced no channels (no humanoid overlap?)");
    }
    Ok(Clip {
        name: vrma.name.clone(),
        duration: vrma.duration,
        channels,
    })
}

struct HipsConversion {
    source_parent: Mat4,
    target_parent_inverse: Mat4,
    scale: f32,
}

fn prepare_vrm1_hips_conversion(
    vrma: &VrmaDoc,
    source_hips: usize,
    target_hips: usize,
    model_skeleton: &Skeleton,
) -> Result<HipsConversion> {
    let source_rest =
        rest_global_matrix(source_hips, &vrma.nodes.parents, &vrma.nodes.rest, "source")?;
    let target_rest = rest_global_matrix(
        target_hips,
        &model_skeleton.parents,
        &model_skeleton.rest,
        "target",
    )?;
    let source_height = source_rest.transform_point3(Vec3::ZERO).y;
    let target_height = target_rest.transform_point3(Vec3::ZERO).y;
    ensure!(
        source_height.is_finite() && source_height > 1.0e-4,
        "source hips rest height is not positive ({source_height})"
    );
    ensure!(
        target_height.is_finite() && target_height > 1.0e-4,
        "target hips rest height is not positive ({target_height})"
    );

    let source_parent =
        rest_parent_global_matrix(source_hips, &vrma.nodes.parents, &vrma.nodes.rest, "source")?;
    let target_parent = rest_parent_global_matrix(
        target_hips,
        &model_skeleton.parents,
        &model_skeleton.rest,
        "target",
    )?;
    let target_parent_inverse =
        checked_inverse(target_parent, "target hips parent rest transform")?;
    let scale = target_height / source_height;
    ensure!(
        scale.is_finite() && scale > 0.0,
        "invalid hips height scale ({scale})"
    );

    Ok(HipsConversion {
        source_parent,
        target_parent_inverse,
        scale,
    })
}

fn convert_vrm1_hips_translation(ch: &Channel, conversion: &HipsConversion) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(ch.values.len());
    let mut first_model: Option<Vec3> = None;
    for key in ch.values.as_chunks::<3>().0 {
        let source_local = Vec3::from_array(*key);
        let mut model = conversion.source_parent.transform_point3(source_local) * conversion.scale;
        ensure!(model.is_finite(), "VRM1 hips translation became non-finite");
        if let Some(first) = first_model {
            model.x -= first.x;
            model.z -= first.z;
        } else {
            first_model = Some(model);
            model.x = 0.0;
            model.z = 0.0;
        }
        let target_local = conversion.target_parent_inverse.transform_point3(model);
        ensure!(
            target_local.is_finite(),
            "VRM1 hips translation became non-finite in target space"
        );
        out.extend_from_slice(&target_local.to_array());
    }
    Ok(out)
}

fn validate_vrma(vrma: &VrmaDoc) -> Result<()> {
    ensure!(
        vrma.duration.is_finite() && vrma.duration >= 0.0,
        "VRMA duration is invalid"
    );
    validate_skeleton(&vrma.nodes.parents, &vrma.nodes.rest, "VRMA rest hierarchy")?;
    for (name, node) in &vrma.humanoid {
        ensure!(
            *node < vrma.nodes.rest.len(),
            "VRMA humanoid bone {name} node {node} is out of range"
        );
        validate_node_trs(
            &vrma.nodes.rest[*node],
            "VRMA humanoid rest transform",
            false,
        )?;
    }
    for ch in &vrma.channels {
        let mapped = vrma.humanoid.iter().any(|(_, node)| *node == ch.node);
        if !mapped {
            continue;
        }
        ch.validate()?;
        if ch.path == ChannelPath::Rotation {
            for (key, q) in ch.values.as_chunks::<4>().0.iter().enumerate() {
                let length_squared = q.iter().map(|value| value * value).sum::<f32>();
                ensure!(
                    length_squared.is_finite() && length_squared > 1.0e-12,
                    "VRMA rotation key {key} has a zero or non-finite quaternion"
                );
            }
        }
    }
    Ok(())
}

fn validate_target_mappings(target: HumanoidView<'_>, skeleton: &Skeleton) -> Result<()> {
    target.visit_mappings(|name, node| {
        ensure!(
            node < skeleton.rest.len(),
            "target humanoid bone {name} node {node} is out of range"
        );
        validate_node_trs(
            &skeleton.rest[node],
            "target humanoid rest transform",
            false,
        )
    })
}

fn validate_skeleton(parents: &[usize], rest: &[NodeTrs], label: &str) -> Result<()> {
    ensure!(
        parents.len() == rest.len(),
        "{label} parent/rest length mismatch"
    );
    for (index, &parent) in parents.iter().enumerate() {
        ensure!(
            parent == usize::MAX || parent < rest.len(),
            "{label} node {index} parent {parent} is out of range"
        );
        validate_node_trs(&rest[index], label, false)?;
    }
    Ok(())
}

fn validate_node_trs(trs: &NodeTrs, label: &str, invertible: bool) -> Result<()> {
    ensure!(
        trs.translation.is_finite() && trs.rotation.is_finite() && trs.scale.is_finite(),
        "{label} contains non-finite values"
    );
    ensure!(
        trs.rotation.length_squared() > 1.0e-12,
        "{label} contains a zero rotation"
    );
    if invertible {
        checked_inverse(trs.matrix(), label)?;
    }
    Ok(())
}

fn normalized_rest_rotation(rotation: Quat, label: &str) -> Result<Quat> {
    ensure!(
        rotation.is_finite() && rotation.length_squared() > 1.0e-12,
        "{label} is invalid"
    );
    Ok(rotation.normalize())
}

fn normalized_animation_rotation(values: [f32; 4]) -> Result<Quat> {
    let rotation = Quat::from_array(values);
    ensure!(
        rotation.is_finite() && rotation.length_squared() > 1.0e-12,
        "VRMA animation rotation is invalid"
    );
    Ok(rotation.normalize())
}

fn rest_global_matrix(
    node: usize,
    parents: &[usize],
    rest: &[NodeTrs],
    label: &str,
) -> Result<Mat4> {
    let chain = rest_chain(node, parents, rest, label)?;
    let mut global = Mat4::IDENTITY;
    for index in chain.into_iter().rev() {
        validate_node_trs(&rest[index], label, true)?;
        global *= rest[index].matrix();
        ensure!(global.is_finite(), "{label} rest transform is non-finite");
    }
    checked_inverse(global, &format!("{label} rest transform"))?;
    Ok(global)
}

fn rest_parent_global_matrix(
    node: usize,
    parents: &[usize],
    rest: &[NodeTrs],
    label: &str,
) -> Result<Mat4> {
    ensure!(node < rest.len(), "{label} node {node} is out of range");
    match parents[node] {
        usize::MAX => Ok(Mat4::IDENTITY),
        parent => rest_global_matrix(parent, parents, rest, label),
    }
}

fn rest_global_rotation(
    node: usize,
    parents: &[usize],
    rest: &[NodeTrs],
    label: &str,
) -> Result<Quat> {
    let chain = rest_chain(node, parents, rest, label)?;
    let mut global = Quat::IDENTITY;
    for index in chain.into_iter().rev() {
        validate_node_trs(&rest[index], label, true)?;
        let local = normalized_rest_rotation(rest[index].rotation, "rest rotation")?;
        global = (global * local).normalize();
    }
    ensure!(global.is_finite(), "{label} rest rotation is non-finite");
    Ok(global)
}

fn rest_chain(node: usize, parents: &[usize], rest: &[NodeTrs], label: &str) -> Result<Vec<usize>> {
    ensure!(node < rest.len(), "{label} node {node} is out of range");
    let mut chain = Vec::new();
    let mut current = node;
    loop {
        ensure!(
            current < rest.len(),
            "{label} node {current} is out of range"
        );
        chain.push(current);
        if parents[current] == usize::MAX {
            break;
        }
        current = parents[current];
        ensure!(
            chain.len() <= rest.len(),
            "{label} hierarchy contains a cycle"
        );
    }
    Ok(chain)
}

fn checked_inverse(matrix: Mat4, label: &str) -> Result<Mat4> {
    ensure!(matrix.is_finite(), "{label} is non-finite");
    let inverse = matrix.inverse();
    ensure!(
        inverse.is_finite(),
        "{label} is non-invertible or its inverse is non-finite"
    );
    Ok(inverse)
}

/// Rest-pose hips height in channel units: the parent chain accumulated
/// with rotations applied but ancestor scales ignored, because animation
/// translation keys are parent-local (see the module docs).
fn channel_units_height(node: usize, parents: &[usize], rest: &[NodeTrs]) -> Result<f32> {
    ensure!(
        node < rest.len() && parents.len() == rest.len(),
        "invalid rest hierarchy"
    );
    let mut p = rest[node].translation;
    let mut cur = parents[node];
    let mut steps = 0;
    while cur != usize::MAX {
        ensure!(cur < rest.len(), "invalid rest hierarchy parent");
        let t = &rest[cur];
        p = t.translation + t.rotation * p;
        cur = parents[cur];
        steps += 1;
        ensure!(steps <= rest.len(), "rest hierarchy contains a cycle");
    }
    Ok(p.y)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use glam::{Quat, Vec3};
    use pocket3d::anim::NodeTrs;

    use crate::Vrm1HumanBone;

    use super::*;

    fn nodes(rest: Vec<NodeTrs>, parents: Vec<usize>) -> GltfNodes {
        let mut children = vec![Vec::new(); rest.len()];
        for (node, &parent) in parents.iter().enumerate() {
            if parent != usize::MAX {
                children[parent].push(node);
            }
        }
        GltfNodes {
            names: (0..rest.len()).map(|i| format!("node{i}")).collect(),
            parents,
            rest,
            children,
        }
    }

    fn skeleton(rest: Vec<NodeTrs>, parents: Vec<usize>) -> Skeleton {
        Skeleton {
            order: (0..rest.len()).collect(),
            parents,
            rest,
        }
    }

    fn vrm1_humanoid(entries: &[(Vrm1HumanBone, usize)]) -> Vrm1Humanoid {
        Vrm1Humanoid {
            human_bones: entries.iter().copied().collect::<BTreeMap<_, _>>(),
        }
    }

    fn rotation_channel(node: usize, values: [f32; 4]) -> Channel {
        Channel {
            node,
            path: ChannelPath::Rotation,
            interpolation: Interpolation::Linear,
            times: vec![0.0],
            values: values.to_vec(),
        }
    }

    fn approx_quat(actual: Quat, expected: Quat) {
        assert!(
            actual.normalize().dot(expected.normalize()).abs() > 1.0 - 1.0e-5,
            "{actual:?} != {expected:?}"
        );
    }

    fn json_glb(json: &str) -> Vec<u8> {
        let mut payload = json.as_bytes().to_vec();
        while !payload.len().is_multiple_of(4) {
            payload.push(b' ');
        }
        let total = 12 + 8 + payload.len();
        let mut glb = Vec::with_capacity(total);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total as u32).to_le_bytes());
        glb.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&payload);
        glb
    }

    /// Synthetic rig: vrma hips at rest height 2.0, model hips at 1.0 —
    /// hips translation must scale by 0.5 and re-anchor X/Z to the first key.
    #[test]
    fn retarget_scales_and_reanchors_hips() {
        let vrma = VrmaDoc {
            name: "test".into(),
            duration: 1.0,
            humanoid: vec![("hips".into(), 1), ("spine".into(), 2)],
            channels: vec![
                Channel {
                    node: 1,
                    path: ChannelPath::Translation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: vec![1.0, 2.0, 3.0, 2.0, 2.2, 4.0],
                },
                Channel {
                    node: 2,
                    path: ChannelPath::Rotation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: vec![
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        std::f32::consts::FRAC_1_SQRT_2,
                        0.0,
                        std::f32::consts::FRAC_1_SQRT_2,
                    ],
                },
                Channel {
                    // Not in the humanoid map: must be dropped.
                    node: 3,
                    path: ChannelPath::Rotation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0],
                    values: vec![0.0, 0.0, 0.0, 1.0],
                },
            ],
            nodes: {
                let mut rest = vec![NodeTrs::IDENTITY; 3];
                rest[1].translation = Vec3::new(0.0, 2.0, 0.0);
                GltfNodes {
                    names: vec!["root".into(), "hips".into(), "spine".into()],
                    parents: vec![usize::MAX, 0, 1],
                    rest,
                    children: vec![vec![1], vec![2], vec![]],
                }
            },
        };
        let mut rest = vec![NodeTrs::IDENTITY; 2];
        rest[1].translation = Vec3::new(0.0, 1.0, 0.0);
        let model = Skeleton {
            parents: vec![usize::MAX, 0],
            rest,
            order: vec![0, 1],
        };
        let humanoid = vec![("hips".to_string(), 1usize), ("spine".to_string(), 0)];
        let clip = retarget(&vrma, &humanoid, &model).unwrap();
        assert_eq!(clip.channels.len(), 2);
        let t = clip
            .channels
            .iter()
            .find(|c| c.path == ChannelPath::Translation)
            .unwrap();
        assert_eq!(t.node, 1);
        // Scaled by 0.5, X/Z re-anchored to key 0, then the +Z→-Z facing
        // flip negates X/Z.
        assert_eq!(t.values, vec![-0.0, 1.0, -0.0, -0.5, 1.1, -0.5]);
        let r = clip
            .channels
            .iter()
            .find(|c| c.path == ChannelPath::Rotation)
            .unwrap();
        assert_eq!(r.node, 0); // "spine" mapped onto model node 0
    }

    #[test]
    fn vrm0_quaternion_conversion_is_exactly_preserved() {
        let q = [0.1, 0.2, 0.3, 0.9];
        let vrma = VrmaDoc {
            name: "legacy".into(),
            duration: 0.0,
            humanoid: vec![("hips".into(), 0)],
            channels: vec![rotation_channel(0, q)],
            nodes: nodes(vec![NodeTrs::IDENTITY], vec![usize::MAX]),
        };
        let model = skeleton(
            vec![NodeTrs {
                translation: Vec3::Y,
                ..NodeTrs::IDENTITY
            }],
            vec![usize::MAX],
        );

        let clip = retarget(&vrma, &[("hips".into(), 0)], &model).unwrap();
        assert_eq!(clip.channels[0].values, vec![-0.1, 0.2, -0.3, 0.9]);
    }

    #[test]
    fn vrm1_semantic_mapping_targets_the_typed_skeleton_node() {
        let vrma = VrmaDoc {
            name: "mapped".into(),
            duration: 0.0,
            humanoid: vec![("spine".into(), 0)],
            channels: vec![rotation_channel(0, Quat::IDENTITY.to_array())],
            nodes: nodes(vec![NodeTrs::IDENTITY], vec![usize::MAX]),
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Spine, 2)]);
        let model = skeleton(vec![NodeTrs::IDENTITY; 3], vec![usize::MAX; 3]);

        let clip = retarget_with_humanoid(&vrma, HumanoidView::Vrm1(&humanoid), &model).unwrap();
        assert_eq!(clip.channels[0].node, 2);
    }

    #[test]
    fn vrm1_identity_rest_basis_passes_through_normalized_rotation() {
        let q = Quat::from_rotation_x(0.37) * Quat::from_rotation_z(-0.21);
        let vrma = VrmaDoc {
            name: "identity".into(),
            duration: 0.0,
            humanoid: vec![("spine".into(), 0)],
            channels: vec![rotation_channel(0, q.to_array())],
            nodes: nodes(vec![NodeTrs::IDENTITY], vec![usize::MAX]),
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Spine, 0)]);
        let model = skeleton(vec![NodeTrs::IDENTITY], vec![usize::MAX]);

        let clip = retarget_with_humanoid(&vrma, HumanoidView::Vrm1(&humanoid), &model).unwrap();
        approx_quat(
            Quat::from_array(clip.channels[0].values[..4].try_into().unwrap()),
            q,
        );
    }

    #[test]
    fn vrm1_non_identity_rest_basis_converts_local_rotation() {
        let source_parent = Quat::from_rotation_y(0.31);
        let source_local = Quat::from_rotation_x(-0.22);
        let target_parent = Quat::from_rotation_z(-0.19);
        let target_local = Quat::from_rotation_y(0.47);
        let animated = Quat::from_rotation_x(0.53) * Quat::from_rotation_z(0.17);
        let vrma = VrmaDoc {
            name: "based".into(),
            duration: 0.0,
            humanoid: vec![("spine".into(), 1)],
            channels: vec![rotation_channel(1, animated.to_array())],
            nodes: nodes(
                vec![
                    NodeTrs {
                        rotation: source_parent,
                        ..NodeTrs::IDENTITY
                    },
                    NodeTrs {
                        rotation: source_local,
                        ..NodeTrs::IDENTITY
                    },
                ],
                vec![usize::MAX, 0],
            ),
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Spine, 1)]);
        let model = skeleton(
            vec![
                NodeTrs {
                    rotation: target_parent,
                    ..NodeTrs::IDENTITY
                },
                NodeTrs {
                    rotation: target_local,
                    ..NodeTrs::IDENTITY
                },
            ],
            vec![usize::MAX, 0],
        );

        let clip = retarget_with_humanoid(&vrma, HumanoidView::Vrm1(&humanoid), &model).unwrap();
        let source_world = (source_parent * source_local).normalize();
        let target_world = (target_parent * target_local).normalize();
        let normalized =
            (source_world * source_local.inverse() * animated * source_world.inverse()).normalize();
        let expected =
            (target_local * target_world.inverse() * normalized * target_world).normalize();
        approx_quat(
            Quat::from_array(clip.channels[0].values[..4].try_into().unwrap()),
            expected,
        );
        assert!(
            Quat::from_array(clip.channels[0].values[..4].try_into().unwrap())
                .angle_between(animated)
                > 0.05
        );
    }

    #[test]
    fn same_normalized_motion_survives_different_vrm1_rest_bases() {
        let normalized = Quat::from_rotation_y(0.42) * Quat::from_rotation_x(-0.18);
        let source_parent_a = Quat::IDENTITY;
        let source_local_a = Quat::IDENTITY;
        let source_parent_b = Quat::from_rotation_z(0.29);
        let source_local_b = Quat::from_rotation_x(-0.34);
        let target_parent_a = Quat::IDENTITY;
        let target_local_a = Quat::IDENTITY;
        let target_parent_b = Quat::from_rotation_y(-0.23);
        let target_local_b = Quat::from_rotation_z(0.38);

        let source_animation = |parent: Quat, local: Quat| {
            let world = (parent * local).normalize();
            (local * world.inverse() * normalized * world).normalize()
        };
        let target_normalized = |output: Quat, parent: Quat, local: Quat| {
            let world = (parent * local).normalize();
            (world * local.inverse() * output * world.inverse()).normalize()
        };
        let make_vrma = |parent: Quat, local: Quat| VrmaDoc {
            name: "same-motion".into(),
            duration: 0.0,
            humanoid: vec![("spine".into(), 1)],
            channels: vec![rotation_channel(
                1,
                source_animation(parent, local).to_array(),
            )],
            nodes: nodes(
                vec![
                    NodeTrs {
                        rotation: parent,
                        ..NodeTrs::IDENTITY
                    },
                    NodeTrs {
                        rotation: local,
                        ..NodeTrs::IDENTITY
                    },
                ],
                vec![usize::MAX, 0],
            ),
        };
        let make_model = |parent: Quat, local: Quat| {
            skeleton(
                vec![
                    NodeTrs {
                        rotation: parent,
                        ..NodeTrs::IDENTITY
                    },
                    NodeTrs {
                        rotation: local,
                        ..NodeTrs::IDENTITY
                    },
                ],
                vec![usize::MAX, 0],
            )
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Spine, 1)]);
        let clip_a = retarget_with_humanoid(
            &make_vrma(source_parent_a, source_local_a),
            HumanoidView::Vrm1(&humanoid),
            &make_model(target_parent_a, target_local_a),
        );
        let source_b = make_vrma(source_parent_b, source_local_b);
        let model_b = make_model(target_parent_b, target_local_b);
        let clip_b =
            retarget_with_humanoid(&source_b, HumanoidView::Vrm1(&humanoid), &model_b).unwrap();
        let output_a =
            Quat::from_array(clip_a.unwrap().channels[0].values[..4].try_into().unwrap());
        let output_b = Quat::from_array(clip_b.channels[0].values[..4].try_into().unwrap());
        approx_quat(
            target_normalized(output_a, target_parent_a, target_local_a),
            target_normalized(output_b, target_parent_b, target_local_b),
        );
        approx_quat(
            target_normalized(output_b, target_parent_b, target_local_b),
            normalized,
        );
    }

    #[test]
    fn vrm1_hips_parent_rotation_and_authored_xz_are_preserved() {
        let vrma = VrmaDoc {
            name: "hips-rotation".into(),
            duration: 1.0,
            humanoid: vec![("hips".into(), 1)],
            channels: vec![Channel {
                node: 1,
                path: ChannelPath::Translation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![0.0, 2.0, 0.0, 1.0, 2.0, 1.0],
            }],
            nodes: nodes(
                vec![
                    NodeTrs {
                        rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
                        ..NodeTrs::IDENTITY
                    },
                    NodeTrs {
                        translation: Vec3::new(0.0, 2.0, 0.0),
                        ..NodeTrs::IDENTITY
                    },
                ],
                vec![usize::MAX, 0],
            ),
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Hips, 1)]);
        let model = skeleton(
            vec![
                NodeTrs {
                    rotation: Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2),
                    ..NodeTrs::IDENTITY
                },
                NodeTrs {
                    translation: Vec3::Y,
                    ..NodeTrs::IDENTITY
                },
            ],
            vec![usize::MAX, 0],
        );

        let clip = retarget_with_humanoid(&vrma, HumanoidView::Vrm1(&humanoid), &model).unwrap();
        let translation = &clip.channels[0];
        assert!((translation.values[3] + 0.5).abs() < 1.0e-5);
        assert!((translation.values[5] + 0.5).abs() < 1.0e-5);
    }

    #[test]
    fn vrm1_hips_parent_scale_is_applied_in_both_rest_spaces() {
        let vrma = VrmaDoc {
            name: "hips-scale".into(),
            duration: 1.0,
            humanoid: vec![("hips".into(), 1)],
            channels: vec![Channel {
                node: 1,
                path: ChannelPath::Translation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            }],
            nodes: nodes(
                vec![
                    NodeTrs {
                        scale: Vec3::new(2.0, 1.0, 1.0),
                        ..NodeTrs::IDENTITY
                    },
                    NodeTrs {
                        translation: Vec3::Y,
                        ..NodeTrs::IDENTITY
                    },
                ],
                vec![usize::MAX, 0],
            ),
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Hips, 1)]);
        let model = skeleton(
            vec![
                NodeTrs {
                    scale: Vec3::new(3.0, 1.0, 1.0),
                    ..NodeTrs::IDENTITY
                },
                NodeTrs {
                    translation: Vec3::Y,
                    ..NodeTrs::IDENTITY
                },
            ],
            vec![usize::MAX, 0],
        );

        let clip = retarget_with_humanoid(&vrma, HumanoidView::Vrm1(&humanoid), &model).unwrap();
        let values = &clip.channels[0].values;
        assert!((values[1] - 1.0).abs() < 1.0e-5);
        assert!((values[4] - 1.0).abs() < 1.0e-5);
        assert_eq!(values[0], 0.0);
        assert_eq!(values[2], 0.0);
        assert!((values[3] - 2.0 / 3.0).abs() < 1.0e-5);
    }

    #[test]
    fn invalid_target_bone_index_returns_an_error() {
        let vrma = VrmaDoc {
            name: "invalid-index".into(),
            duration: 0.0,
            humanoid: vec![("spine".into(), 0)],
            channels: vec![rotation_channel(0, Quat::IDENTITY.to_array())],
            nodes: nodes(vec![NodeTrs::IDENTITY], vec![usize::MAX]),
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Spine, 4)]);
        let model = skeleton(vec![NodeTrs::IDENTITY], vec![usize::MAX]);
        let error = match retarget_with_humanoid(&vrma, HumanoidView::Vrm1(&humanoid), &model) {
            Ok(_) => panic!("invalid target mapping unexpectedly succeeded"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("out of range"), "{error}");
    }

    #[test]
    fn non_finite_and_non_invertible_rest_transforms_fail_cleanly() {
        let source = VrmaDoc {
            name: "invalid-rest".into(),
            duration: 1.0,
            humanoid: vec![("hips".into(), 1)],
            channels: vec![Channel {
                node: 1,
                path: ChannelPath::Translation,
                interpolation: Interpolation::Linear,
                times: vec![0.0],
                values: vec![0.0, 1.0, 0.0],
            }],
            nodes: nodes(
                vec![
                    NodeTrs {
                        scale: Vec3::ZERO,
                        ..NodeTrs::IDENTITY
                    },
                    NodeTrs {
                        translation: Vec3::Y,
                        ..NodeTrs::IDENTITY
                    },
                ],
                vec![usize::MAX, 0],
            ),
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Hips, 1)]);
        let singular_model = skeleton(
            vec![
                NodeTrs {
                    scale: Vec3::ZERO,
                    ..NodeTrs::IDENTITY
                },
                NodeTrs {
                    translation: Vec3::Y,
                    ..NodeTrs::IDENTITY
                },
            ],
            vec![usize::MAX, 0],
        );
        assert!(
            retarget_with_humanoid(&source, HumanoidView::Vrm1(&humanoid), &singular_model)
                .is_err()
        );

        let mut non_finite_model = singular_model;
        non_finite_model.rest[0].scale = Vec3::ONE;
        non_finite_model.rest[0].rotation = Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0);
        assert!(
            retarget_with_humanoid(&source, HumanoidView::Vrm1(&humanoid), &non_finite_model)
                .is_err()
        );
    }

    #[test]
    fn small_uniform_rest_scale_remains_usable_for_vrm1_rotation_retargeting() {
        let source = VrmaDoc {
            name: "small-scale".into(),
            duration: 0.0,
            humanoid: vec![("spine".into(), 1)],
            channels: vec![rotation_channel(1, Quat::from_rotation_y(0.2).to_array())],
            nodes: nodes(
                vec![
                    NodeTrs {
                        scale: Vec3::splat(0.001),
                        ..NodeTrs::IDENTITY
                    },
                    NodeTrs::IDENTITY,
                ],
                vec![usize::MAX, 0],
            ),
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Spine, 1)]);
        let target = skeleton(
            vec![
                NodeTrs {
                    scale: Vec3::splat(0.001),
                    ..NodeTrs::IDENTITY
                },
                NodeTrs::IDENTITY,
            ],
            vec![usize::MAX, 0],
        );

        let clip = retarget_with_humanoid(&source, HumanoidView::Vrm1(&humanoid), &target)
            .expect("finite inverse for a valid 0.001 uniform scale");
        assert!(
            clip.channels[0]
                .values
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn missing_optional_humanoid_bones_drop_their_channels() {
        let vrma = VrmaDoc {
            name: "optional".into(),
            duration: 1.0,
            humanoid: vec![("hips".into(), 0), ("head".into(), 1)],
            channels: vec![
                Channel {
                    node: 0,
                    path: ChannelPath::Translation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0],
                    values: vec![0.0, 1.0, 0.0],
                },
                rotation_channel(1, Quat::IDENTITY.to_array()),
            ],
            nodes: nodes(
                vec![
                    NodeTrs {
                        translation: Vec3::Y,
                        ..NodeTrs::IDENTITY
                    },
                    NodeTrs::IDENTITY,
                ],
                vec![usize::MAX, 0],
            ),
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Hips, 0)]);
        let model = skeleton(
            vec![NodeTrs {
                translation: Vec3::Y,
                ..NodeTrs::IDENTITY
            }],
            vec![usize::MAX],
        );
        let clip = retarget_with_humanoid(&vrma, HumanoidView::Vrm1(&humanoid), &model).unwrap();
        assert_eq!(clip.channels.len(), 1);
        assert_eq!(clip.channels[0].path, ChannelPath::Translation);
    }

    #[test]
    fn minimal_vrm1_humanoid_receives_compatible_required_channel() {
        let vrma = VrmaDoc {
            name: "minimal".into(),
            duration: 0.0,
            humanoid: vec![("spine".into(), 0)],
            channels: vec![rotation_channel(0, Quat::IDENTITY.to_array())],
            nodes: nodes(vec![NodeTrs::IDENTITY], vec![usize::MAX]),
        };
        let humanoid = vrm1_humanoid(&[(Vrm1HumanBone::Spine, 0)]);
        let model = skeleton(vec![NodeTrs::IDENTITY], vec![usize::MAX]);
        assert!(retarget_with_humanoid(&vrma, HumanoidView::Vrm1(&humanoid), &model).is_ok());
    }

    #[test]
    fn vrma_rejects_unsupported_animation_spec_version() {
        let error = match load_vrma_bytes(&json_glb(
            r#"{"extensions":{"VRMC_vrm_animation":{"specVersion":"0.9"}}}"#,
        )) {
            Ok(_) => panic!("unsupported VRMA version unexpectedly succeeded"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("specVersion") && error.contains("1.0"),
            "{error}"
        );
    }
}
