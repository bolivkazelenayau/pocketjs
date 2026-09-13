//! Shared Verlet spring solver for VRM 0.x and VRM 1.0 spring bones.
//!
//! VRM 0.x is lowered with its historical descendant/leaf semantics. VRM 1.0
//! is lowered from its explicit head/tail pairs into the same particle solver.

use glam::{Mat4, Quat, Vec3};
use pocket3d::anim::{NodeTrs, Skeleton};

use crate::parse::SpringConfig;
use crate::vrm1::{Vrm1ColliderShape, Vrm1SpringBone};

const LEAF_TAIL_LEN: f32 = 0.07;
const MAX_SUB_DT: f32 = 1.0 / 30.0;
const MAX_SUBSTEPS: usize = 4;

#[derive(Clone, Copy)]
struct Params {
    stiffness: f32,
    drag: f32,
    gravity_dir: Vec3,
    gravity_power: f32,
    hit_radius: f32,
}
#[derive(Clone, Copy)]
enum ColliderShape {
    Sphere {
        offset: Vec3,
        radius: f32,
    },
    Capsule {
        offset: Vec3,
        tail: Vec3,
        radius: f32,
    },
}
#[derive(Clone, Copy)]
struct Collider {
    node: usize,
    shape: ColliderShape,
}
struct Joint {
    node: usize,
    parent: usize,
    params: Params,
    bone_axis: Vec3,
    length: f32,
    tail: Option<usize>,
    authored_rotation: Quat,
    center: Option<usize>,
    colliders: Vec<usize>,
    current_tail: Vec3,
    prev_tail: Vec3,
}
#[derive(Clone, Copy)]
enum WorldCollider {
    Sphere { center: Vec3, radius: f32 },
    Capsule { start: Vec3, end: Vec3, radius: f32 },
}

/// A source-neutral, fully resolved spring runtime. JSON lookup and hierarchy
/// resolution happen in the constructors; `step` only reads this runtime.
pub struct SpringSolver {
    vrm1: bool,
    joints: Vec<Joint>,
    colliders: Vec<Collider>,
    globals: Vec<Mat4>,
    world_rot: Vec<Quat>,
    animated_rot: Vec<Quat>,
    collider_world: Vec<WorldCollider>,
}

impl SpringSolver {
    /// Lower legacy VRM 0.x without changing its permissive parser semantics.
    pub fn new(config: &SpringConfig, skeleton: &Skeleton, initial_locals: &[NodeTrs]) -> Self {
        let n = skeleton.rest.len();
        assert_eq!(initial_locals.len(), n, "initial_locals length != skeleton");
        let mut children = vec![Vec::new(); n];
        for &i in &skeleton.order {
            if skeleton.parents[i] != usize::MAX {
                children[skeleton.parents[i]].push(i);
            }
        }
        let mut globals = vec![Mat4::IDENTITY; n];
        compute_globals(
            skeleton,
            initial_locals,
            Mat4::IDENTITY,
            &mut globals,
            &mut vec![Quat::IDENTITY; n],
        );
        let mut colliders = Vec::new();
        let mut joints = Vec::new();
        for (group_index, group) in config.bone_groups.iter().enumerate() {
            let mut group_colliders = Vec::new();
            for &cgi in &group.collider_group_indices {
                let Some(cg) = config.collider_groups.get(cgi) else {
                    log::warn!("spring group {group_index}: collider group {cgi} out of range");
                    continue;
                };
                if cg.node >= n {
                    log::warn!("collider group {cgi}: node {} out of range", cg.node);
                    continue;
                }
                for sphere in &cg.spheres {
                    let i = colliders.len();
                    colliders.push(Collider {
                        node: cg.node,
                        shape: ColliderShape::Sphere {
                            offset: sphere.offset,
                            radius: sphere.radius,
                        },
                    });
                    group_colliders.push(i);
                }
            }
            let params = Params {
                stiffness: group.stiffness,
                drag: group.drag_force.clamp(0.0, 1.0),
                gravity_dir: group.gravity_dir.try_normalize().unwrap_or(Vec3::NEG_Y),
                gravity_power: group.gravity_power,
                hit_radius: group.hit_radius,
            };
            let mut stack = Vec::new();
            for &root in group.bones.iter().rev() {
                if root < n {
                    stack.push(root)
                } else {
                    log::warn!("spring group {group_index}: root bone {root} out of range")
                }
            }
            while let Some(node) = stack.pop() {
                for &child in children[node].iter().rev() {
                    stack.push(child);
                }
                let tail_local = children[node]
                    .first()
                    .map(|&c| initial_locals[c].translation)
                    .unwrap_or_else(|| leaf_tail(skeleton, &globals, node));
                let Some(axis) = tail_local.try_normalize() else {
                    log::warn!("spring joint at node {node} has a zero-length tail; skipped");
                    continue;
                };
                let tail = globals[node].transform_point3(tail_local);
                joints.push(Joint {
                    node,
                    parent: skeleton.parents[node],
                    params,
                    bone_axis: axis,
                    length: tail_local.length(),
                    tail: None,
                    authored_rotation: initial_locals[node].rotation,
                    center: None,
                    colliders: group_colliders.clone(),
                    current_tail: tail,
                    prev_tail: tail,
                });
            }
        }
        let m = colliders.len();
        let j = joints.len();
        Self {
            vrm1: false,
            joints,
            colliders,
            globals,
            world_rot: vec![Quat::IDENTITY; n],
            animated_rot: vec![Quat::IDENTITY; j],
            collider_world: vec![
                WorldCollider::Sphere {
                    center: Vec3::ZERO,
                    radius: 0.0
                };
                m
            ],
        }
    }

    /// Resolve a VRM 1.0 springBone document into explicit head/tail joints.
    /// A returned `None` means there is no usable spring runtime (including the
    /// all-springs-downgraded case); a valid one-joint spring returns a no-op
    /// solver so it still represents a successfully resolved feature.
    pub fn new_vrm1(
        config: &Vrm1SpringBone,
        skeleton: &Skeleton,
        initial_locals: &[NodeTrs],
    ) -> Option<Self> {
        let n = skeleton.rest.len();
        if initial_locals.len() != n {
            return None;
        }
        let mut globals = vec![Mat4::IDENTITY; n];
        let mut world_rot = vec![Quat::IDENTITY; n];
        compute_globals(
            skeleton,
            initial_locals,
            Mat4::IDENTITY,
            &mut globals,
            &mut world_rot,
        );
        let path_ok = |node: usize| valid_path(node, skeleton, initial_locals, &globals);
        let mut colliders = Vec::with_capacity(config.colliders.len());
        let mut collider_ok = Vec::with_capacity(config.colliders.len());
        for c in &config.colliders {
            let shape = match &c.shape {
                Vrm1ColliderShape::Sphere(s) => ColliderShape::Sphere {
                    offset: s.offset,
                    radius: s.radius,
                },
                Vrm1ColliderShape::Capsule(s) => ColliderShape::Capsule {
                    offset: s.offset,
                    tail: s.tail,
                    radius: s.radius,
                },
            };
            let good = path_ok(c.node)
                && mat_is_finite(globals[c.node])
                && mat_is_finite(globals[c.node].inverse())
                && match shape {
                    ColliderShape::Sphere { offset, .. } => {
                        globals[c.node].transform_point3(offset).is_finite()
                    }
                    ColliderShape::Capsule { offset, tail, .. } => {
                        globals[c.node].transform_point3(offset).is_finite()
                            && globals[c.node].transform_point3(tail).is_finite()
                    }
                };
            collider_ok.push(good);
            colliders.push(Collider {
                node: c.node,
                shape,
            });
        }
        let mut joints = Vec::new();
        let mut any_valid = false;
        for (si, spring) in config.springs.iter().enumerate() {
            let center_ok = spring
                .center
                .map(|c| {
                    path_ok(c) && mat_is_finite(globals[c]) && mat_is_finite(globals[c].inverse())
                })
                .unwrap_or(true);
            let mut spring_ok = center_ok;
            let mut spring_colliders = Vec::new();
            for &group_index in &spring.collider_groups {
                let Some(group) = config.collider_groups.get(group_index) else {
                    spring_ok = false;
                    continue;
                };
                for &ci in &group.colliders {
                    if !collider_ok[ci] {
                        spring_ok = false;
                    }
                    spring_colliders.push(ci);
                }
            }
            let mut spring_joints = Vec::new();
            for pair in spring.joints.windows(2) {
                let head = pair[0].node;
                let tail = pair[1].node;
                if !path_ok(head)
                    || !path_ok(tail)
                    || !mat_is_finite(globals[head])
                    || !mat_is_finite(globals[head].inverse())
                {
                    spring_ok = false;
                    continue;
                }
                let hp = globals[head].transform_point3(Vec3::ZERO);
                let tp = globals[tail].transform_point3(Vec3::ZERO);
                let delta = tp - hp;
                let Some(axis) = globals[head]
                    .inverse()
                    .transform_vector3(delta)
                    .try_normalize()
                else {
                    spring_ok = false;
                    continue;
                };
                let length = delta.length();
                if !length.is_finite() || length <= 0.0 {
                    spring_ok = false;
                    continue;
                }
                if !spring_ok {
                    continue;
                }
                let j = pair[0].clone();
                let params = Params {
                    stiffness: j.stiffness,
                    drag: j.drag_force,
                    gravity_dir: j.gravity_dir,
                    gravity_power: j.gravity_power,
                    hit_radius: j.hit_radius,
                };
                let tail_history = spring
                    .center
                    .map(|c| globals[c].inverse().transform_point3(tp))
                    .unwrap_or(tp);
                if !tail_history.is_finite() {
                    spring_ok = false;
                    continue;
                }
                spring_joints.push(Joint {
                    node: head,
                    parent: skeleton.parents[head],
                    params,
                    bone_axis: axis,
                    length,
                    tail: Some(tail),
                    authored_rotation: initial_locals[head].rotation,
                    center: spring.center,
                    colliders: spring_colliders.clone(),
                    current_tail: tail_history,
                    prev_tail: tail_history,
                });
            }
            if spring_ok {
                any_valid = true;
                joints.extend(spring_joints);
            } else {
                log::warn!(
                    "VRM1 spring {si} was downgraded because a participating transform or collider is unusable"
                );
            }
        }
        if !any_valid {
            if !config.springs.is_empty() {
                log::warn!("all VRM1 springs were downgraded; spring capability is unavailable");
            }
            return None;
        }
        // Separate springs can control ancestor-related heads. Stable depth
        // ordering makes those dependencies deterministic without changing the
        // legacy VRM0 traversal order.
        joints.sort_by_key(|joint| node_depth(joint.node, &skeleton.parents));
        let m = colliders.len();
        let j = joints.len();
        Some(Self {
            vrm1: true,
            joints,
            colliders,
            globals,
            world_rot,
            animated_rot: vec![Quat::IDENTITY; j],
            collider_world: vec![
                WorldCollider::Sphere {
                    center: Vec3::ZERO,
                    radius: 0.0
                };
                m
            ],
        })
    }

    pub fn reset(&mut self, skeleton: &Skeleton, locals: &[NodeTrs], root_transform: Mat4) {
        let root = if self.vrm1 {
            Mat4::IDENTITY
        } else {
            root_transform
        };
        compute_globals(
            skeleton,
            locals,
            root,
            &mut self.globals,
            &mut self.world_rot,
        );
        for j in &mut self.joints {
            let tail = j
                .tail
                .map(|node| self.globals[node].transform_point3(Vec3::ZERO))
                .unwrap_or_else(|| self.globals[j.node].transform_point3(j.bone_axis * j.length));
            let tail = if let Some(c) = j.center {
                self.globals[c].inverse().transform_point3(tail)
            } else {
                tail
            };
            j.current_tail = tail;
            j.prev_tail = tail;
        }
    }

    pub fn step(
        &mut self,
        dt: f32,
        skeleton: &Skeleton,
        locals: &mut [NodeTrs],
        root_transform: Mat4,
    ) {
        if self.joints.is_empty() || !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let root = if self.vrm1 {
            Mat4::IDENTITY
        } else {
            root_transform
        };
        let (_, root_rot, _) = root.to_scale_rotation_translation();
        let substeps = ((dt / MAX_SUB_DT).ceil() as usize).clamp(1, MAX_SUBSTEPS);
        let sub_dt = (dt / substeps as f32).min(MAX_SUB_DT);
        for (i, j) in self.joints.iter().enumerate() {
            self.animated_rot[i] = locals[j.node].rotation;
        }
        for _ in 0..substeps {
            compute_globals(
                skeleton,
                locals,
                root,
                &mut self.globals,
                &mut self.world_rot,
            );
            if !self.vrm1 {
                self.refresh_colliders();
            }
            for i in 0..self.joints.len() {
                if self.vrm1 {
                    compute_globals(
                        skeleton,
                        locals,
                        root,
                        &mut self.globals,
                        &mut self.world_rot,
                    );
                    self.refresh_colliders();
                }
                let (node, parent, params, axis, length, authored, center, current, previous) = {
                    let j = &self.joints[i];
                    (
                        j.node,
                        j.parent,
                        j.params,
                        j.bone_axis,
                        j.length,
                        j.authored_rotation,
                        j.center,
                        j.current_tail,
                        j.prev_tail,
                    )
                };
                let (parent_mat, parent_rot) = if parent == usize::MAX {
                    (root, root_rot)
                } else {
                    (self.globals[parent], self.world_rot[parent])
                };
                let head = parent_mat.transform_point3(locals[node].translation);
                let target_rot = if self.vrm1 {
                    authored
                } else {
                    self.animated_rot[i]
                };
                let rest_dir = (parent_rot * target_rot * axis).normalize();
                let (mut current, mut previous) = (current, previous);
                if let Some(c) = center {
                    let center_mat = self.globals[c];
                    current = center_mat.transform_point3(current);
                    previous = center_mat.transform_point3(previous);
                }
                let mut next = current
                    + (current - previous) * (1.0 - params.drag)
                    + rest_dir * (params.stiffness * sub_dt)
                    + params.gravity_dir * (params.gravity_power * sub_dt);
                next = head + (next - head).try_normalize().unwrap_or(rest_dir) * length;
                for &ci in &self.joints[i].colliders {
                    let shape = self.collider_world[ci];
                    let radius = params.hit_radius + world_radius(shape);
                    if let Some(pushed) = push_out(next, shape, radius, rest_dir) {
                        next = head + (pushed - head).try_normalize().unwrap_or(rest_dir) * length;
                    }
                }
                let (stored_current, stored_previous) = if let Some(c) = center {
                    let inv = self.globals[c].inverse();
                    (inv.transform_point3(next), inv.transform_point3(current))
                } else {
                    (next, current)
                };
                let tail_dir = (next - head).try_normalize().unwrap_or(rest_dir);
                let world = (Quat::from_rotation_arc(rest_dir, tail_dir)
                    * (parent_rot * target_rot))
                    .normalize();
                let local = (parent_rot.inverse() * world).normalize();
                {
                    let j = &mut self.joints[i];
                    j.prev_tail = stored_previous;
                    j.current_tail = stored_current;
                }
                locals[node].rotation = local;
                if self.vrm1 {
                    compute_globals(
                        skeleton,
                        locals,
                        root,
                        &mut self.globals,
                        &mut self.world_rot,
                    )
                } else {
                    self.globals[node] = parent_mat * locals[node].matrix();
                    self.world_rot[node] = world;
                }
            }
        }
    }

    pub fn joint_count(&self) -> usize {
        self.joints.len()
    }
    fn refresh_colliders(&mut self) {
        for i in 0..self.colliders.len() {
            let c = self.colliders[i];
            let g = self.globals[c.node];
            self.collider_world[i] = match c.shape {
                ColliderShape::Sphere { offset, radius } => WorldCollider::Sphere {
                    center: g.transform_point3(offset),
                    radius,
                },
                ColliderShape::Capsule {
                    offset,
                    tail,
                    radius,
                } => WorldCollider::Capsule {
                    start: g.transform_point3(offset),
                    end: g.transform_point3(tail),
                    radius,
                },
            };
        }
    }
}

fn valid_path(mut node: usize, skeleton: &Skeleton, locals: &[NodeTrs], globals: &[Mat4]) -> bool {
    loop {
        let t = locals[node];
        if !t.translation.is_finite()
            || !t.rotation.is_finite()
            || !t.scale.is_finite()
            || t.scale.x <= 0.0
            || t.scale.y <= 0.0
            || t.scale.z <= 0.0
            || !is_positive_uniform_scale(t.scale)
            || !mat_is_finite(globals[node])
            || !mat_is_finite(globals[node].inverse())
        {
            return false;
        }
        let p = skeleton.parents[node];
        if p == usize::MAX {
            break;
        }
        node = p;
    }
    true
}
fn node_depth(mut node: usize, parents: &[usize]) -> usize {
    let mut depth = 0;
    while parents[node] != usize::MAX {
        depth += 1;
        node = parents[node];
    }
    depth
}
fn is_positive_uniform_scale(scale: Vec3) -> bool {
    if !scale.is_finite() || scale.min_element() <= 0.0 {
        return false;
    }
    let tolerance = 8.0 * f32::EPSILON * scale.abs().max_element();
    (scale.x - scale.y).abs() <= tolerance && (scale.y - scale.z).abs() <= tolerance
}
fn mat_is_finite(m: Mat4) -> bool {
    m.to_cols_array().iter().all(|v| v.is_finite())
}
fn compute_globals(
    skeleton: &Skeleton,
    locals: &[NodeTrs],
    root: Mat4,
    globals: &mut [Mat4],
    world_rot: &mut [Quat],
) {
    let (_, rr, _) = root.to_scale_rotation_translation();
    for &i in &skeleton.order {
        let p = skeleton.parents[i];
        globals[i] = if p == usize::MAX {
            root * locals[i].matrix()
        } else {
            globals[p] * locals[i].matrix()
        };
        world_rot[i] = if p == usize::MAX {
            (rr * locals[i].rotation).normalize()
        } else {
            (world_rot[p] * locals[i].rotation).normalize()
        };
    }
}
fn leaf_tail(skeleton: &Skeleton, globals: &[Mat4], node: usize) -> Vec3 {
    let head = globals[node].transform_point3(Vec3::ZERO);
    let p = skeleton.parents[node];
    let dir = if p == usize::MAX {
        Vec3::NEG_Y
    } else {
        (head - globals[p].transform_point3(Vec3::ZERO))
            .try_normalize()
            .unwrap_or(Vec3::NEG_Y)
    };
    globals[node]
        .inverse()
        .transform_point3(head + dir * LEAF_TAIL_LEN)
}
fn world_radius(c: WorldCollider) -> f32 {
    match c {
        WorldCollider::Sphere { radius, .. } | WorldCollider::Capsule { radius, .. } => radius,
    }
}
fn push_out(point: Vec3, c: WorldCollider, radius: f32, fallback: Vec3) -> Option<Vec3> {
    let (closest, delta) = match c {
        WorldCollider::Sphere { center, .. } => (center, point - center),
        WorldCollider::Capsule { start, end, .. } => {
            let segment = end - start;
            let d = segment.length_squared();
            if d <= 0.0 {
                (start, point - start)
            } else {
                let t = ((point - start).dot(segment) / d).clamp(0.0, 1.0);
                let p = start + segment * t;
                (p, point - p)
            }
        }
    };
    let d2 = delta.length_squared();
    if !d2.is_finite() || d2 >= radius * radius {
        return None;
    }
    let n = delta.try_normalize().unwrap_or(fallback);
    Some(closest + n * radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{ColliderGroup, SphereCollider, SpringGroup};
    fn chain() -> (Skeleton, SpringConfig) {
        let mut r = vec![NodeTrs::IDENTITY; 3];
        r[1].translation = Vec3::new(0.1, 0., 0.);
        r[2].translation = Vec3::new(0.1, 0., 0.);
        (
            Skeleton {
                parents: vec![usize::MAX, 0, 1],
                rest: r.clone(),
                order: vec![0, 1, 2],
            },
            SpringConfig {
                bone_groups: vec![SpringGroup {
                    comment: "t".into(),
                    stiffness: 1.,
                    gravity_power: 1.,
                    gravity_dir: Vec3::NEG_Y,
                    drag_force: 0.4,
                    hit_radius: 0.01,
                    bones: vec![1],
                    collider_group_indices: vec![],
                }],
                collider_groups: vec![],
            },
        )
    }
    #[test]
    fn vrm0_subtree_leaf_count() {
        let (s, c) = chain();
        let x = SpringSolver::new(&c, &s, &s.rest);
        assert_eq!(x.joint_count(), 2);
        assert!((x.joints[1].length - 0.07).abs() < 1e-6)
    }
    #[test]
    fn gravity_and_reset() {
        let (s, c) = chain();
        let mut x = SpringSolver::new(&c, &s, &s.rest);
        let mut l = s.rest.clone();
        for _ in 0..120 {
            x.step(1. / 60., &s, &mut l, Mat4::IDENTITY);
            l.copy_from_slice(&s.rest);
        }
        assert!(x.joints[0].current_tail.is_finite());
        x.reset(&s, &s.rest, Mat4::IDENTITY);
        assert_eq!(x.joints[0].current_tail, x.joints[0].prev_tail)
    }
    #[test]
    fn nonfinite_dt_is_noop() {
        let (s, c) = chain();
        let mut x = SpringSolver::new(&c, &s, &s.rest);
        let mut l = s.rest.clone();
        x.step(f32::INFINITY, &s, &mut l, Mat4::IDENTITY);
        assert_eq!(l[1].rotation, s.rest[1].rotation);
        assert_eq!(l[2].translation, s.rest[2].translation)
    }
    #[test]
    fn capsule_helper_handles_side_end_and_degenerate() {
        let c = WorldCollider::Capsule {
            start: Vec3::ZERO,
            end: Vec3::X,
            radius: 0.2,
        };
        assert!(push_out(Vec3::new(0.1, 0.1, 0.), c, 0.3, Vec3::Y).is_some());
        assert!(push_out(Vec3::new(1.1, 0., 0.), c, 0.3, Vec3::X).is_some());
        let d = WorldCollider::Capsule {
            start: Vec3::ZERO,
            end: Vec3::ZERO,
            radius: 0.2,
        };
        assert!(push_out(Vec3::new(0.1, 0., 0.), d, 0.3, Vec3::X).is_some())
    }
    #[test]
    fn legacy_collider_is_sphere() {
        let (s, mut c) = chain();
        c.collider_groups.push(ColliderGroup {
            node: 0,
            spheres: vec![SphereCollider {
                offset: Vec3::ZERO,
                radius: 0.1,
            }],
        });
        c.bone_groups[0].collider_group_indices = vec![0];
        let x = SpringSolver::new(&c, &s, &s.rest);
        assert_eq!(x.colliders.len(), 1)
    }
}

#[cfg(test)]
mod vrm1_tests {
    use super::*;
    use crate::vrm1::{Vrm1Spring, Vrm1SpringBone, Vrm1SpringJoint};

    fn sparse_chain() -> (Skeleton, Vrm1SpringBone) {
        let mut rest = vec![NodeTrs::IDENTITY; 4];
        rest[1].translation = Vec3::X;
        rest[2].translation = Vec3::X;
        rest[3].translation = Vec3::X;
        let skeleton = Skeleton {
            parents: vec![usize::MAX, 0, 1, 2],
            rest: rest.clone(),
            order: vec![0, 1, 2, 3],
        };
        let joints = |nodes: &[usize]| {
            nodes
                .iter()
                .map(|&node| Vrm1SpringJoint {
                    node,
                    hit_radius: 0.0,
                    stiffness: 1.0,
                    gravity_power: 0.0,
                    gravity_dir: Vec3::ZERO,
                    drag_force: 0.5,
                })
                .collect()
        };
        (
            skeleton,
            Vrm1SpringBone {
                colliders: Vec::new(),
                collider_groups: Vec::new(),
                springs: vec![Vrm1Spring {
                    name: None,
                    center: None,
                    collider_groups: Vec::new(),
                    joints: joints(&[1, 3]),
                }],
            },
        )
    }

    #[test]
    fn explicit_sparse_pair_has_scaled_rest_length() {
        let (mut skeleton, config) = sparse_chain();
        skeleton.rest[0].scale = Vec3::splat(2.0);
        let solver = SpringSolver::new_vrm1(&config, &skeleton, &skeleton.rest).unwrap();
        assert_eq!(solver.joint_count(), 1);
        assert!((solver.joints[0].length - 4.0).abs() < 1.0e-5);
    }

    #[test]
    fn final_explicit_joint_is_tail_only_and_rest_stays_still() {
        let (skeleton, mut config) = sparse_chain();
        config.springs[0].joints = vec![
            Vrm1SpringJoint {
                node: 1,
                hit_radius: 0.0,
                stiffness: 1.0,
                gravity_power: 0.0,
                gravity_dir: Vec3::ZERO,
                drag_force: 0.5,
            },
            Vrm1SpringJoint {
                node: 2,
                hit_radius: 0.0,
                stiffness: 1.0,
                gravity_power: 0.0,
                gravity_dir: Vec3::ZERO,
                drag_force: 0.5,
            },
            Vrm1SpringJoint {
                node: 3,
                hit_radius: 99.0,
                stiffness: 99.0,
                gravity_power: 99.0,
                gravity_dir: Vec3::Y,
                drag_force: 0.0,
            },
        ];
        let mut solver = SpringSolver::new_vrm1(&config, &skeleton, &skeleton.rest).unwrap();
        assert_eq!(solver.joint_count(), 2);
        let mut locals = skeleton.rest.clone();
        let final_rotation = locals[3].rotation;
        solver.step(1.0 / 60.0, &skeleton, &mut locals, Mat4::IDENTITY);
        assert_eq!(locals[3].rotation, final_rotation);
        assert_eq!(locals[1].rotation, Quat::IDENTITY);
        assert_eq!(locals[2].rotation, Quat::IDENTITY);
    }

    #[test]
    fn vrm1_presentation_root_is_not_used_by_physics() {
        let (skeleton, config) = sparse_chain();
        let mut a = SpringSolver::new_vrm1(&config, &skeleton, &skeleton.rest).unwrap();
        let mut b = SpringSolver::new_vrm1(&config, &skeleton, &skeleton.rest).unwrap();
        let mut locals_a = skeleton.rest.clone();
        let mut locals_b = skeleton.rest.clone();
        a.step(1.0 / 30.0, &skeleton, &mut locals_a, Mat4::IDENTITY);
        b.step(
            1.0 / 30.0,
            &skeleton,
            &mut locals_b,
            Mat4::from_rotation_y(core::f32::consts::PI),
        );
        assert_eq!(locals_a[1].rotation, locals_b[1].rotation);
        assert_eq!(a.joints[0].current_tail, b.joints[0].current_tail);
    }

    #[test]
    fn unusable_vrm1_scale_downgrades_the_spring() {
        let (mut skeleton, config) = sparse_chain();
        skeleton.rest[0].scale = Vec3::new(1.0, 2.0, 1.0);
        assert!(SpringSolver::new_vrm1(&config, &skeleton, &skeleton.rest).is_none());
    }

    #[test]
    fn slightly_nonuniform_float_scale_remains_usable() {
        let (mut skeleton, config) = sparse_chain();
        skeleton.rest[0].scale = Vec3::new(1.0, 1.0 + 4.0e-7, 1.0);
        assert!(SpringSolver::new_vrm1(&config, &skeleton, &skeleton.rest).is_some());
    }

    #[test]
    fn vrm1_cross_spring_dependencies_are_ancestry_ordered() {
        let mut rest = vec![NodeTrs::IDENTITY; 5];
        for local in rest.iter_mut().skip(1) {
            local.translation = Vec3::X;
        }
        let skeleton = Skeleton {
            parents: vec![usize::MAX, 0, 1, 2, 3],
            rest: rest.clone(),
            order: vec![0, 1, 2, 3, 4],
        };
        let joint = |node| Vrm1SpringJoint {
            node,
            hit_radius: 0.0,
            stiffness: 1.0,
            gravity_power: 0.0,
            gravity_dir: Vec3::ZERO,
            drag_force: 0.5,
        };
        let config = Vrm1SpringBone {
            colliders: Vec::new(),
            collider_groups: Vec::new(),
            springs: vec![
                Vrm1Spring {
                    name: None,
                    center: None,
                    collider_groups: Vec::new(),
                    joints: vec![joint(3), joint(4)],
                },
                Vrm1Spring {
                    name: None,
                    center: None,
                    collider_groups: Vec::new(),
                    joints: vec![joint(1), joint(2)],
                },
            ],
        };
        let solver = SpringSolver::new_vrm1(&config, &skeleton, &rest).unwrap();
        assert_eq!(
            solver
                .joints
                .iter()
                .map(|joint| joint.node)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn reset_reseats_scaled_explicit_tail() {
        let (mut skeleton, config) = sparse_chain();
        skeleton.rest[0].scale = Vec3::splat(2.0);
        let mut solver = SpringSolver::new_vrm1(&config, &skeleton, &skeleton.rest).unwrap();
        solver.reset(&skeleton, &skeleton.rest, Mat4::from_rotation_y(1.0));
        assert!((solver.joints[0].current_tail.x - 6.0).abs() < 1.0e-5);
        assert_eq!(solver.joints[0].current_tail, solver.joints[0].prev_tail);
    }

    #[test]
    fn unusable_pair_drops_the_entire_vrm1_spring() {
        let (skeleton, config) = sparse_chain();
        let mut locals = skeleton.rest.clone();
        locals[3].translation.x = f32::NAN;
        assert!(SpringSolver::new_vrm1(&config, &skeleton, &locals).is_none());
    }
}
