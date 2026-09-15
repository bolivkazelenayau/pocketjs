#[test]
fn mask_coverage_uses_authored_alpha_while_final_alpha_policy_stays_intact() {
    // A visible authored MASK texel would have failed the previous tinted test.
    let texture_alpha = 0.8_f32;
    let base_factor_alpha = 0.75_f32;
    let tint_alpha = 0.25_f32;
    let cutoff = 0.5_f32;
    let authored_alpha = texture_alpha * base_factor_alpha;
    assert!(authored_alpha >= cutoff);
    assert!(authored_alpha * tint_alpha < cutoff);

    let shader = include_str!("../src/shaders/model.wgsl");
    assert!(shader.contains(
        "let material_albedo = textureSample(t_albedo, s_albedo, in.uv)\n        * material.base_color_factor;"
    ));
    assert!(shader.contains("var albedo = material_albedo * instance.tint;"));
    let cutoff_test = shader
        .lines()
        .find(|line| line.trim_start().starts_with("if alpha_cutoff > 0.0"))
        .expect("fallback shader must alpha-test MASK surfaces");
    assert!(cutoff_test.contains("material_albedo.a < alpha_cutoff"));
    assert!(shader.contains("let alpha = select(1.0, albedo.a, material.params.w > 0.5);"));
}

fn mask_quad(alpha: f32) -> Vec<u8> {
    use serde_json::json;

    let positions = [
        [-1.0_f32, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let mut bin = Vec::new();
    for position in positions {
        for component in position {
            bin.extend_from_slice(&component.to_le_bytes());
        }
    }
    for index in [0_u16, 1, 2, 0, 2, 3] {
        bin.extend_from_slice(&index.to_le_bytes());
    }
    let mut json = serde_json::to_vec(&json!({
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "materials": [{
            "alphaMode": "MASK",
            "alphaCutoff": 0.5,
            "doubleSided": true,
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, alpha]}
        }],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 12}
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [-1.0, -1.0, 0.0], "max": [1.0, 1.0, 0.0]},
            {"bufferView": 1, "componentType": 5123, "count": 6, "type": "SCALAR"}
        ]
    }))
    .unwrap();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let length = 12 + 8 + json.len() + 8 + bin.len();
    let mut glb = Vec::with_capacity(length);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2_u32.to_le_bytes());
    glb.extend_from_slice(&(length as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);
    glb
}

#[test]
fn mask_quad_keeps_coverage_when_instance_tint_alpha_changes() {
    use pocket3d::gpu::{Gpu, OFFSCREEN_FORMAT, OffscreenTarget};
    use pocket3d::hud::Hud;
    use pocket3d::prelude::{Camera, ModelAsset, ModelInstance, Renderer, Scene, Vec3};

    let gpu = Gpu::new_headless().expect("headless GPU required for cutout regression");
    let mut renderer = Renderer::new(&gpu, OFFSCREEN_FORMAT).unwrap();
    let target = OffscreenTarget::new(&gpu, 64, 64);
    let camera = Camera {
        pos: Vec3::new(0.0, 0.0, 3.0),
        znear: 0.01,
        ..Camera::default()
    };
    let mut coverage = |authored_alpha: f32, tint_alpha: f32| -> usize {
        let asset = ModelAsset::load_glb_bytes(
            &gpu,
            &renderer.model_material_layout,
            &renderer.samplers,
            &mask_quad(authored_alpha),
            "mask coverage quad",
        )
        .unwrap();
        let mut instance = ModelInstance::new(asset);
        instance.tint[3] = tint_alpha;
        let scene = Scene {
            transparent_clear: true,
            models: vec![instance],
            ..Scene::default()
        };
        renderer.render(
            &gpu,
            &target.view,
            target.size,
            &scene,
            &camera,
            &Hud::default(),
        );
        target
            .read_rgba(&gpu)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[3] != 0)
            .count()
    };
    let normal = coverage(0.75, 1.0);
    let tinted = coverage(0.75, 0.25);
    let discarded = coverage(0.25, 1.0);
    assert!(
        normal > 100,
        "quad must cover enough pixels to make the regression meaningful"
    );
    assert_eq!(
        tinted, normal,
        "presentation alpha must not change MASK coverage"
    );
    assert_eq!(
        discarded, 0,
        "authored alpha below cutoff must discard every fragment"
    );
}
