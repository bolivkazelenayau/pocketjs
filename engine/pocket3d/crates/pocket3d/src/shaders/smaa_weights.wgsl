// Adapted from SMAA / iryoku/smaa (MIT). See ../../../../../../THIRD_PARTY_NOTICES.md.

@group(0) @binding(0) var edges_tex: texture_2d<f32>;
@group(0) @binding(1) var edges_sampler: sampler;
@group(0) @binding(2) var area_tex: texture_2d<f32>;
@group(0) @binding(3) var area_sampler: sampler;
@group(0) @binding(4) var search_tex: texture_2d<f32>;
@group(0) @binding(5) var search_sampler: sampler;
@group(0) @binding(6) var<uniform> weights_metrics: Metrics;

fn sample_edges(uv: vec2f) -> vec2f {
    return textureSampleLevel(edges_tex, edges_sampler, uv, 0.0).rg;
}

fn search_length(e: vec2f, offset: f32) -> f32 {
    // SearchTex is 64x16, packed from the canonical 66x33 source table.
    let uv = vec2f(
        0.5 * e.x + 0.0078125 + 1.03125 * offset,
        -2.0 * e.y + 2.03125,
    );
    return textureSampleLevel(search_tex, search_sampler, uv, 0.0).r;
}

fn search_x_left(start: vec2f, end: f32) -> f32 {
    var coord = start;
    var e = vec2f(0.0, 1.0);
    for (var i = 0u; i < 8u; i = i + 1u) {
        e = sample_edges(coord);
        coord -= weights_metrics.texel_size * vec2f(2.0, 0.0);
        if (!(coord.x > end && e.y > 0.8281 && e.x == 0.0)) {
            break;
        }
    }
    let correction = -(255.0 / 127.0) * search_length(e, 0.0) + 3.25;
    return coord.x + weights_metrics.texel_size.x * correction;
}

fn search_x_right(start: vec2f, end: f32) -> f32 {
    var coord = start;
    var e = vec2f(0.0, 1.0);
    for (var i = 0u; i < 8u; i = i + 1u) {
        e = sample_edges(coord);
        coord += weights_metrics.texel_size * vec2f(2.0, 0.0);
        if (!(coord.x < end && e.y > 0.8281 && e.x == 0.0)) {
            break;
        }
    }
    let correction = -(255.0 / 127.0) * search_length(e, 0.5) + 3.25;
    return coord.x - weights_metrics.texel_size.x * correction;
}

fn search_y_up(start: vec2f, end: f32) -> f32 {
    var coord = start;
    var e = vec2f(1.0, 0.0);
    for (var i = 0u; i < 8u; i = i + 1u) {
        e = sample_edges(coord);
        coord -= weights_metrics.texel_size * vec2f(0.0, 2.0);
        if (!(coord.y > end && e.x > 0.8281 && e.y == 0.0)) {
            break;
        }
    }
    let correction = -(255.0 / 127.0) * search_length(e.yx, 0.0) + 3.25;
    return coord.y + weights_metrics.texel_size.y * correction;
}

fn search_y_down(start: vec2f, end: f32) -> f32 {
    var coord = start;
    var e = vec2f(1.0, 0.0);
    for (var i = 0u; i < 8u; i = i + 1u) {
        e = sample_edges(coord);
        coord += weights_metrics.texel_size * vec2f(0.0, 2.0);
        if (!(coord.y < end && e.x > 0.8281 && e.y == 0.0)) {
            break;
        }
    }
    let correction = -(255.0 / 127.0) * search_length(e.yx, 0.5) + 3.25;
    return coord.y - weights_metrics.texel_size.y * correction;
}

fn area_ortho_weights(dist: vec2f, e1: f32, e2: f32) -> vec2f {
    let coord = vec2f(16.0) * round(4.0 * vec2f(e1, e2)) + dist;
    let uv = (coord + vec2f(0.5)) / vec2f(160.0, 560.0);
    return textureSampleLevel(area_tex, area_sampler, uv, 0.0).rg;
}

fn area_diag_weights(dist: vec2f, e1: f32, e2: f32) -> vec2f {
    let coord = vec2f(20.0) * vec2f(e1, e2) + dist;
    let uv = (vec2f(80.0, 0.0) + coord + vec2f(0.5)) / vec2f(160.0, 560.0);
    return textureSampleLevel(area_tex, area_sampler, uv, 0.0).rg;
}

fn decode_diag_bilinear_access(e: vec2f) -> vec2f {
    var decoded = e;
    decoded.x *= abs(5.0 * decoded.x - 3.75);
    return round(decoded);
}

fn decode_diag_bilinear_access4(e: vec4f) -> vec4f {
    let rb = e.rb * abs(5.0 * e.rb - vec2f(3.75));
    return round(vec4f(rb.x, e.y, rb.y, e.w));
}

struct DiagSearchResult {
    distance: vec2f,
    end: vec2f,
};

fn search_diag1(start: vec2f, dir: vec2f) -> DiagSearchResult {
    var coord = vec4f(start, -1.0, 1.0);
    var end = vec2f(0.0);
    let t = weights_metrics.texel_size;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (!(coord.z < 7.0 && coord.w > 0.9)) {
            break;
        }
        coord.x += t.x * dir.x;
        coord.y += t.y * dir.y;
        coord.z += 1.0;
        end = sample_edges(coord.xy);
        coord.w = dot(end, vec2f(0.5));
    }
    return DiagSearchResult(coord.zw, end);
}

fn search_diag2(start: vec2f, dir: vec2f) -> DiagSearchResult {
    var coord = vec4f(start, -1.0, 1.0);
    var end = vec2f(0.0);
    coord.x += 0.25 * weights_metrics.texel_size.x;
    let t = weights_metrics.texel_size;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (!(coord.z < 7.0 && coord.w > 0.9)) {
            break;
        }
        coord.x += t.x * dir.x;
        coord.y += t.y * dir.y;
        coord.z += 1.0;
        end = decode_diag_bilinear_access(sample_edges(coord.xy));
        coord.w = dot(end, vec2f(0.5));
    }
    return DiagSearchResult(coord.zw, end);
}

fn calculate_diag_weights(uv: vec2f, edge: vec2f) -> vec2f {
    let t = weights_metrics.texel_size;
    var weights = vec2f(0.0);
    var d = vec4f(0.0);

    if (edge.x > 0.0) {
        let result = search_diag1(uv, vec2f(-1.0, 1.0));
        d.x = result.distance.x;
        d.z = result.distance.y;
        d.x += select(0.0, 1.0, result.end.y > 0.9);
    }
    let diag1_right = search_diag1(uv, vec2f(1.0, -1.0));
    d.y = diag1_right.distance.x;
    d.w = diag1_right.distance.y;

    if (d.x + d.y > 2.0) {
        let coords_left = uv + t * vec2f(-d.x + 0.25, d.x);
        let coords_right = uv + t * vec2f(d.y, -d.y - 0.25);
        let left = sample_edges(coords_left - t * vec2f(1.0, 0.0));
        let right = sample_edges(coords_right + t * vec2f(1.0, 0.0));
        let decoded = decode_diag_bilinear_access4(vec4f(left, right));
        let c = vec4f(decoded.y, decoded.x, decoded.w, decoded.z);
        var crossing = 2.0 * c.xz + c.yw;
        crossing *= vec2f(1.0) - step(vec2f(0.9), d.zw);
        weights += area_diag_weights(d.xy, crossing.x, crossing.y);
    }

    let diag2_left = search_diag2(uv, vec2f(-1.0, -1.0));
    d.x = diag2_left.distance.x;
    d.z = diag2_left.distance.y;
    if (sample_edges(uv + t * vec2f(1.0, 0.0)).x > 0.0) {
        let result = search_diag2(uv, vec2f(1.0, 1.0));
        d.y = result.distance.x;
        d.w = result.distance.y;
        d.y += select(0.0, 1.0, result.end.y > 0.9);
    } else {
        d.y = 0.0;
        d.w = 0.0;
    }

    if (d.x + d.y > 2.0) {
        let coords_left = uv - t * vec2f(d.x, d.x);
        let coords_right = uv + t * vec2f(d.y, d.y);
        let left = sample_edges(coords_left - t * vec2f(1.0, 0.0)).g;
        let left_top = sample_edges(coords_left - t * vec2f(0.0, 1.0)).x;
        let right = sample_edges(coords_right + t * vec2f(1.0, 0.0)).gr;
        var crossing = 2.0 * vec2f(left, right.x) + vec2f(left_top, right.y);
        crossing *= vec2f(1.0) - step(vec2f(0.9), d.zw);
        weights += area_diag_weights(d.xy, crossing.x, crossing.y).yx;
    }

    return weights;
}

fn detect_horizontal_corner_pattern(
    weights: vec2f,
    left_coord: vec2f,
    right_coord: vec2f,
    distances: vec2f,
) -> vec2f {
    let left_right = step(distances, distances.yx);
    let rounding = 0.75 * left_right / max(1.0, left_right.x + left_right.y);
    var factor = vec2f(1.0);
    factor.x -= rounding.x * sample_edges(left_coord + weights_metrics.texel_size * vec2f(0.0, 1.0)).x;
    factor.x -= rounding.y * sample_edges(right_coord + weights_metrics.texel_size * vec2f(1.0, 1.0)).x;
    factor.y -= rounding.x * sample_edges(left_coord + weights_metrics.texel_size * vec2f(0.0, -2.0)).x;
    factor.y -= rounding.y * sample_edges(right_coord + weights_metrics.texel_size * vec2f(1.0, -2.0)).x;
    return weights * clamp(factor, vec2f(0.0), vec2f(1.0));
}

fn detect_vertical_corner_pattern(
    weights: vec2f,
    top_coord: vec2f,
    bottom_coord: vec2f,
    distances: vec2f,
) -> vec2f {
    let left_right = step(distances, distances.yx);
    let rounding = 0.75 * left_right / max(1.0, left_right.x + left_right.y);
    var factor = vec2f(1.0);
    factor.x -= rounding.x * sample_edges(top_coord + weights_metrics.texel_size * vec2f(1.0, 0.0)).y;
    factor.x -= rounding.y * sample_edges(bottom_coord + weights_metrics.texel_size * vec2f(1.0, 1.0)).y;
    factor.y -= rounding.x * sample_edges(top_coord + weights_metrics.texel_size * vec2f(-2.0, 0.0)).y;
    factor.y -= rounding.y * sample_edges(bottom_coord + weights_metrics.texel_size * vec2f(-2.0, 1.0)).y;
    return weights * clamp(factor, vec2f(0.0), vec2f(1.0));
}

@fragment
fn fs_weights(in: VsOut) -> @location(0) vec4f {
    let t = weights_metrics.texel_size;
    let e = sample_edges(in.uv);
    let pix = in.uv / t;
    var weights = vec4f(0.0); // horizontal (left/right), vertical (top/bottom)

    if (e.y > 0.0) {
        let diagonal = calculate_diag_weights(in.uv, e);
        if (dot(diagonal, vec2f(1.0)) > 0.0) {
            // Diagonal patterns have priority over the orthogonal searches.
            return vec4f(diagonal, 0.0, 0.0);
        }
    }

    if (e.y > 0.0) {
        let left_start = in.uv + t * vec2f(-0.25, -0.125);
        let right_start = in.uv + t * vec2f(1.25, -0.125);
        let left_end = left_start.x - t.x * 16.0;
        let right_end = right_start.x + t.x * 16.0;
        let left_x = search_x_left(left_start, left_end);
        let right_x = search_x_right(right_start, right_end);
        let distances = abs(round(vec2f(left_x, right_x) / t.x - vec2f(pix.x)));
        let crossing_y = in.uv.y - 0.25 * t.y;
        let crossing_left = sample_edges(vec2f(left_x, crossing_y)).x;
        let crossing_right = sample_edges(vec2f(right_x + t.x, crossing_y)).x;
        let a = area_ortho_weights(sqrt(distances), crossing_left, crossing_right);
        let rounded = detect_horizontal_corner_pattern(
            a,
            vec2f(left_x, in.uv.y),
            vec2f(right_x, in.uv.y),
            distances,
        );
        weights.x = rounded.x;
        weights.y = rounded.y;
    }

    if (e.x > 0.0) {
        let top_start = in.uv + t * vec2f(-0.125, -0.25);
        let bottom_start = in.uv + t * vec2f(-0.125, 1.25);
        let top_end = top_start.y - t.y * 16.0;
        let bottom_end = bottom_start.y + t.y * 16.0;
        let top_y = search_y_up(top_start, top_end);
        let bottom_y = search_y_down(bottom_start, bottom_end);
        let distances = abs(round(vec2f(top_y, bottom_y) / t.y - vec2f(pix.y)));
        let crossing_x = in.uv.x - 0.25 * t.x;
        let crossing_top = sample_edges(vec2f(crossing_x, top_y)).y;
        let crossing_bottom = sample_edges(vec2f(crossing_x, bottom_y + t.y)).y;
        let a = area_ortho_weights(sqrt(distances), crossing_top, crossing_bottom);
        let rounded = detect_vertical_corner_pattern(
            a,
            vec2f(in.uv.x, top_y),
            vec2f(in.uv.x, bottom_y),
            distances,
        );
        weights.z = rounded.x;
        weights.w = rounded.y;
    }

    return clamp(weights, vec4f(0.0), vec4f(1.0));
}
