// Sky pass: a fullscreen triangle with no vertex buffers. The fragment
// shader reconstructs the view ray from the inverse view-projection and
// picks a color from the vertical angle, then (task 2.15) adds the sun's
// glow and disk and a procedural cloud layer.

struct FrameUniforms {
    view_proj: mat4x4f,
    inv_view_proj: mat4x4f,
    camera_pos_hi: vec4f,
    camera_pos_lo: vec4f,
    horizon_color: vec4f,
    zenith_color: vec4f,
    ground_color: vec4f,
    viewport: vec4f,
    // Task 2.15 (see FrameUniforms in render/renderer.rs).
    sun_dir: vec4f,        // xyz toward the sun, w = time (s)
    sun_color: vec4f,      // rgb sun radiance, w = sun lighting on
    ambient_sky: vec4f,    // rgb sky ambient, w = clouds on
    ambient_ground: vec4f, // rgb ground bounce, w = water shading on
    camera_wrap: vec4f,    // xyz camera position mod WRAP_PERIOD
};

@group(0) @binding(0) var<uniform> u: FrameUniforms;

struct VsOut {
    @builtin(position) pos: vec4f,
    // Clip-space x/y of this pixel, also its NDC after /w (w == 1).
    @location(0) ndc: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    // One triangle oversized to cover the screen.
    let p = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    var out: VsOut;
    out.pos = vec4f(p[i], 1.0, 1.0);
    out.ndc = p[i];
    return out;
}

// ---------------------------------------------------------------------------
// Task 2.15: the sun disk and the clouds.
//
// The sun disk is a soft-edged circle SUN_RADIUS_DEGREES wide around the
// sun direction, over the shared sun_glow halo (sky_gradient.wgsl).
//
// The clouds are a flat layer at world height CLOUD_HEIGHT: each sky pixel
// whose ray climbs toward it (camera below the layer, ray above the
// horizon) intersects the plane and reads a 2D value-noise fbm there. The
// noise is periodic with WRAP_PERIOD blocks (every octave's cell count
// divides it), and the plane coordinates start from camera_wrap (the
// camera position reduced mod WRAP_PERIOD in f64), so clouds stay sharp
// and seamless at any world coordinate. Deterministic: the only input
// besides the view is the time uniform (drift), which tests leave at 0.
// Clouds thin out toward the horizon (CLOUD_HORIZON_FADE_*) and are never
// drawn below it, and fade into the sky with distance.
const SUN_RADIUS_DEGREES: f32 = 1.3;
const SUN_DISK_COLOR: vec3f = vec3f(1.0, 0.97, 0.90);

const CLOUD_HEIGHT: f32 = 560.0;
const CLOUD_CELL: f32 = 256.0; // octave-0 noise cell, in blocks
const CLOUD_OCTAVES: i32 = 5;
const WRAP_PERIOD: f32 = 65536.0; // mirrored by WRAP_PERIOD in renderer.rs
const CLOUD_WIND: vec2f = vec2f(4.0, 1.5); // blocks per second
const CLOUD_COVER: f32 = 0.52; // fbm below this is clear sky
const CLOUD_SOFTNESS: f32 = 0.22; // fbm range from clear to fully dense
const CLOUD_HORIZON_FADE_START: f32 = 0.02; // ray y where clouds begin
const CLOUD_HORIZON_FADE_END: f32 = 0.25; // ray y where they are full
const CLOUD_FADE_DISTANCE: f32 = 9000.0; // blocks; e-folding into the sky
const CLOUD_LIT: vec3f = vec3f(1.0, 0.98, 0.95);
const CLOUD_SHADOW: vec3f = vec3f(0.62, 0.67, 0.76);
const CLOUD_MAX_OPACITY: f32 = 0.92;

// A 2D integer hash to [0, 1]; the cell wraps at `period` cells.
fn cloud_hash(cell: vec2i, period: i32) -> f32 {
    let c = vec2u(((cell % period) + period) % period);
    var h = c.x * 0x8da6b343u ^ c.y * 0xd8163841u ^ 0x9e3779b9u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h) * (1.0 / 4294967295.0);
}

// Smooth value noise, periodic with `period` cells.
fn cloud_noise(p: vec2f, period: i32) -> f32 {
    let cell = vec2i(floor(p));
    let f = p - floor(p);
    let w = f * f * (3.0 - 2.0 * f);
    let a = cloud_hash(cell, period);
    let b = cloud_hash(cell + vec2i(1, 0), period);
    let c = cloud_hash(cell + vec2i(0, 1), period);
    let d = cloud_hash(cell + vec2i(1, 1), period);
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

// The cloud fbm at plane coordinates `xz` (blocks), in [0, 1].
fn cloud_fbm(xz: vec2f) -> f32 {
    var p = xz / CLOUD_CELL;
    var period = i32(WRAP_PERIOD / CLOUD_CELL);
    var amplitude = 0.5;
    var sum = 0.0;
    var norm = 0.0;
    for (var i = 0; i < CLOUD_OCTAVES; i++) {
        sum += amplitude * cloud_noise(p, period);
        norm += amplitude;
        // Each octave doubles the frequency and the period (in cells), so
        // every octave still tiles WRAP_PERIOD blocks; the shift
        // decorrelates the octaves' lattices.
        p = p * 2.0 + vec2f(19.7, 7.3);
        period = period * 2;
        amplitude = amplitude * 0.5;
    }
    return sum / norm;
}

// Cloud color (rgb) and opacity (a) along `dir`, over the sky color `sky`.
fn clouds(dir: vec3f, sky: vec3f) -> vec4f {
    let camera_y = u.camera_pos_hi.y + u.camera_pos_lo.y;
    if (u.ambient_sky.w < 0.5 || dir.y <= CLOUD_HORIZON_FADE_START || camera_y >= CLOUD_HEIGHT) {
        return vec4f(0.0);
    }
    let t = (CLOUD_HEIGHT - camera_y) / dir.y;
    let drift = CLOUD_WIND * u.sun_dir.w;
    let xz = u.camera_wrap.xz + dir.xz * t + drift;
    let density = smoothstep(CLOUD_COVER, CLOUD_COVER + CLOUD_SOFTNESS, cloud_fbm(xz));
    if (density <= 0.0) {
        return vec4f(0.0);
    }
    // Self-shadowing: density a little toward the sun darkens the cloud
    // (the sun-facing edges stay bright, thick cores turn grey-blue).
    let toward_sun = normalize(u.sun_dir.xz + vec2f(1e-4, 0.0)) * 48.0;
    let shadow = smoothstep(CLOUD_COVER, CLOUD_COVER + 2.0 * CLOUD_SOFTNESS, cloud_fbm(xz + toward_sun));
    var color = mix(CLOUD_LIT, CLOUD_SHADOW, 0.75 * shadow);
    // Silver lining near the sun.
    color += sun_glow(dir, u.sun_dir.xyz, u.sun_color.rgb) * 1.5;
    // Distant clouds take on the sky color (aerial perspective) and the
    // layer thins toward the horizon.
    color = mix(color, sky, 1.0 - exp(-t / CLOUD_FADE_DISTANCE));
    let horizon = smoothstep(CLOUD_HORIZON_FADE_START, CLOUD_HORIZON_FADE_END, dir.y);
    return vec4f(color, density * horizon * CLOUD_MAX_OPACITY);
}

// The full sky color along `dir` (gradient, glow, sun disk, clouds).
fn sky_color(dir: vec3f) -> vec3f {
    let sun = u.sun_dir.xyz;
    var color = sky_gradient(dir, u.horizon_color.rgb, u.zenith_color.rgb, u.ground_color.rgb)
        + sun_glow(dir, sun, u.sun_color.rgb);
    if (dir.y > 0.0) {
        let c = dot(dir, sun);
        let r = radians(SUN_RADIUS_DEGREES);
        let disk = smoothstep(cos(r * 1.25), cos(r * 0.8), c);
        color = mix(color, SUN_DISK_COLOR, disk);
        let cloud = clouds(dir, color);
        color = mix(color, cloud.rgb, cloud.a);
    }
    return color;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    // A point on the near plane along this pixel's view ray; the matrix
    // is rotation-only (camera-relative), so normalizing it gives the
    // world-space ray direction.
    let near = u.inv_view_proj * vec4f(in.ndc, 1.0, 1.0);
    let dir = normalize(near.xyz / near.w);
    // The gradient and glow live in sky_gradient.wgsl, concatenated ahead
    // of this text at pipeline creation (task 2.9): terrain.wgsl's haze
    // evaluates the very same functions.
    return vec4f(sky_color(dir), 1.0);
}
