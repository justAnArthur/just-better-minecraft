//! Task 2.12's screenshots: the playground world streamed by a real
//! server, meshed by the real client pipeline, rendered offscreen with
//! each debug view on, saved under `target/2.12-shots/`. Not goldens —
//! the deliverable is legible PNGs the report lists; the views'
//! correctness is asserted in jbm-client's `debug_views` tests.

use glam::DVec3;
use jbm_client::camera::Camera;
use jbm_client::gpu::Gpu;
use jbm_client::mesh::appearance::{BlockAppearance, builtin_textures};
use jbm_client::render::lines::LineVertex;
use jbm_client::render::offscreen::OffscreenTarget;
use jbm_client::render::renderer::{DepthTarget, FrameInputs, Renderer};
use jbm_client::render::terrain::{TerrainPass, lod_tint_color};
use jbm_client::render::texture_array::BlockTextures;
use jbm_client::terrain_sync::TerrainSync;
use jbm_client::testing::screenshot::gpu_or_skip;
use jbm_client::world::ClientWorld;
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::protocol::ServerMsg;
use jbm_server::worldgen::Preset;
use jbm_server::{DEFAULT_SEED, ServerConfig};
use jbm_testkit::HeadlessServer;

/// The playground scene's server: small deterministic view, inline
/// generation, two LOD levels so the tint shot has more than one color.
#[cfg(not(feature = "cube16"))]
fn server_config() -> ServerConfig {
    ServerConfig {
        worldgen: Preset::Playground,
        seed: DEFAULT_SEED,
        streaming: true,
        view_radius: 4,
        vertical_radius: 2,
        send_budget_per_tick: 128,
        gen_threads: 0,
        gen_budget_per_tick: 256,
        max_in_flight: 2048,
        world_dir: None,
        save_interval: std::time::Duration::from_secs(5),
        lod_levels: 2,
        lod_view_radius: 3,
        lod_vertical_radius: 2,
        lod_send_budget_per_tick: 128,
        lod_max_in_flight: 512,
        ..ServerConfig::default()
    }
}

/// Where the shots go: the workspace `target/2.12-shots`.
fn shot_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target")
        .join("2.12-shots")
}

/// The chunk-border line vertices, the same rule the app builds them
/// with: twelve edges per resident cube within a radius of 3 per level,
/// colored by the tint palette.
#[cfg(not(feature = "cube16"))]
fn border_vertices(terrain: &TerrainPass, camera: &Camera) -> Vec<LineVertex> {
    /// Cubes of each level around the camera that draw borders.
    const BORDER_RADIUS: i64 = 3;
    let (camera_cube, camera_frac) = jbm_client::render::terrain::camera_split(camera.position);
    let mut vertices = Vec::new();
    for key in terrain.resident_keys() {
        let stride = (jbm_core::coords::CUBE_SIZE as i64) << key.level;
        let center = [
            (camera.position.x / stride as f64).floor() as i64,
            (camera.position.y / stride as f64).floor() as i64,
            (camera.position.z / stride as f64).floor() as i64,
        ];
        let far = [
            i64::from(key.pos.x) - center[0],
            i64::from(key.pos.y) - center[1],
            i64::from(key.pos.z) - center[2],
        ]
        .iter()
        .map(|d| d.abs())
        .max()
        .unwrap_or(i64::MAX);
        if far > BORDER_RADIUS {
            continue;
        }
        let min = jbm_client::render::terrain::cube_origin(key, camera_cube, camera_frac);
        let size = jbm_client::render::terrain::space::cube_size(key.level);
        let max = min + size;
        let [r, g, b] = lod_tint_color(key.level);
        push_box(&mut vertices, min, max, [r, g, b, 0.8]);
    }
    vertices
}

/// The frozen cull frustum as its corner box, re-based into the real
/// camera's space (the app's rule).
#[cfg(not(feature = "cube16"))]
fn frozen_frustum_vertices(terrain: &TerrainPass, camera: &Camera) -> Vec<LineVertex> {
    /// The far cutoff of the drawn frustum, in blocks.
    const FAR_CUTOFF_DEPTH: f32 = 1.0 / 2048.0;
    let Some((frozen_pos, view_proj)) = terrain.frozen_view() else {
        return Vec::new();
    };
    let inverse = view_proj.inverse();
    let rebased = |ndc: glam::Vec4| -> glam::Vec3 {
        let un = inverse * ndc;
        (un.truncate() / un.w) + (frozen_pos - camera.position).as_vec3()
    };
    let near = [
        rebased(glam::Vec4::new(-1.0, -1.0, 1.0, 1.0)),
        rebased(glam::Vec4::new(1.0, -1.0, 1.0, 1.0)),
        rebased(glam::Vec4::new(1.0, 1.0, 1.0, 1.0)),
        rebased(glam::Vec4::new(-1.0, 1.0, 1.0, 1.0)),
    ];
    let far = [
        rebased(glam::Vec4::new(-1.0, -1.0, FAR_CUTOFF_DEPTH, 1.0)),
        rebased(glam::Vec4::new(1.0, -1.0, FAR_CUTOFF_DEPTH, 1.0)),
        rebased(glam::Vec4::new(1.0, 1.0, FAR_CUTOFF_DEPTH, 1.0)),
        rebased(glam::Vec4::new(-1.0, 1.0, FAR_CUTOFF_DEPTH, 1.0)),
    ];
    let color = [1.0, 0.25, 0.2, 0.9];
    let mut vertices = Vec::new();
    for ring in [near, far] {
        for i in 0..4 {
            segment(&mut vertices, ring[i], ring[(i + 1) % 4], color);
        }
    }
    for i in 0..4 {
        segment(&mut vertices, near[i], far[i], color);
    }
    vertices
}

fn push_box(vertices: &mut Vec<LineVertex>, min: glam::Vec3, max: glam::Vec3, color: [f32; 4]) {
    let (a, b) = (min, max);
    let corners = [
        glam::Vec3::new(a.x, a.y, a.z),
        glam::Vec3::new(b.x, a.y, a.z),
        glam::Vec3::new(b.x, b.y, a.z),
        glam::Vec3::new(a.x, b.y, a.z),
        glam::Vec3::new(a.x, a.y, b.z),
        glam::Vec3::new(b.x, a.y, b.z),
        glam::Vec3::new(b.x, b.y, b.z),
        glam::Vec3::new(a.x, b.y, b.z),
    ];
    for i in 0..4 {
        segment(vertices, corners[i], corners[(i + 1) % 4], color);
        segment(vertices, corners[i + 4], corners[(i + 1) % 4 + 4], color);
        segment(vertices, corners[i], corners[i + 4], color);
    }
}

fn segment(vertices: &mut Vec<LineVertex>, from: glam::Vec3, to: glam::Vec3, color: [f32; 4]) {
    let vertex = |p: glam::Vec3| LineVertex {
        pos: [p.x, p.y, p.z, 0.0],
        color,
    };
    vertices.push(vertex(from));
    vertices.push(vertex(to));
}

/// One shared render context.
struct Ctx {
    gpu: Gpu,
    target: OffscreenTarget,
    depth: DepthTarget,
    renderer: Renderer,
    terrain: TerrainPass,
}

impl Ctx {
    fn new(gpu: Gpu) -> Ctx {
        let target = OffscreenTarget::new(&gpu.device, 512, 512);
        let depth = DepthTarget::new(&gpu.device, 512, 512);
        let mut renderer = Renderer::new(&gpu, target.format());
        renderer.set_viewport(512, 512);
        let (layers, _) = builtin_textures();
        let textures = BlockTextures::new(&gpu, &layers).expect("texture array");
        let terrain = TerrainPass::new(&gpu, target.format(), renderer.uniform_layout(), &textures);
        Ctx {
            gpu,
            target,
            depth,
            renderer,
            terrain,
        }
    }

    fn render(&mut self, camera: &Camera) -> image::RgbaImage {
        self.renderer.render(
            &self.gpu,
            FrameInputs {
                target: &self.target.create_view(),
                depth: &self.depth,
                camera,
                terrain: Some(&mut self.terrain),
                outline: None,
                overlay: None,
            },
        );
        self.target.read_rgba(&self.gpu)
    }

    fn save(&self, name: &str, image: &image::RgbaImage) {
        let path = shot_dir().join(format!("{name}.png"));
        image
            .save(&path)
            .unwrap_or_else(|e| panic!("save {path:?}: {e}"));
        println!("shot: {}", path.display());
    }
}

/// Streams the playground into a settled client world (quiet: no cube
/// traffic for 20 ticks) and meshes it through the real pipeline.
#[cfg(not(feature = "cube16"))]
fn settled_scene(ctx: &mut Ctx, appearance: &BlockAppearance) -> DVec3 {
    let mut server = HeadlessServer::with_config(server_config());
    let mut client = server.connect("viewer");
    let mut world = ClientWorld::new();
    let mut spawn = None;
    let mut quiet = 0;
    for _ in 0..600 {
        server.step(1);
        let messages = client.drain();
        let busy = messages.iter().any(|m| {
            matches!(
                m,
                ServerMsg::CubeData { .. } | ServerMsg::LodCubeData { .. }
            )
        });
        for msg in messages {
            if let ServerMsg::Welcome { spawn: s, .. } = msg {
                spawn = Some(DVec3::from(s));
            }
            world.apply(&msg);
        }
        quiet = if busy { 0 } else { quiet + 1 };
        if quiet >= 20 && spawn.is_some() {
            break;
        }
    }
    let spawn = spawn.expect("no Welcome");
    assert!(quiet >= 20, "the playground stream never settled");
    let mut sync = TerrainSync::with_world(world, std::sync::Arc::new(appearance.clone()), 0);
    sync.settle_sync(&ctx.gpu, &mut ctx.terrain, appearance, spawn);
    spawn
}

/// Renders the six task 2.12 screenshots. The assertions are structural
/// (files land, the views change the frame): the pixel-level contracts
/// live in jbm-client's debug_views tests, and the goldens elsewhere.
#[test]
#[cfg(not(feature = "cube16"))]
fn playground_debug_view_screenshots() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    std::fs::create_dir_all(shot_dir()).expect("create shot dir");
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    let appearance = BlockAppearance::new(&registry);
    let _ = blocks;
    let mut ctx = Ctx::new(gpu);
    let spawn = settled_scene(&mut ctx, &appearance);

    // The overview camera: high southwest of spawn, looking northeast
    // across the feature band (the plane, cliff, staircase, pool).
    let overview = Camera::looking_at(
        spawn + DVec3::new(-140.0, 110.0, 190.0),
        DVec3::new(60.0, 66.0, 10.0),
        70.0_f32.to_radians(),
        1.0,
    );

    // 1. The plain overview.
    let plain = ctx.render(&overview);
    ctx.save("playground_overview", &plain);

    // 2. Wireframe over the shaded terrain.
    ctx.renderer.settings_mut().wireframe = true;
    let wired = ctx.render(&overview);
    ctx.renderer.settings_mut().wireframe = false;
    ctx.save("wireframe", &wired);
    assert!(
        wired
            .pixels()
            .zip(plain.pixels())
            .filter(|(a, b)| a != b)
            .count()
            > 500
    );

    // 3. Chunk borders.
    let borders = border_vertices(&ctx.terrain, &overview);
    assert!(!borders.is_empty(), "no resident cubes near the camera");
    ctx.renderer.set_debug_lines(&ctx.gpu, &borders);
    let bordered = ctx.render(&overview);
    ctx.renderer.set_debug_lines(&ctx.gpu, &[]);
    ctx.save("chunk_borders", &bordered);
    assert!(
        bordered
            .pixels()
            .zip(plain.pixels())
            .filter(|(a, b)| a != b)
            .count()
            > 200
    );

    // 4. Exploded by 4.
    ctx.renderer.settings_mut().explode = 4;
    let exploded = ctx.render(&overview);
    ctx.renderer.settings_mut().explode = 0;
    ctx.save("exploded", &exploded);
    assert!(
        exploded
            .pixels()
            .zip(plain.pixels())
            .filter(|(a, b)| a != b)
            .count()
            > 500
    );

    // 5. Level tint with level 0 hidden.
    ctx.renderer.settings_mut().lod_tint = true;
    ctx.renderer.settings_mut().level_mask = 0x7E;
    let tinted = ctx.render(&overview);
    ctx.renderer.settings_mut().lod_tint = false;
    ctx.renderer.settings_mut().level_mask = 0x7F;
    ctx.save("tint_level0_hidden", &tinted);
    assert!(
        tinted
            .pixels()
            .zip(plain.pixels())
            .filter(|(a, b)| a != b)
            .count()
            > 500
    );

    // 6. The frozen frustum, seen from outside: freeze at the overview
    // camera, then step far east and look back — the cull keeps the
    // overview's set and the frustum box draws where it was captured.
    ctx.renderer.settings_mut().freeze_cull = true;
    let _ = ctx.render(&overview); // captures the frozen view
    let outside = Camera::looking_at(
        spawn + DVec3::new(600.0, 160.0, 500.0),
        DVec3::new(0.0, 66.0, 20.0),
        70.0_f32.to_radians(),
        1.0,
    );
    let frustum_lines = frozen_frustum_vertices(&ctx.terrain, &outside);
    assert!(!frustum_lines.is_empty(), "no frozen view captured");
    ctx.renderer.set_debug_lines(&ctx.gpu, &frustum_lines);
    let frozen = ctx.render(&outside);
    ctx.renderer.set_debug_lines(&ctx.gpu, &[]);
    ctx.renderer.settings_mut().freeze_cull = false;
    ctx.save("frozen_frustum_outside", &frozen);
    // The frozen set still drew (the cull kept the overview's cubes).
    assert!(ctx.terrain.read_cull_stats_blocking(&ctx.gpu).drawn() > 0);
}
