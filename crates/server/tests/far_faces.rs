//! Task 2.13 regression ("sand in the ocean", "broken chunks"): beyond
//! `camera_window_radius` a column streams only its surface span, and the
//! client mesher treats a missing neighbour cube as opaque — it culls any
//! face toward a cube it never received. So the streamed set must hold
//! every cube a visible face looks into: the air above a column's top
//! block (the sea surface sits on a cube's top row) and the cubes beside a
//! neighbouring cliff. This settles a real server on the overworld and
//! checks that no face of a sent cube borders an unsent cube it would be
//! visible against.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use glam::DVec3;
use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CUBE_SIZE, CubePos, LocalPos};
use jbm_core::cube::Cube;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, PlayerId, ServerMsg};
use jbm_server::DEFAULT_SEED;
use jbm_server::server::{Server, ServerConfig};
use jbm_server::terrain::{Generator, SEA_LEVEL};
use jbm_server::worldgen::Preset;
use jbm_testkit::builtin_blocks;

const TIMEOUT: Duration = Duration::from_secs(120);

/// Whether a face between `from` (inside a sent cube) and `to` (in an
/// unsent neighbour cube) would be drawn: anything against air, and a
/// solid block against water.
fn face_visible(blocks: &Blocks, from: BlockStateId, to: BlockStateId) -> bool {
    if from == BlockStateId::AIR {
        return false;
    }
    to == BlockStateId::AIR || (to == blocks.water && from != blocks.water)
}

#[test]
fn far_columns_stream_every_cube_their_faces_look_into() {
    let blocks = builtin_blocks();
    let generator = Generator::with_preset(DEFAULT_SEED, blocks, Preset::Overworld);
    let side = i64::from(CUBE_SIZE);

    // A camera on land six columns from the first all-ocean column along
    // +x: the view holds sea, coast and relief, all beyond the camera
    // window.
    let ocean_cx = (0..4000)
        .find(|&cx| generator.column_surface_bounds(cx, 0).1 < SEA_LEVEL)
        .expect("an ocean column on the overworld");
    let cam_x = i64::from(ocean_cx - 6) * side + side / 2;
    let cam_z = side / 2;
    let cam_y = generator.surface_height(cam_x, cam_z).max(SEA_LEVEL) + 2;

    let view_radius = 8;
    let window_r = 2;
    let mut server = Server::with_config(ServerConfig {
        world_dir: None,
        view_radius,
        camera_window_radius: window_r,
        gen_threads: 0,
        gen_budget_per_tick: 512,
        send_budget_per_tick: 512,
        max_in_flight: 4096,
        lod_levels: 1,
        ..ServerConfig::default()
    });
    let (mut client, server_end) = jbm_core::link::in_process_pair();
    server.add_connection(server_end);
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "far".to_owned(),
        })
        .unwrap();
    let deadline = Instant::now() + TIMEOUT;
    let id = PlayerId(1);
    while server.players().next().is_none() {
        server.tick();
        while client.try_recv().unwrap().is_some() {}
        assert!(Instant::now() < deadline, "never joined");
    }
    server.teleport_player(id, DVec3::new(cam_x as f64, cam_y as f64, cam_z as f64));

    // Settle, mirroring the client's resident set.
    let mut resident: HashMap<CubePos, Box<Cube>> = HashMap::new();
    let mut quiet = 0;
    while quiet < 10 {
        server.tick();
        while let Some(msg) = client.try_recv().unwrap() {
            match msg {
                ServerMsg::CubeData { pos, cube } => {
                    resident.insert(pos, cube);
                }
                ServerMsg::UnloadCube { pos } => {
                    resident.remove(&pos);
                }
                _ => {}
            }
        }
        let (_, _, pending_spans, _) = server.span_stats();
        let settled = pending_spans == 0 && server.lod0_pending(id) == 0 && server.in_flight() == 0;
        quiet = if settled { quiet + 1 } else { 0 };
        assert!(Instant::now() < deadline, "never settled");
    }

    // Every face of a resident cube toward an unsent cube, within the
    // disk's interior (the outer rim's outward faces are the view's edge).
    let center = CubePos {
        x: cam_x.div_euclid(side) as i32,
        y: cam_y.div_euclid(side) as i32,
        z: cam_z.div_euclid(side) as i32,
    };
    let interior = |p: CubePos| {
        let (dx, dz) = (i64::from(p.x - center.x), i64::from(p.z - center.z));
        let r = i64::from(view_radius) - 1;
        dx * dx + dz * dz <= r * r
    };
    let n = CUBE_SIZE;
    let mut unsent: HashMap<CubePos, Cube> = HashMap::new();
    let mut violations = 0usize;
    let mut examples = Vec::new();
    let mut far_cubes = 0usize;
    for (&pos, cube) in &resident {
        if !interior(pos) {
            continue;
        }
        let (dx, dz) = (i64::from(pos.x - center.x), i64::from(pos.z - center.z));
        if dx * dx + dz * dz > i64::from(window_r * window_r) {
            far_cubes += 1;
        }
        for dir in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let npos = CubePos {
                x: pos.x + dir.0,
                y: pos.y + dir.1,
                z: pos.z + dir.2,
            };
            if resident.contains_key(&npos) || !interior(npos) {
                continue;
            }
            let neighbour = unsent
                .entry(npos)
                .or_insert_with(|| generator.generate(npos));
            let mut faces = 0usize;
            for a in 0..n {
                for b in 0..n {
                    // `from` on this cube's boundary layer, `to` across it.
                    let (from, to) = match dir {
                        (1, 0, 0) => ((n - 1, a, b), (0, a, b)),
                        (-1, 0, 0) => ((0, a, b), (n - 1, a, b)),
                        (0, 1, 0) => ((a, n - 1, b), (a, 0, b)),
                        (0, -1, 0) => ((a, 0, b), (a, n - 1, b)),
                        (0, 0, 1) => ((a, b, n - 1), (a, b, 0)),
                        _ => ((a, b, 0), (a, b, n - 1)),
                    };
                    let at = |c: &Cube, (x, y, z): (i32, i32, i32)| {
                        c.get(LocalPos::new(x as u8, y as u8, z as u8).unwrap())
                    };
                    if face_visible(&blocks, at(cube, from), at(neighbour, to)) {
                        faces += 1;
                    }
                }
            }
            if faces > 0 {
                violations += 1;
                if examples.len() < 8 {
                    examples.push((pos, dir, faces));
                }
            }
        }
    }
    println!(
        "resident {} cubes ({far_cubes} far, interior); {violations} cube faces lose \
         visible block faces to an unsent neighbour",
        resident.len()
    );
    assert!(
        far_cubes > 0,
        "the view must reach beyond the camera window"
    );
    assert_eq!(
        violations, 0,
        "faces culled against unsent cubes (cube, direction, faces): {examples:?}"
    );
}
