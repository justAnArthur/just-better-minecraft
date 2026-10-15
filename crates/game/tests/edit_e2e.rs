//! End-to-end block editing without a GPU (task 1.7): a headless server
//! with inline generation, a real `ClientWorld`, and edits driven through
//! real protocol messages — the same path the app's optimistic edits take,
//! minus the window.

use glam::DVec3;
use jbm_client::world::ClientWorld;
use jbm_core::block::{BlockRegistry, BlockStateId, builtin};
use jbm_core::coords::{BlockPos, CubePos};
use jbm_core::protocol::{ClientMsg, ServerMsg};
use jbm_server::terrain::Generator;
use jbm_server::{DEFAULT_SEED, ServerConfig};
use jbm_testkit::HeadlessServer;

/// The streaming shape of the spec: R = 3, V = 2, inline generation.
fn config() -> ServerConfig {
    ServerConfig {
        seed: DEFAULT_SEED,
        streaming: true,
        view_radius: 3,
        vertical_radius: 2,
        send_budget_per_tick: 64,
        gen_threads: 0,
        gen_budget_per_tick: 64,
        max_in_flight: 512,
        world_dir: None,
        save_interval: std::time::Duration::from_secs(5),
        lod_levels: 4,
        lod_view_radius: 8,
        lod_vertical_radius: 3,
        lod_send_budget_per_tick: 32,
        lod_max_in_flight: 256,
    }
}

/// Cubes in the desired set for R = 3, V = 2: 29 columns of 5.
const DESIRED_CUBES: usize = 29 * 5;

/// Joins and waits for the whole desired set to stream in. Returns the
/// world with the dirty queue drained.
fn loaded_world(server: &mut HeadlessServer) -> (jbm_testkit::TestClient, ClientWorld) {
    let mut client = server.connect("player");
    let mut world = ClientWorld::new();
    for _ in 0..200 {
        server.step(1);
        for msg in client.drain() {
            world.apply(&msg);
        }
        if client.loaded().len() == DESIRED_CUBES {
            break;
        }
    }
    assert_eq!(
        client.loaded().len(),
        DESIRED_CUBES,
        "streaming did not settle"
    );
    let _ = world.take_dirty(usize::MAX, DVec3::ZERO);
    assert_eq!(world.dirty_count(), 0);
    (client, world)
}

#[test]
#[cfg(not(feature = "cube16"))] // end-to-end tests are 32³-only (see jbm-core's README)
fn breaking_a_surface_block_lands_within_one_tick() {
    let mut server = HeadlessServer::with_config(config());
    let (mut client, mut world) = loaded_world(&mut server);

    // The surface block in spawn's column, one under the resting feet:
    // next to spawn, in reach, not air.
    let mut registry = BlockRegistry::new();
    let blocks = builtin::Blocks::register(&mut registry);
    let surface = Generator::new(DEFAULT_SEED, blocks).surface_height(0, 0);
    let at = BlockPos {
        x: 0,
        y: surface,
        z: 0,
    };
    assert_ne!(
        world.block(at),
        Some(BlockStateId::AIR),
        "not a solid block"
    );
    let before = world.block(at).unwrap();

    // Break it through the protocol. Optimistically, the app would also
    // set the block here; the authoritative path below reverts or agrees.
    client.send(ClientMsg::BreakBlock { pos: at });
    server.step(1);
    let mut saw_change = false;
    for msg in client.drain() {
        if let ServerMsg::BlockChange { pos, state } = &msg {
            assert_eq!((*pos, *state), (at, BlockStateId::AIR));
            saw_change = true;
        }
        world.apply(&msg);
    }
    assert!(saw_change, "no BlockChange came back");
    assert_eq!(world.block(at), Some(BlockStateId::AIR));

    // The edited cube re-meshes first, and exactly the rule's neighbors
    // are dirty. Reconstruct the expected set from the block's local
    // position: offsets on each axis for local 0 or CUBE_SIZE - 1.
    let (cube, local) = at.split();
    let offsets = |v: u8| -> Vec<i32> {
        let mut out = vec![0];
        if v == 0 {
            out.push(-1);
        } else if v as i32 == jbm_core::coords::CUBE_SIZE - 1 {
            out.push(1);
        }
        out
    };
    let mut expected = std::collections::HashSet::new();
    for dx in offsets(local.x) {
        for dy in offsets(local.y) {
            for dz in offsets(local.z) {
                let neighbor = CubePos {
                    x: cube.x + dx,
                    y: cube.y + dy,
                    z: cube.z + dz,
                };
                if client.loaded().contains(&neighbor) {
                    expected.insert(neighbor);
                }
            }
        }
    }
    let dirty: std::collections::HashSet<CubePos> = world
        .take_dirty(usize::MAX, DVec3::ZERO)
        .into_iter()
        .map(|(pos, _, _)| pos)
        .collect();
    assert_eq!(dirty, expected);

    // And with a budget of one, an edit-touched cube comes out first even
    // with streaming dirt queued nearer the camera. (Touched neighbors
    // share edit priority: a late neighbor would show a hole.) Put the
    // block back to re-dirty the same set; re-setting air would be a
    // no-op.
    let far_cube = client
        .loaded()
        .iter()
        .copied()
        .find(|pos| !expected.contains(pos))
        .expect("a loaded cube the edit does not touch");
    world.apply(&ServerMsg::CubeData {
        pos: far_cube,
        cube: Box::new(world.cube(far_cube).unwrap().as_ref().clone()),
    });
    world.set_block(at, before);
    let camera = DVec3::new(
        f64::from(far_cube.x) * 32.0 + 16.0,
        f64::from(far_cube.y) * 32.0 + 16.0,
        f64::from(far_cube.z) * 32.0 + 16.0,
    );
    let first = world.take_dirty(1, camera);
    assert_eq!(first.len(), 1);
    assert!(
        expected.contains(&first[0].0),
        "{:?} is not an edit-touched cube",
        first[0].0
    );
}
