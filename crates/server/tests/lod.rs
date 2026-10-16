//! LOD pyramid tests (task 2.4): thin features surviving the levels, edit
//! propagation through the real `Server` paths, incremental-vs-rebuild
//! equality, and the ingest/eviction rules.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use jbm_core::block::{BlockRegistry, BlockStateId, builtin::Blocks};
use jbm_core::coords::{BlockPos, CUBE_SIZE, CubePos, LocalPos};
use jbm_core::cube::Cube;
use jbm_core::link::{ClientLink, in_process_pair};
use jbm_core::lod::{octant_offset, parent};
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};
use jbm_server::lod::LodPyramid;
use jbm_server::storage::Storage;
use jbm_server::terrain::Generator;
use jbm_server::{DEFAULT_SEED, Server, ServerConfig};

/// The builtin block states, against a fresh registry.
fn blocks() -> Blocks {
    Blocks::register(&mut BlockRegistry::new())
}

/// Small local xorshift, so tests never need the `rand` crate.
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// A value in `0..n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A small-radius inline-generation streaming server with the pyramid on.
fn streaming_config() -> ServerConfig {
    ServerConfig {
        streaming: true,
        view_radius: 1,
        vertical_radius: 1,
        gen_threads: 0,
        gen_budget_per_tick: 64,
        max_in_flight: 512,
        lod_levels: 4,
        ..ServerConfig::default()
    }
}

/// Streams a server with one joined, settled player (inline generation),
/// plus the drained client link.
fn settled_server(config: ServerConfig) -> (Server, ClientLink) {
    let mut server = Server::with_config(config);
    let (mut client, server_end) = in_process_pair();
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "lod-test".to_owned(),
        })
        .unwrap();
    server.add_connection(server_end);
    for _ in 0..300 {
        server.tick();
        drain(&mut client);
    }
    (server, client)
}

fn drain(client: &mut ClientLink) -> Vec<ServerMsg> {
    let mut messages = Vec::new();
    while let Ok(Some(msg)) = client.try_recv() {
        messages.push(msg);
    }
    messages
}

/// The level-`level` cube and cell covering world block `(x, y, z)`. A
/// level-*n* cube spans `2^(5+n)` blocks; its cells span `2^n`.
fn lod_cell(level: u8, x: i64, y: i64, z: i64) -> (CubePos, LocalPos) {
    let shift = 5 + level as i64;
    let mask = (1i64 << shift) - 1;
    (
        CubePos {
            x: (x >> shift) as i32,
            y: (y >> shift) as i32,
            z: (z >> shift) as i32,
        },
        LocalPos::new_unchecked(
            ((x & mask) >> level) as u8,
            ((y & mask) >> level) as u8,
            ((z & mask) >> level) as u8,
        ),
    )
}

/// Builds a LOD0 cube from a per-local-cell closure.
fn lod0_cube(f: impl Fn(u8, u8, u8) -> BlockStateId) -> Cube {
    Cube::from_fn(|p| f(p.x, p.y, p.z))
}

/// The y of the topmost non-air LOD0 block in the spawn column — the
/// surface under the player's feet.
fn surface_y(server: &Server) -> i64 {
    let feet = server.players().next().unwrap().state.position.y.floor() as i64;
    let mut y = feet;
    while server
        .block(BlockPos { x: 0, y, z: 0 })
        .is_none_or(|state| state == BlockStateId::AIR)
    {
        y -= 1;
    }
    y
}

/// The incremental-vs-reference property: every cube the incremental
/// pyramid holds **ready** must equal the reference (which only ever holds
/// ready cubes), cell for cell and `known` for `known`. Mid-stream the
/// incremental may additionally hold pending cubes whose fills have not
/// run; `full` (the final check, after everything was requested and
/// drained) also demands the cube sets match exactly.
fn assert_pyramids_equal(incremental: &LodPyramid, rebuilt: &LodPyramid, full: bool, what: &str) {
    assert_eq!(
        incremental.max_levels(),
        rebuilt.max_levels(),
        "{what}: level counts differ"
    );
    for level in 1..=incremental.max_levels() {
        let a: HashMap<CubePos, (u8, Cube)> = incremental
            .iter(level)
            .filter(|(_, lc)| lc.ready())
            .map(|(&pos, lc)| (pos, (lc.known, lc.cube.clone())))
            .collect();
        let b: HashMap<CubePos, (u8, Cube)> = rebuilt
            .iter(level)
            .map(|(&pos, lc)| (pos, (lc.known, lc.cube.clone())))
            .collect();
        if full {
            assert_eq!(
                a.len(),
                b.len(),
                "{what}: level {level} ready cube counts differ (only-in-incremental: {:?}, only-in-rebuild: {:?})",
                a.keys().filter(|k| !b.contains_key(*k)).collect::<Vec<_>>(),
                b.keys().filter(|k| !a.contains_key(*k)).collect::<Vec<_>>()
            );
        }
        for (pos, (known, cube)) in &a {
            let other = b
                .get(pos)
                .unwrap_or_else(|| panic!("{what}: level {level} misses {pos:?}"));
            assert_eq!(known, &other.0, "{what}: level {level} {pos:?} known mask");
            if cube != &other.1 {
                for index in 0..jbm_core::coords::CUBE_VOLUME {
                    let local = LocalPos::from_index(index);
                    if cube.get(local) != other.1.get(local) {
                        panic!(
                            "{what}: level {level} {pos:?} cell {local:?}: \
                             incremental {:?} rebuild {:?} (known {known:08b})",
                            cube.get(local),
                            other.1.get(local)
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Thin features

/// Inserts every LOD0 cube of the box `min..=max` (per axis), each built
/// by `cube_at`, so the whole chain above is fully real and ready without
/// any fill — the synthetic-world setup the thin-feature tests use.
fn insert_footprint(
    pyramid: &mut LodPyramid,
    min: CubePos,
    max: CubePos,
    cube_at: impl Fn(CubePos) -> Cube,
) {
    for x in min.x..=max.x {
        for y in min.y..=max.y {
            for z in min.z..=max.z {
                let pos = CubePos { x, y, z };
                let cube = cube_at(pos);
                pyramid.insert_lod0(pos, &cube);
            }
        }
    }
}

#[test]
fn one_block_wall_survives_lod1_and_lod2_at_both_parities() {
    let planks = blocks().oak_planks;
    for wall_x in [7u8, 8u8] {
        let mut pyramid = LodPyramid::new(4);
        // A wall 1 block thick, 32 long and 16 tall inside cube (0, 0, 0),
        // otherwise air. The parent's whole 2×2×2 footprint of LOD0 cubes
        // is inserted so the chain is fully real (no fills: synthetic air).
        insert_footprint(
            &mut pyramid,
            CubePos { x: 0, y: 0, z: 0 },
            CubePos { x: 1, y: 1, z: 1 },
            |pos| {
                if pos == (CubePos { x: 0, y: 0, z: 0 }) {
                    lod0_cube(|x, _y, _z| {
                        if x == wall_x {
                            planks
                        } else {
                            BlockStateId::AIR
                        }
                    })
                } else {
                    Cube::filled(BlockStateId::AIR)
                }
            },
        );

        // LOD1: a filled column at the wall's cell x for every covered y
        // and z (8 and 16 of them), with air on both sides.
        let cell_x = wall_x / 2;
        let lod1 = pyramid.get(1, CubePos { x: 0, y: 0, z: 0 }).unwrap();
        for y in 0..8u8 {
            for z in 0..16u8 {
                assert_eq!(
                    lod1.cube.get(LocalPos::new_unchecked(cell_x, y, z)),
                    planks,
                    "LOD1 wall cell at x={wall_x}, y={y}, z={z}"
                );
                for side in [cell_x - 1, cell_x + 1] {
                    assert_eq!(
                        lod1.cube.get(LocalPos::new_unchecked(side, y, z)),
                        BlockStateId::AIR,
                        "LOD1 beside the wall (x={wall_x}, side={side}, y={y}, z={z})"
                    );
                }
            }
        }

        // LOD2: the same wall, 4 and 8 cells deep.
        let cell2_x = wall_x / 4;
        let lod2 = pyramid.get(2, CubePos { x: 0, y: 0, z: 0 }).unwrap();
        for y in 0..4u8 {
            for z in 0..8u8 {
                assert_eq!(
                    lod2.cube.get(LocalPos::new_unchecked(cell2_x, y, z)),
                    planks,
                    "LOD2 wall cell at x={wall_x}, y={y}, z={z}"
                );
                for side in [cell2_x - 1, cell2_x + 1] {
                    assert_eq!(
                        lod2.cube.get(LocalPos::new_unchecked(side, y, z)),
                        BlockStateId::AIR,
                        "LOD2 beside the wall (x={wall_x}, side={side}, y={y}, z={z})"
                    );
                }
            }
        }
    }
}

#[test]
fn tree_trunk_survives_lod1_and_lod2_and_keeps_log() {
    let log = blocks().oak_log;
    let mut pyramid = LodPyramid::new(4);
    // A 1×1 trunk, 8 tall, surrounded by air, with the parent's full
    // footprint inserted so the chain is fully real.
    insert_footprint(
        &mut pyramid,
        CubePos { x: 0, y: 0, z: 0 },
        CubePos { x: 1, y: 1, z: 1 },
        |pos| {
            if pos == (CubePos { x: 0, y: 0, z: 0 }) {
                lod0_cube(|x, y, z| {
                    if (x, z) == (7, 7) && y < 8 {
                        log
                    } else {
                        BlockStateId::AIR
                    }
                })
            } else {
                Cube::filled(BlockStateId::AIR)
            }
        },
    );

    for level in [1u8, 2u8] {
        let (pos, _) = lod_cell(level, 7, 7, 7);
        assert_eq!(pos, CubePos { x: 0, y: 0, z: 0 });
        let lod = pyramid.get(level, pos).unwrap();
        let tall = 8usize >> level;
        for y in 0..tall as u8 {
            assert_eq!(
                lod.cube
                    .get(LocalPos::new_unchecked(7 >> level, y, 7 >> level)),
                log,
                "trunk cell at level {level}, y={y}"
            );
        }
    }
}

#[test]
fn a_single_floating_block_vanishes_at_lod1() {
    // Expected behavior: one filled child of eight is below the occupancy
    // threshold, so the feature disappears from every LOD level.
    let planks = blocks().oak_planks;
    let mut pyramid = LodPyramid::new(2);
    let cube = lod0_cube(|x, y, z| {
        if (x, y, z) == (7, 7, 7) {
            planks
        } else {
            BlockStateId::AIR
        }
    });
    pyramid.insert_lod0(CubePos { x: 0, y: 0, z: 0 }, &cube);
    let lod1 = pyramid.get(1, CubePos { x: 0, y: 0, z: 0 }).unwrap();
    assert_eq!(lod1.cube.uniform(), Some(BlockStateId::AIR));
}

#[test]
fn walls_straddling_cube_borders_survive() {
    // Walls at world x = -1 (local 31 of cube -1) and x = 0 (local 0 of
    // cube 0), each spanning z = -16..16, so they cross the x = 0 and
    // z = 0 cube borders. The whole LOD0 footprint of the two level-2
    // cubes involved is inserted, so the chain is fully real.
    let planks = blocks().oak_planks;
    let mut pyramid = LodPyramid::new(2);
    insert_footprint(
        &mut pyramid,
        CubePos { x: -4, y: 0, z: -4 },
        CubePos { x: 3, y: 3, z: 3 },
        |pos| {
            let base_x = i64::from(pos.x) * i64::from(CUBE_SIZE);
            let base_z = i64::from(pos.z) * i64::from(CUBE_SIZE);
            let wall_x = if pos.x < 0 { 31u8 } else { 0u8 };
            let in_wall_z = if pos.z < 0 { 16u8..32 } else { 0u8..16 };
            let _ = (base_x, base_z);
            lod0_cube(move |x, _y, z| {
                if x == wall_x && in_wall_z.contains(&z) {
                    planks
                } else {
                    BlockStateId::AIR
                }
            })
        },
    );
    for level in [1u8, 2u8] {
        // The wall at world x = 0 lands in the non-negative level cubes at
        // cell x 0; the wall at x = -1 in the negative ones, at cell x
        // 15 >> level - 1. Both survive with planks on both sides of x = 0.
        for world_z in [-8i64, 8] {
            for world_x in [0i64, -1] {
                let (pos, cell) = lod_cell(level, world_x, 0, world_z);
                let lod = pyramid.get(level, pos).expect("ancestor exists");
                assert_eq!(
                    lod.cube.get(cell),
                    planks,
                    "wall cell at level {level}, world x={world_x}, z={world_z}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------
// Known masks, re-insertion

#[test]
fn inserting_a_cube_marks_its_whole_ancestor_chain() {
    let stone = blocks().stone;
    let mut pyramid = LodPyramid::new(4);
    let pos = CubePos { x: 1, y: 2, z: 3 };
    pyramid.insert_lod0(pos, &Cube::filled(stone));

    // Level 1 holds the real octant at once; level 2 holds it pending —
    // real by definition, computed only once its child is ready. The walk
    // stops there: nothing above is touched until the deferral completes
    // (which never happens in a bare pyramid: nothing fills).
    let mut child = pos;
    for level in 1..=2u8 {
        let (ancestor, octant) = parent(child);
        let lod = pyramid.get(level, ancestor).expect("chain cube exists");
        if level == 1 {
            assert_eq!(lod.known, 1 << octant, "level 1 known mask");
            assert_eq!(lod.pending, 0);
        } else {
            assert_eq!(lod.pending, 1 << octant, "level 2 pending mask");
            assert_eq!(lod.known, 0, "level 2 is not real yet");
        }
        assert!(!lod.ready(), "level {level} is not ready without fills");
        child = ancestor;
    }
    assert_eq!(pyramid.total_cubes(), 2, "levels 3 and 4 are not touched");

    // Negative coordinates: parent and octant math through div_euclid.
    let neg = CubePos {
        x: -1,
        y: -2,
        z: -3,
    };
    let mut pyramid = LodPyramid::new(1);
    pyramid.insert_lod0(neg, &Cube::filled(stone));
    let (ancestor, octant) = parent(neg);
    // rem_euclid(2) of (-1, -2, -3) is (1, 0, 1), so the octant index is
    // dx + 2·dz + 4·dy = 1 + 2·1 + 0 = 3.
    assert_eq!(
        (ancestor, octant),
        (
            CubePos {
                x: -1,
                y: -1,
                z: -2
            },
            3
        )
    );
    assert_eq!(pyramid.get(1, ancestor).unwrap().known, 1 << 3);
}

#[test]
fn reinserting_an_unchanged_cube_reports_no_changes() {
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let h = generator.surface_height(0, 0);
    let pos = CubePos {
        x: 0,
        y: (h >> 5) as i32,
        z: 0,
    };
    let cube = generator.generate(pos);
    let mut pyramid = inline_pyramid(4, 64);
    // The parent is not ready until its fill runs, so the insert lists
    // nothing (no partial cubes); drained, its chain readies and reports.
    let first = pyramid.insert_lod0(pos, &cube);
    assert!(
        first.is_empty(),
        "a not-ready cube is never listed: {first:?}"
    );
    drain_pyramid(&mut pyramid);
    let p1 = parent(pos).0;
    let p2 = parent(p1).0;
    let p3 = parent(p2).0;
    assert_eq!(
        pyramid.take_changed(),
        vec![(1, p1), (2, p2), (3, p3)],
        "the first insert changes content"
    );
    let second = pyramid.insert_lod0(pos, &cube);
    assert!(second.is_empty(), "re-inserting changed {second:?}");
    assert!(
        pyramid.take_changed().is_empty(),
        "re-inserting changed ready content"
    );
}

#[test]
fn pyramid_accessors_report_levels_and_sizes() {
    let stone = blocks().stone;
    let mut pyramid = LodPyramid::new(3);
    assert_eq!(pyramid.max_levels(), 3);
    assert_eq!(pyramid.len(1), 0);
    assert_eq!(pyramid.len(4), 0, "above the top");
    assert!(
        pyramid.get(0, CubePos { x: 0, y: 0, z: 0 }).is_none(),
        "level 0"
    );
    assert_eq!(pyramid.total_cubes(), 0);
    assert_eq!(pyramid.heap_bytes(), 0, "nothing allocated while empty");

    pyramid.insert_lod0(CubePos { x: 0, y: 0, z: 0 }, &Cube::filled(stone));
    // The deferral walk reaches level 2 (pending) and stops: one cube at
    // each of levels 1 and 2, none above.
    assert_eq!(pyramid.len(1), 1);
    assert_eq!(pyramid.len(2), 1, "one pending cube at level 2");
    assert_eq!(pyramid.len(3), 0, "the walk stops at the deferral");
    assert_eq!(pyramid.total_cubes(), 2);
    assert!(pyramid.heap_bytes() > 0);
}

// ---------------------------------------------------------------------
// Edit propagation

#[test]
fn early_stop_an_edit_inside_solid_stone_reports_no_changes() {
    let stone = blocks().stone;
    let mut pyramid = LodPyramid::new(4);
    let pos = CubePos { x: 0, y: -10, z: 0 };
    pyramid.insert_lod0(pos, &Cube::filled(stone));
    let before = pyramid.get(1, parent(pos).0).unwrap().clone();

    // Breaking one block deep inside solid stone: the group keeps 7 of 8
    // filled and stone on every column top, so no level changes.
    let mut cube = Cube::filled(stone);
    let local = LocalPos::new_unchecked(7, 7, 7);
    cube.set(local, BlockStateId::AIR);
    let changes = pyramid.edit_lod0(pos.block(local), &cube);
    assert!(changes.is_empty(), "changes reported: {changes:?}");
    let after = pyramid.get(1, parent(pos).0).unwrap();
    assert_eq!(after.cube, before.cube);
    assert_eq!(after.known, before.known);
}

#[test]
fn incremental_equals_rebuild_under_random_interleavings() {
    // The content-model property (2.7): after any order of inserts, edits,
    // requests, fills and polls, every cube equals the reference — cell for
    // cell and `known` for `known` — once the pipeline drains. Seeded from
    // JBM_LOD_TEST_SEED (hex) so acceptance runs can vary it; fixed by
    // default so the suite is deterministic.
    let seed = std::env::var("JBM_LOD_TEST_SEED")
        .ok()
        .and_then(|s| u64::from_str_radix(&s, 16).ok())
        .unwrap_or(0x10D5_EED1);
    run_property_ops(seed, 2_000);
}

#[test]
fn content_model_property_holds_for_ten_seeds() {
    // The acceptance run of the property above: ten different seeds, a
    // shorter stream each.
    for seed in [0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xA] {
        run_property_ops(seed | 0x5EED_0000, 500);
    }
}

/// The property tests' shared world: a 4×2×4 region of generated LOD0
/// cubes straddling the origin (so edits land on cube borders and at
/// negative coordinates) with their latest content, plus the per-level
/// ancestor boxes that requests stay inside, keeping the incremental cube
/// set comparable to the reference's.
struct PropertyWorld {
    region: Vec<CubePos>,
    lod0: HashMap<CubePos, Cube>,
    boxes: HashMap<u8, (CubePos, CubePos)>,
}

impl PropertyWorld {
    fn new(generator: &Generator) -> Self {
        let h = generator.surface_height(0, 0);
        let y_base = (h >> 5) as i32 - 1;
        let region: Vec<CubePos> = (-2..2)
            .flat_map(|x| (y_base..y_base + 2).map(move |y| (x, y)))
            .flat_map(|(x, y)| (-2..2).map(move |z| CubePos { x, y, z }))
            .collect();
        assert_eq!(region.len(), 32);
        let lod0: HashMap<CubePos, Cube> = region
            .iter()
            .map(|&pos| (pos, generator.generate(pos)))
            .collect();
        // The boxes the reference fills: the ancestors' bounding box per
        // level.
        let parents: Vec<CubePos> = region.iter().map(|&pos| parent(pos).0).collect();
        let mut lo = parents[0];
        let mut hi = parents[0];
        for &p in &parents {
            lo = CubePos {
                x: lo.x.min(p.x),
                y: lo.y.min(p.y),
                z: lo.z.min(p.z),
            };
            hi = CubePos {
                x: hi.x.max(p.x),
                y: hi.y.max(p.y),
                z: hi.z.max(p.z),
            };
        }
        let mut boxes: HashMap<u8, (CubePos, CubePos)> = HashMap::new();
        for level in 1..=4u8 {
            boxes.insert(level, (lo, hi));
            lo = parent(lo).0;
            hi = parent(hi).0;
        }
        Self {
            region,
            lod0,
            boxes,
        }
    }

    /// A random cube of `level`'s box.
    fn random_in_box(&self, level: u8, rng: &mut XorShift64) -> CubePos {
        let &(lo, hi) = &self.boxes[&level];
        CubePos {
            x: lo.x + rng.below((hi.x - lo.x + 1) as u64) as i32,
            y: lo.y + rng.below((hi.y - lo.y + 1) as u64) as i32,
            z: lo.z + rng.below((hi.z - lo.z + 1) as u64) as i32,
        }
    }
}

/// Requests every cube of the per-level boxes: after a drain the ready
/// sets of the incremental pyramid and the reference are comparable cube
/// for cube.
fn request_boxes(pyramid: &mut LodPyramid, world: &PropertyWorld) {
    for level in 1..=4u8 {
        let &(lo, hi) = &world.boxes[&level];
        for x in lo.x..=hi.x {
            for y in lo.y..=hi.y {
                for z in lo.z..=hi.z {
                    pyramid.request(level, CubePos { x, y, z });
                }
            }
        }
    }
}

fn run_property_ops(seed: u64, ops: u64) {
    let mut rng = XorShift64(seed | 1);

    let block_set = blocks();
    let states = [
        BlockStateId::AIR,
        block_set.stone,
        block_set.dirt,
        block_set.grass_block,
        block_set.oak_planks,
        block_set.oak_log,
        block_set.glass,
        block_set.water,
    ];
    let generator = Generator::new(DEFAULT_SEED, block_set);
    let mut world = PropertyWorld::new(&generator);

    // Inline pipeline with a small budget, so fills interleave with the
    // operations instead of draining at once.
    let mut pyramid = LodPyramid::with_workers(
        4,
        std::sync::Arc::new(Generator::new(DEFAULT_SEED, blocks())),
        None,
        0,
        7,
    );
    // The model is a function of the LOD0 cubes *ever seen*, so the whole
    // region is inserted up front; the op stream re-inserts (as streaming
    // would), edits and requests on top.
    for &pos in &world.region {
        pyramid.insert_lod0(pos, &world.lod0[&pos]);
    }

    for op in 0..ops {
        match rng.below(100) {
            0..=39 => {
                // An edit through the 2.4 generator.
                let pos = world.region[rng.below(world.region.len() as u64) as usize];
                let local = LocalPos::new_unchecked(
                    rng.below(CUBE_SIZE as u64) as u8,
                    rng.below(CUBE_SIZE as u64) as u8,
                    rng.below(CUBE_SIZE as u64) as u8,
                );
                let state = states[rng.below(states.len() as u64) as usize];
                let cube = world.lod0.get_mut(&pos).expect("region cube");
                cube.set(local, state);
                pyramid.edit_lod0(pos.block(local), cube);
            }
            40..=59 => {
                // A (re-)insert of a region cube, as streaming would.
                let pos = world.region[rng.below(world.region.len() as u64) as usize];
                pyramid.insert_lod0(pos, &world.lod0[&pos]);
            }
            _ => {
                // A request at a random level inside the fill boxes.
                let level = 1 + rng.below(4) as u8;
                pyramid.request(level, world.random_in_box(level, &mut rng));
                pyramid.poll();
            }
        }
        if op % 100 == 99 {
            // Drain the pipeline completely: every cube the chain needs is
            // ready, and every ready cube equals the reference.
            drain_pyramid(&mut pyramid);
            let rebuilt = LodPyramid::rebuild(
                4,
                &Generator::new(DEFAULT_SEED, blocks()),
                world.region.iter().map(|&pos| (pos, &world.lod0[&pos])),
            );
            let final_check = op == ops - 1;
            if final_check {
                // Ask for every fill-box cube, so the ready sets are
                // comparable cube for cube.
                request_boxes(&mut pyramid, &world);
                drain_pyramid(&mut pyramid);
            }
            assert_pyramids_equal(&pyramid, &rebuilt, final_check, "after op {op}");
        }
    }
}

// ---------------------------------------------------------------------
// Storage-backed content-model property (review finding 2)

#[test]
fn storage_backed_content_model_property_holds_for_ten_seeds() {
    // The property above, with persistence in the interleaving: the op
    // stream also flushes the dirty set to redb and drops and reopens the
    // world mid-stream, after which every fill is a storage load. Seeded
    // from the same JBM_LOD_TEST_SEED env var (hex) as its sibling; ten
    // derived seeds run per invocation, so one `cargo test` covers them
    // all.
    let seed = std::env::var("JBM_LOD_TEST_SEED")
        .ok()
        .and_then(|s| u64::from_str_radix(&s, 16).ok())
        .unwrap_or(0x570A_5EED);
    for round in 0..10u64 {
        run_storage_property_ops(seed + 0x100 * round, 150);
    }
}

/// Flushes the pyramid's dirty set in one batch, as the server's save path
/// does: drain first, then every changed ready cube.
fn flush_lod(pyramid: &mut LodPyramid, storage: &Storage) {
    drain_pyramid(pyramid);
    let dirty = pyramid.take_store_dirty();
    let records: Vec<(u8, CubePos, Cube, u8)> = dirty
        .iter()
        .map(|&(level, pos)| {
            let lc = pyramid.get(level, pos).expect("dirty cubes are held");
            (level, pos, lc.cube.clone(), lc.known)
        })
        .collect();
    let lod_refs: Vec<(u8, CubePos, &Cube, u8)> = records
        .iter()
        .map(|(level, pos, cube, known)| (*level, *pos, cube, *known))
        .collect();
    storage.save_lod_batch(&[], &lod_refs, &[]).unwrap();
}

fn run_storage_property_ops(seed: u64, ops: u64) {
    let mut rng = XorShift64(seed | 1);

    let block_set = blocks();
    let states = [
        BlockStateId::AIR,
        block_set.stone,
        block_set.dirt,
        block_set.grass_block,
        block_set.oak_planks,
        block_set.oak_log,
        block_set.glass,
        block_set.water,
    ];
    let generator = Arc::new(Generator::new(DEFAULT_SEED, block_set));
    let mut world = PropertyWorld::new(&generator);

    // The same inline pipeline as above, over a tempdir world with stored
    // LOD behind it.
    let dir = tempfile::TempDir::new().unwrap();
    let mut storage = open_lod_storage(dir.path());
    let mut pyramid =
        LodPyramid::with_workers(4, Arc::clone(&generator), Some(Arc::clone(&storage)), 0, 7);
    for &pos in &world.region {
        pyramid.insert_lod0(pos, &world.lod0[&pos]);
    }

    for op in 0..ops {
        match rng.below(100) {
            0..=34 => {
                // An edit through the 2.4 generator.
                let pos = world.region[rng.below(world.region.len() as u64) as usize];
                let local = LocalPos::new_unchecked(
                    rng.below(CUBE_SIZE as u64) as u8,
                    rng.below(CUBE_SIZE as u64) as u8,
                    rng.below(CUBE_SIZE as u64) as u8,
                );
                let state = states[rng.below(states.len() as u64) as usize];
                let cube = world.lod0.get_mut(&pos).expect("region cube");
                cube.set(local, state);
                pyramid.edit_lod0(pos.block(local), cube);
            }
            35..=54 => {
                // A (re-)insert of a region cube, over whatever the
                // storage loads now hold.
                let pos = world.region[rng.below(world.region.len() as u64) as usize];
                pyramid.insert_lod0(pos, &world.lod0[&pos]);
            }
            55..=84 => {
                // A request at a random level: a storage load where a
                // flushed record exists, a worldgen sample otherwise.
                let level = 1 + rng.below(4) as u8;
                pyramid.request(level, world.random_in_box(level, &mut rng));
                pyramid.poll();
            }
            85..=92 => {
                // A save interval: one batch with every dirty cube.
                flush_lod(&mut pyramid, &storage);
            }
            _ => {
                // A restart: flush, close the world, reopen it empty and
                // ask for the view again — the fills are now storage
                // loads. No LOD0 cube re-enters (nobody walked there),
                // and the reference's inputs are unchanged, so the
                // pyramid must still equal it.
                flush_lod(&mut pyramid, &storage);
                drop(pyramid);
                drop(storage);
                storage = open_lod_storage(dir.path());
                pyramid = LodPyramid::with_workers(
                    4,
                    Arc::clone(&generator),
                    Some(Arc::clone(&storage)),
                    0,
                    7,
                );
                request_boxes(&mut pyramid, &world);
                drain_pyramid(&mut pyramid);
            }
        }
        if op % 50 == 49 {
            drain_pyramid(&mut pyramid);
            let rebuilt = LodPyramid::rebuild(
                4,
                &generator,
                world.region.iter().map(|&pos| (pos, &world.lod0[&pos])),
            );
            let final_check = op == ops - 1;
            if final_check {
                request_boxes(&mut pyramid, &world);
                drain_pyramid(&mut pyramid);
            }
            assert_pyramids_equal(&pyramid, &rebuilt, final_check, "storage op {op}");
        }
    }
    assert!(
        storage.lod_record_count().unwrap() > 0,
        "the stream stored LOD records"
    );
}

// ---------------------------------------------------------------------
// Through the real Server

/// The topmost non-air block of LOD0 column (x, z), from `top` down.
fn lod0_top(server: &Server, x: i64, z: i64, top: i64) -> Option<BlockStateId> {
    (1..=top)
        .rev()
        .map(|y| server.block(BlockPos { x, y, z }))
        .find(|state| state.is_some_and(|s| s != BlockStateId::AIR))
        .flatten()
}

/// The topmost non-air cell of the level-`level` column over world block
/// (x, z), scanning level cubes downward from the one holding `hint`.
fn lod_top(server: &Server, level: u8, x: i64, z: i64, hint: i64) -> Option<BlockStateId> {
    let (mut cube_pos, probe) = lod_cell(level, x, hint, z);
    for _ in 0..8 {
        if let Some(cube) = server.lod_cube(level, cube_pos) {
            for cy in (0..32u8).rev() {
                let state = cube.get(LocalPos::new_unchecked(probe.x, cy, probe.z));
                if state != BlockStateId::AIR {
                    return Some(state);
                }
            }
        }
        cube_pos.y -= 1;
    }
    None
}

#[test]
fn terrain_lod_keeps_grass_on_top() {
    // Stream the default view around spawn through the Server (inline
    // generation), then measure how many LOD columns whose LOD0 columns
    // are all grass-topped keep grass as their top filled cell. Steep
    // spots are the expected exceptions; the bar is 90%.
    let (server, _client) = settled_server(ServerConfig {
        gen_threads: 0,
        ..ServerConfig::default()
    });
    let grass = blocks().grass_block;
    const TOP: i64 = 92; // the shipped graph's MAX_HEIGHT

    for level in [1u8, 2u8] {
        let step = 1i64 << level;
        let mut measured = 0usize;
        let mut kept = 0usize;
        for x in (-48..48).step_by(step as usize) {
            for z in (-48..48).step_by(step as usize) {
                let all_grass = (0..step).all(|dx| {
                    (0..step).all(|dz| lod0_top(&server, x + dx, z + dz, TOP) == Some(grass))
                });
                if !all_grass {
                    continue;
                }
                if let Some(top) = lod_top(&server, level, x, z, 64) {
                    measured += 1;
                    if top == grass {
                        kept += 1;
                    }
                }
            }
        }
        let fraction = kept as f64 / measured.max(1) as f64;
        println!(
            "terrain_lod_keeps_grass_on_top: level {level}: {kept}/{measured} = {fraction:.3}"
        );
        assert!(
            fraction >= 0.90,
            "level {level}: only {fraction:.3} of grass columns keep grass"
        );
    }
}

/// Places `state` at `pos` through the real message path.
fn place(server: &mut Server, client: &mut ClientLink, pos: BlockPos, state: BlockStateId) {
    client.send(ClientMsg::PlaceBlock { pos, state }).unwrap();
    server.tick();
    drain(client);
    assert_eq!(server.block(pos), Some(state), "place at {pos:?} rejected");
}

/// Breaks the block at `pos` through the real message path.
fn break_block(server: &mut Server, client: &mut ClientLink, pos: BlockPos) {
    client.send(ClientMsg::BreakBlock { pos }).unwrap();
    server.tick();
    drain(client);
    assert_eq!(
        server.block(pos),
        Some(BlockStateId::AIR),
        "break at {pos:?} rejected"
    );
}

/// The y of the topmost non-air block of column `(x, z)`, or `None` when
/// the column is empty (or not loaded).
fn column_top(server: &Server, x: i64, z: i64) -> Option<i64> {
    (1..=92i64).rev().find(|&y| {
        server
            .block(BlockPos { x, y, z })
            .is_some_and(|s| s != BlockStateId::AIR)
    })
}

/// A flat 2×2 grass-topped patch near spawn, aligned to even coordinates
/// with an even top so the pillar over it fills two LOD1 cells, that does
/// not cover the player's own column.
fn pillar_patch(server: &Server) -> (i64, i64, i64) {
    let grass = blocks().grass_block;
    for x0 in (-2..=2i64).step_by(2) {
        for z0 in (-2..=2i64).step_by(2) {
            let tops: Vec<Option<i64>> = (0..2)
                .flat_map(|dx| (0..2).map(move |dz| (dx, dz)))
                .map(|(dx, dz)| column_top(server, x0 + dx, z0 + dz))
                .collect();
            let flat = tops.windows(2).all(|w| w[0] == w[1]);
            let Some(top) = tops[0] else {
                continue;
            };
            if top % 2 != 0 {
                continue; // an odd top folds both pillar layers into one LOD1 cell
            }
            let all_grass = (0..2i64).all(|dx| {
                (0..2i64).all(|dz| {
                    server.block(BlockPos {
                        x: x0 + dx,
                        y: top,
                        z: z0 + dz,
                    }) == Some(grass)
                })
            });
            // Breaking the block under the player would drop them; skip
            // any patch covering column (0, 0).
            let clear_of_player = !(x0..x0 + 2).contains(&0) || !(z0..z0 + 2).contains(&0);
            if flat && all_grass && clear_of_player {
                return (x0, top, z0);
            }
        }
    }
    panic!("no even, flat 2×2 grass patch near spawn");
}

#[test]
fn edits_show_in_lod_parents_and_removal_restores() {
    let (mut server, mut client) = settled_server(streaming_config());
    let planks = blocks().oak_planks;
    let (x0, top, z0) = pillar_patch(&server);

    // The LOD2 group the pillar's top occupies is [base, base + 4) with
    // base = top + 1 rounded down to a multiple of 4. Every other column
    // of that group's 4×4 footprint whose terrain reaches `base` would
    // nominate grass and outvote the pillar's single planks column, so
    // clear those tops below the base first — digging is an edit too.
    let base = (top + 1) & !3;
    let (fx, fz) = (x0 & !3, z0 & !3);
    for x in fx..fx + 4 {
        for z in fz..fz + 4 {
            // Never dig out the block under the player's feet.
            if (x, z) == (0, 0) {
                continue;
            }
            let in_patch = (x0..x0 + 2).contains(&x) && (z0..z0 + 2).contains(&z);
            while !in_patch && column_top(&server, x, z).is_some_and(|t| t >= base) {
                let t = column_top(&server, x, z).unwrap();
                break_block(&mut server, &mut client, BlockPos { x, y: t, z });
            }
        }
    }

    // A 2×2×2 planks pillar on the surface, through Server edits.
    let cells: Vec<BlockPos> = (x0..x0 + 2)
        .flat_map(|x| (top + 1..top + 3).map(move |y| (x, y)))
        .flat_map(|(x, y)| (z0..z0 + 2).map(move |z| BlockPos { x, y, z }))
        .collect();
    assert_eq!(cells.len(), 8);

    // Snapshot every LOD cube the pillar may touch.
    let touched = |server: &Server, level: u8| -> Vec<(CubePos, Cube)> {
        let mut positions: Vec<CubePos> = cells
            .iter()
            .map(|&p| lod_cell(level, p.x, p.y, p.z).0)
            .collect();
        positions.sort_unstable_by_key(|c| (c.x, c.y, c.z));
        positions.dedup();
        positions
            .into_iter()
            .map(|pos| (pos, server.lod_cube(level, pos).unwrap().clone()))
            .collect()
    };
    let before1 = touched(&server, 1);
    let before2 = touched(&server, 2);

    for &pos in &cells {
        place(&mut server, &mut client, pos, planks);
    }

    // LOD1 and LOD2 show planks at the pillar's cells.
    for &pos in &cells {
        for level in [1u8, 2u8] {
            let (cube_pos, local) = lod_cell(level, pos.x, pos.y, pos.z);
            assert_eq!(
                server.lod_cube(level, cube_pos).unwrap().get(local),
                planks,
                "level {level} at {pos:?}"
            );
        }
    }

    // Removing the pillar restores the previous cells exactly.
    for &pos in cells.iter().rev() {
        break_block(&mut server, &mut client, pos);
    }
    for (pos, before) in &before1 {
        assert_eq!(server.lod_cube(1, *pos).unwrap(), before);
    }
    for (pos, before) in &before2 {
        assert_eq!(server.lod_cube(2, *pos).unwrap(), before);
    }
}

/// A flat 2×2 grass-topped patch near spawn, aligned to even coordinates so
/// it fills exactly one LOD1 column, that does not cover the player's own
/// column. For the material-only edit.
fn flat_grass_patch(server: &Server) -> (i64, i64, i64) {
    let grass = blocks().grass_block;
    for x0 in (-2..=2i64).step_by(2) {
        for z0 in (-2..=2i64).step_by(2) {
            let tops: Vec<Option<i64>> = (0..2)
                .flat_map(|dx| (0..2).map(move |dz| (dx, dz)))
                .map(|(dx, dz)| column_top(server, x0 + dx, z0 + dz))
                .collect();
            let flat = tops.windows(2).all(|w| w[0] == w[1]) && tops[0].is_some();
            let all_grass = flat
                && (0..2i64).all(|dx| {
                    (0..2i64).all(|dz| {
                        server.block(BlockPos {
                            x: x0 + dx,
                            y: tops[0].unwrap(),
                            z: z0 + dz,
                        }) == Some(grass)
                    })
                });
            // Breaking the block under the player would drop them; skip
            // any patch covering column (0, 0).
            let clear_of_player = !(x0..x0 + 2).contains(&0) || !(z0..z0 + 2).contains(&0);
            if flat && all_grass && clear_of_player {
                return (x0, tops[0].unwrap(), z0);
            }
        }
    }
    panic!("no flat 2×2 grass patch near spawn");
}

#[test]
fn material_only_edit_changes_the_lod1_material() {
    let (mut server, mut client) = settled_server(streaming_config());
    let planks = blocks().oak_planks;
    let (x0, top, z0) = flat_grass_patch(&server);
    // Replace the 2×2 grass tops with planks: occupancy does not change,
    // only the material. (`PlaceBlock` needs air, so break first.)
    for dx in 0..2i64 {
        for dz in 0..2i64 {
            let pos = BlockPos {
                x: x0 + dx,
                y: top,
                z: z0 + dz,
            };
            break_block(&mut server, &mut client, pos);
            place(&mut server, &mut client, pos, planks);
        }
    }
    let (l1, local) = lod_cell(1, x0, top, z0);
    assert_eq!(
        server.lod_cube(1, l1).unwrap().get(local),
        planks,
        "the material edit must reach LOD1"
    );
}

#[test]
fn lod_levels_zero_disables_the_pyramid_entirely() {
    let (mut server, mut client) = settled_server(ServerConfig {
        lod_levels: 0,
        ..streaming_config()
    });
    assert!(server.world_cube_count() > 1, "terrain streamed");
    assert_eq!(server.lod_cube_count(), 0);
    assert_eq!(server.lod_cube(1, CubePos { x: 0, y: 0, z: 0 }), None);
    assert_eq!(server.lod_known(1, CubePos { x: 0, y: 0, z: 0 }), None);

    // An edit still works and still allocates nothing.
    let ground = BlockPos {
        x: 0,
        y: surface_y(&server),
        z: 0,
    };
    break_block(&mut server, &mut client, ground);
    assert_eq!(server.block(ground), Some(BlockStateId::AIR));
    assert_eq!(server.lod_cube_count(), 0);
}

#[test]
fn eviction_and_regeneration_leave_the_pyramid_unchanged() {
    let (mut server, client) = settled_server(streaming_config());
    // Snapshot the whole pyramid: position, known mask and content.
    let snapshot = |server: &Server| {
        let mut cubes: Vec<(u8, CubePos, u8, Cube)> = Vec::new();
        for level in 1..=4u8 {
            for (pos, lc) in server.lod_pyramid().iter(level) {
                cubes.push((level, *pos, lc.known, lc.cube.clone()));
            }
        }
        // (level, pos) is unique per entry, so ordering by it is total.
        cubes.sort_unstable_by_key(|(level, pos, _, _)| (*level, *pos));
        cubes
    };
    let before = snapshot(&server);
    assert!(!before.is_empty());

    // The player leaves: everything evicts (nothing is pinned), but the
    // pyramid outlives its LOD0 children.
    drop(client);
    for _ in 0..5 {
        server.tick();
    }
    assert_eq!(server.players().count(), 0);
    assert_eq!(server.world_cube_count(), 0, "nothing pinned, all evicted");
    assert!(server.lod_cube_count() > 0, "the pyramid outlives LOD0");

    // A new player joins the same spot: regeneration re-inserts every
    // cube, which must change no content — the pyramid compares equal.
    let (mut client2, server_end) = in_process_pair();
    client2
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "returning".to_owned(),
        })
        .unwrap();
    server.add_connection(server_end);
    for _ in 0..300 {
        server.tick();
        drain(&mut client2);
    }
    assert!(server.world_cube_count() > 1, "terrain regenerated");
    assert_eq!(snapshot(&server), before);
}

// ---------------------------------------------------------------------
// Worldgen LOD sampling (task 2.7)

/// The exact level-`level` downsample at `pos`: generated LOD0 cubes,
/// downsampled octant by octant all the way up. Level 0 is `generate`.
fn exact_cube(generator: &Generator, level: u8, pos: CubePos) -> Cube {
    if level == 0 {
        return generator.generate(pos);
    }
    let mut cube = Cube::filled(BlockStateId::AIR);
    for octant in 0..8u8 {
        let [dx, dy, dz] = octant_offset(octant).map(i32::from);
        let child = CubePos {
            x: pos.x * 2 + dx,
            y: pos.y * 2 + dy,
            z: pos.z * 2 + dz,
        };
        let child_cube = exact_cube(generator, level - 1, child);
        jbm_core::lod::downsample_octant(&child_cube, &mut cube, octant);
    }
    cube
}

/// Seeded level-`level` positions whose column intersects the surface,
/// spread over ±2048 blocks so no single plateau dominates the accuracy
/// numbers (mixed cubes, the sampling path's real case).
fn surface_positions(
    generator: &Generator,
    level: u8,
    count: u64,
    rng: &mut XorShift64,
) -> Vec<CubePos> {
    let shift = 5 + level as i64;
    let mut positions = Vec::new();
    let mut attempts = 0;
    while positions.len() < count as usize {
        attempts += 1;
        assert!(attempts < 2_000, "no surface positions found");
        let x = (rng.below(8192) as i64 - 4096) >> level;
        let z = (rng.below(8192) as i64 - 4096) >> level;
        let h = generator.surface_height(x << shift, z << shift);
        let y = (h >> shift) as i32;
        let pos = CubePos {
            x: x as i32,
            y,
            z: z as i32,
        };
        if !positions.contains(&pos) {
            positions.push(pos);
        }
    }
    positions
}

#[test]
fn sample_lod_level_1_is_the_downsample_of_generated_cubes() {
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let mut rng = XorShift64(0x5EED_10D1);
    // Negative coordinates and both parities, plus a cube straddling
    // SAND_LEVEL (a beach column) and one high grass column.
    let mut positions = surface_positions(&generator, 1, 18, &mut rng);
    positions.retain(|&p| p.x != 0 || p.z != 0);
    'beach: for x in (-512..512).step_by(4) {
        for z in (-512..512).step_by(16) {
            let sand_side = generator.surface_height(x, z) <= 61;
            let other_side = generator.surface_height(x + 32, z + 16) <= 61;
            if sand_side != other_side {
                // A level-1 cube whose 64-block footprint spans both a
                // sand and a grass column: it straddles SAND_LEVEL.
                let h = generator.surface_height(x, z);
                positions.push(CubePos {
                    x: (x >> 6) as i32,
                    y: (h >> 6) as i32,
                    z: (z >> 6) as i32,
                });
                break 'beach;
            }
        }
    }
    positions.push(CubePos {
        x: -1,
        y: -1,
        z: -1,
    });
    positions.push(CubePos { x: 0, y: 0, z: 0 });
    assert!(positions.len() >= 20, "{} positions", positions.len());

    let samples_before = generator.lod_sample_calls();
    for &pos in &positions {
        // The expectation generates; the counters are snapshotted around
        // the sample call itself.
        let expected = exact_cube(&generator, 1, pos);
        let generates = generator.generate_calls();
        let sampled = generator.sample_lod(1, pos);
        assert_eq!(sampled, expected, "level 1 at {pos:?}");
        assert_eq!(
            generator.generate_calls(),
            generates,
            "sample_lod never generates, at {pos:?}"
        );
    }
    assert!(
        generator.lod_sample_calls() > samples_before,
        "the mixed-path samples must count"
    );
}

#[test]
fn sample_lod_levels_2_3_4_track_the_exact_downsample() {
    // How far the worldgen guess can be from real terrain: per column, the
    // top filled cell within ±1 cell (≥95%) and the same top material
    // (≥90%). These bound the visible pop when a guess is replaced by real
    // data. Level 4 covers a 16³ LOD0 footprint, so it keeps to a few
    // positions.
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let mut rng = XorShift64(0xACC0_E552_0007);
    for (level, count) in [(2u8, 30u64), (3, 10), (4, 10)] {
        let positions = surface_positions(&generator, level, count, &mut rng);
        let mut columns = 0usize;
        let mut within_one = 0usize;
        let mut same_material = 0usize;
        for &pos in &positions {
            let sampled = generator.sample_lod(level, pos);
            let expected = exact_cube(&generator, level, pos);
            let top = |cube: &Cube, x: u8, z: u8| -> Option<(u8, BlockStateId)> {
                (0..CUBE_SIZE as u8).rev().find_map(|y| {
                    let state = cube.get(LocalPos::new_unchecked(x, y, z));
                    (state != BlockStateId::AIR).then_some((y, state))
                })
            };
            for x in 0..CUBE_SIZE as u8 {
                for z in 0..CUBE_SIZE as u8 {
                    let (columns_within, columns_material) =
                        match (top(&sampled, x, z), top(&expected, x, z)) {
                            (Some((sy, ss)), Some((ey, es))) => {
                                ((sy.abs_diff(ey) <= 1) as usize, (ss == es) as usize)
                            }
                            (None, None) => (1, 1),
                            // A column filled only on one side counts as a miss
                            // unless the miss is exactly one cell of height.
                            (Some((sy, _)), None) => ((sy == 0) as usize, 0),
                            (None, Some((ey, _))) => ((ey == 0) as usize, 0),
                        };
                    columns += 1;
                    within_one += columns_within;
                    same_material += columns_material;
                }
            }
        }
        let height_fraction = within_one as f64 / columns as f64;
        let material_fraction = same_material as f64 / columns as f64;
        println!(
            "sample_lod accuracy level {level}: top within ±1: {within_one}/{columns} = \
             {height_fraction:.3}, same top material: {same_material}/{columns} = \
             {material_fraction:.3}"
        );
        assert!(
            height_fraction >= 0.95,
            "level {level}: only {height_fraction:.3} of columns within ±1"
        );
        assert!(
            material_fraction >= 0.90,
            "level {level}: only {material_fraction:.3} of columns keep their material"
        );
    }
}

#[test]
fn sample_lod_bounds_fast_paths_sample_nothing_and_never_generate() {
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let samples = generator.lod_sample_calls();
    let generates = generator.generate_calls();
    for level in 1..=4u8 {
        let air = generator.sample_lod(
            level,
            CubePos {
                x: 0,
                y: 100,
                z: -3,
            },
        );
        assert_eq!(air, Cube::default(), "air fast path at level {level}");
        let stone = generator.sample_lod(
            level,
            CubePos {
                x: 3,
                y: -100,
                z: -3,
            },
        );
        assert_eq!(
            stone,
            Cube::filled(blocks().stone),
            "stone fast path at level {level}"
        );
    }
    assert_eq!(generator.lod_sample_calls(), samples, "no sampling ran");
    assert_eq!(generator.generate_calls(), generates, "no generation ran");
}

// ---------------------------------------------------------------------
// Content model: pending, no placeholder air, real data wins

/// An inline pyramid with the fill pipeline: what the content-model tests
/// drive directly.
fn inline_pyramid(levels: u8, budget: usize) -> LodPyramid {
    LodPyramid::with_workers(
        levels,
        Arc::new(Generator::new(DEFAULT_SEED, blocks())),
        None,
        0,
        budget,
    )
}

/// Pops every queued job, internal (pending-propagation) ones included:
/// after this the pyramid holds no pending work.
fn drain_pyramid(pyramid: &mut LodPyramid) {
    let mut spins = 0;
    loop {
        pyramid.poll();
        if pyramid.in_flight_total() == 0 {
            break;
        }
        spins += 1;
        assert!(
            spins < 10_000,
            "drain stuck with {} jobs",
            pyramid.in_flight_total()
        );
    }
}

/// Asserts every cell of `octant` inside `got` equals `expected` (a cell
/// probe at the octant's offset).
fn assert_octant_equals(
    got: &Cube,
    octant: u8,
    expected: impl Fn(LocalPos) -> BlockStateId,
    what: &str,
) {
    let half = (CUBE_SIZE / 2) as usize;
    let [ox, oy, oz] = octant_offset(octant).map(|v| v as usize * half);
    for y in 0..half {
        for z in 0..half {
            for x in 0..half {
                let cell = LocalPos::new_unchecked((ox + x) as u8, (oy + y) as u8, (oz + z) as u8);
                assert_eq!(got.get(cell), expected(cell), "{what} at {cell:?}");
            }
        }
    }
}

#[test]
fn no_placeholder_air_half_visited_level_2_cube() {
    // Visit the bottom half of a level-2 cube's footprint only, then ask
    // for the cube: its unvisited octants equal the worldgen sample, its
    // visited octants the downsample of the real level-1 children — which
    // are themselves filled. Placeholder air appears nowhere.
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let mut pyramid = inline_pyramid(2, 64);
    let h = generator.surface_height(0, 0);
    let level2 = CubePos {
        x: 0,
        y: (h >> 7) as i32,
        z: 0,
    };
    let y0 = level2.y * 4;
    // The bottom half: LOD0 y 0..2 of the level-2 cube's 4-cube-per-axis
    // span — the four level-1 children with dy = 0, all eight octants real.
    for dy in 0..2i32 {
        for dx in 0..4i32 {
            for dz in 0..4i32 {
                let pos = CubePos {
                    x: level2.x * 4 + dx,
                    y: y0 + dy,
                    z: level2.z * 4 + dz,
                };
                let cube = generator.generate(pos);
                pyramid.insert_lod0(pos, &cube);
            }
        }
    }
    pyramid.request(2, level2);
    drain_pyramid(&mut pyramid);

    let lod2 = pyramid.ready(2, level2).expect("the level-2 cube is ready");
    let sample = generator.sample_lod(2, level2);
    let base = CubePos {
        x: level2.x * 2,
        y: level2.y * 2,
        z: level2.z * 2,
    };
    for octant in 0..8u8 {
        let [dx, dy, dz] = octant_offset(octant).map(i32::from);
        let child = CubePos {
            x: base.x + dx,
            y: base.y + dy,
            z: base.z + dz,
        };
        if dy == 0 {
            // Visited: the downsample of the real, filled level-1 child.
            let child_lod = pyramid
                .ready(1, child)
                .expect("the visited level-1 child is ready");
            let mut want = Cube::filled(BlockStateId::AIR);
            jbm_core::lod::downsample_octant(child_lod, &mut want, octant);
            assert_octant_equals(
                lod2,
                octant,
                |cell| want.get(cell),
                "visited octant (real level-1 child)",
            );
        } else {
            // Unvisited: the worldgen sample's octant.
            let expected = sample.clone();
            assert_octant_equals(
                lod2,
                octant,
                move |cell| expected.get(cell),
                "unvisited octant (worldgen sample)",
            );
        }
    }
}

#[test]
fn ready_is_none_while_pending_and_changed_reports_only_ready_cubes() {
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let mut pyramid = inline_pyramid(4, 64);
    let h = generator.surface_height(0, 0);
    let pos = CubePos {
        x: 3,
        y: (h >> 5) as i32,
        z: -2,
    };
    let cube = generator.generate(pos);
    pyramid.insert_lod0(pos, &cube);
    let (p1, _) = parent(pos);
    let (p2, _) = parent(p1);
    // The chain exists but nothing is ready, and nothing is reported.
    assert!(pyramid.get(1, p1).is_some());
    assert!(pyramid.ready(1, p1).is_none(), "one octant of eight");
    assert!(pyramid.ready(2, p2).is_none());
    assert!(pyramid.take_changed().is_empty());

    // Fill the level-1 cube: it readies, and the deferral cascade fills
    // the whole chain — each ancestor gains its real octant and is filled
    // in turn, so every level becomes ready and reported.
    pyramid.request(1, p1);
    drain_pyramid(&mut pyramid);
    assert!(pyramid.ready(1, p1).is_some());
    let p3 = parent(p2).0;
    let changed = pyramid.take_changed();
    assert_eq!(
        changed,
        vec![(1, p1), (2, p2), (3, p3)],
        "every readied level of the chain is reported"
    );
    for (level, cube_pos) in [(1, p1), (2, p2), (3, p3)] {
        assert!(pyramid.ready(level, cube_pos).is_some());
    }
}

#[test]
fn insert_and_edit_changes_list_ready_cubes_only() {
    // The no-partial-cubes contract on the returned LodChanges: an insert
    // or an edit under a cube that is not ready lists nothing, and the
    // cube is still delivered by take_changed — edit included — when it
    // becomes ready.
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let mut pyramid = inline_pyramid(2, 64);
    let pos = CubePos {
        x: 3,
        y: (generator.surface_height(0, 0) >> 5) as i32,
        z: -2,
    };
    let mut cube = generator.generate(pos);
    let changes = pyramid.insert_lod0(pos, &cube);
    assert!(
        changes.is_empty(),
        "the not-ready parent is never listed: {changes:?}"
    );

    // An edit that flips a whole level-1 cell group (2×2×2 planks, aligned
    // to the cell grid) still lists nothing while the parent is not ready.
    let planks = blocks().oak_planks;
    let base = (6u8, 6, 6);
    for dy in 0..2u8 {
        for dz in 0..2u8 {
            for dx in 0..2u8 {
                cube.set(
                    LocalPos::new_unchecked(base.0 + dx, base.1 + dy, base.2 + dz),
                    planks,
                );
            }
        }
    }
    let changes = pyramid.edit_lod0(
        pos.block(LocalPos::new_unchecked(base.0, base.1, base.2)),
        &cube,
    );
    assert!(
        changes.is_empty(),
        "the not-ready parent is never listed: {changes:?}"
    );

    // The fill readies the parent: take_changed delivers it, with the edit
    // in its content — the filtered result lost nothing.
    let p1 = parent(pos).0;
    pyramid.request(1, p1);
    drain_pyramid(&mut pyramid);
    assert_eq!(pyramid.take_changed(), vec![(1, p1)]);
    let world = pos.block(LocalPos::new_unchecked(base.0, base.1, base.2));
    let (_, cell) = lod_cell(1, world.x, world.y, world.z);
    assert_eq!(
        pyramid.ready(1, p1).unwrap().get(cell),
        planks,
        "the edit must be in the delivered cube"
    );
}

#[test]
fn internal_requests_never_consume_the_streaming_cap() {
    // The pyramid's own fill requests (deferred propagations) do not count
    // in `in_flight` — the number 2.6 caps its own requests with — so a
    // burst of LOD0 ingest during exploration can never crowd streaming's
    // requests out of their budget.
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let mut pyramid = inline_pyramid(4, 64);
    let y = (generator.surface_height(0, 0) >> 5) as i32;
    // A 2×2 footprint of LOD0 cubes: every insert defers (the level-1
    // parent is not ready) and queues an internal fill for it.
    for dx in 0..2i32 {
        for dz in 0..2i32 {
            let pos = CubePos { x: dx, y, z: dz };
            pyramid.insert_lod0(pos, &generator.generate(pos));
        }
    }
    assert!(
        pyramid.internal_requests() > 0,
        "the inserts queued internal fills"
    );
    assert_eq!(
        pyramid.in_flight(),
        0,
        "internal requests must not consume the streaming cap"
    );
    // An external request is still admitted at the cap, and the deferred
    // work still runs to completion (the level-2 cube readies from its
    // real level-1 child plus fills).
    let level2 = CubePos {
        x: 0,
        y: y >> 2,
        z: 0,
    };
    pyramid.request(2, level2);
    assert_eq!(pyramid.in_flight(), 1);
    drain_pyramid(&mut pyramid);
    assert!(pyramid.ready(2, level2).is_some());
    assert_eq!(pyramid.in_flight(), 0);
}

#[test]
fn an_edit_under_a_ready_level_3_cube_reaches_levels_1_to_3() {
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let mut pyramid = inline_pyramid(3, 256);
    let h = generator.surface_height(0, 0);
    // One LOD0 cube on the surface; its chain runs to level 3. Requesting
    // the level-3 cube fills everything the chain needs.
    let pos = CubePos {
        x: 0,
        y: (h >> 5) as i32,
        z: 0,
    };
    let mut cube = generator.generate(pos);
    pyramid.insert_lod0(pos, &cube);
    let p1 = parent(pos).0;
    let p2 = parent(p1).0;
    let p3 = parent(p2).0;
    pyramid.request(3, p3);
    drain_pyramid(&mut pyramid);
    let chain = [p1, p2, p3];
    for (level, &cube_pos) in chain.iter().enumerate() {
        assert!(
            pyramid.ready(level as u8 + 1, cube_pos).is_some(),
            "level {} ready before the edit",
            level + 1
        );
    }
    // A material edit at the surface: the whole 2×2×2 group of the first
    // grass block turns to planks (a single block could not outvote its
    // group), so the level-1 cell flips. The edit must reach level 3 and
    // be reported at every level.
    let surface_local = (0..jbm_core::coords::CUBE_VOLUME)
        .map(LocalPos::from_index)
        .find(|&local| cube.get(local) == blocks().grass_block)
        .expect("a grass block in the surface cube");
    let group_base = (
        surface_local.x & !1,
        surface_local.y & !1,
        surface_local.z & !1,
    );
    for dy in 0..2u8 {
        for dz in 0..2u8 {
            for dx in 0..2u8 {
                let local = LocalPos::new_unchecked(
                    group_base.0 + dx,
                    group_base.1 + dy,
                    group_base.2 + dz,
                );
                cube.set(local, blocks().oak_planks);
                pyramid.edit_lod0(pos.block(local), &cube);
            }
        }
    }
    let changed = pyramid.take_changed();
    for (level, &cube_pos) in chain.iter().enumerate() {
        assert!(
            changed.contains(&(level as u8 + 1, cube_pos)),
            "level {} missing from {changed:?}",
            level + 1
        );
    }
    let world = pos.block(surface_local);
    let (cube1, cell1) = lod_cell(1, world.x, world.y, world.z);
    assert_eq!(cube1, p1);
    assert_eq!(
        pyramid.ready(1, cube1).unwrap().get(cell1),
        blocks().oak_planks,
        "the material edit must show at level 1"
    );
}

#[test]
fn real_data_wins_over_an_in_flight_sample() {
    // Threaded, workers held: the sample job for the parent is in flight
    // while the LOD0 cube lands. The real octant survives the merge.
    let generator = Arc::new(Generator::new(DEFAULT_SEED, blocks()));
    let mut pyramid = LodPyramid::with_workers(1, Arc::clone(&generator), None, 2, 0);
    pyramid.hold_workers();
    let h = generator.surface_height(0, 0);
    let child = CubePos {
        x: 0,
        y: (h >> 5) as i32,
        z: 0,
    };
    let (p1, octant) = parent(child);
    pyramid.request(1, p1);
    // Give the workers a moment to take the job, then insert real data.
    std::thread::sleep(std::time::Duration::from_millis(50));
    let cube = generator.generate(child);
    pyramid.insert_lod0(child, &cube);
    pyramid.release_workers();
    let deadline = Instant::now() + std::time::Duration::from_secs(5);
    while pyramid.ready(1, p1).is_none() {
        pyramid.poll();
        assert!(Instant::now() < deadline, "the held job never finished");
    }
    let merged = pyramid.get(1, p1).unwrap();
    assert_eq!(merged.known, 1 << octant, "the real octant survived");
    assert_eq!(merged.filled, !(1 << octant), "the other seven are filled");
    assert!(merged.ready());
    // The real octant's content is the child's downsample, not the sample.
    let mut expected = Cube::filled(BlockStateId::AIR);
    jbm_core::lod::downsample_octant(&cube, &mut expected, octant);
    let expected = expected;
    assert_octant_equals(
        &merged.cube,
        octant,
        |cell| expected.get(cell),
        "the real octant is the child's downsample",
    );
}

#[test]
fn a_lod_job_panic_reaches_the_calling_thread() {
    let mut pyramid = inline_pyramid(1, 8);
    pyramid.fail_next_job();
    pyramid.request(1, CubePos { x: 0, y: 0, z: 0 });
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pyramid.poll();
    }));
    assert!(panicked.is_err(), "the injected panic must surface");
}

#[test]
fn drop_with_a_deep_lod_backlog_returns_promptly() {
    let start = Instant::now();
    {
        let mut pyramid = LodPyramid::with_workers(
            2,
            Arc::new(Generator::new(DEFAULT_SEED, blocks())),
            None,
            2,
            0,
        );
        for x in 0..64i32 {
            for z in 0..64i32 {
                pyramid.request(1, CubePos { x, y: 0, z });
            }
        }
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < std::time::Duration::from_secs(2),
        "pyramid drop with a deep backlog took {elapsed:?}"
    );
}

// ---------------------------------------------------------------------
// Stored LOD through the Server

/// The default 2.6 disks around spawn: every level's
/// `lod_view_radius`/`lod_vertical_radius` cube set, centered on spawn.
fn ring_positions(generator: &Generator) -> Vec<(u8, CubePos)> {
    let [sx, sy, sz] = generator.spawn();
    let mut out = Vec::new();
    for level in 1..=4u8 {
        let shift = 5 + level as i64;
        let (cx, cy, cz) = (
            (sx as i64) >> shift,
            (sy as i64) >> shift,
            (sz as i64) >> shift,
        );
        let (r, v) = (8i64, 3i64);
        for dy in -v..=v {
            for dx in -r..=r {
                for dz in -r..=r {
                    if dx * dx + dz * dz <= r * r {
                        out.push((
                            level,
                            CubePos {
                                x: (cx + dx) as i32,
                                y: (cy + dy) as i32,
                                z: (cz + dz) as i32,
                            },
                        ));
                    }
                }
            }
        }
    }
    out
}

/// A persistence server in inline mode with a huge poll budget: one tick
/// drains whatever is queued.
fn lod_server(dir: &std::path::Path) -> Server {
    Server::with_config(ServerConfig {
        world_dir: Some(dir.to_owned()),
        gen_threads: 0,
        gen_budget_per_tick: 20_000,
        ..ServerConfig::default()
    })
}

/// Requests `rings` and ticks until every cube is ready. Returns how long
/// that took.
fn request_until_ready(server: &mut Server, rings: &[(u8, CubePos)]) -> std::time::Duration {
    let start = Instant::now();
    for &(level, pos) in rings {
        server.request_lod(level, pos);
    }
    let deadline = Instant::now() + std::time::Duration::from_secs(120);
    loop {
        server.tick();
        if rings
            .iter()
            .all(|&(level, pos)| server.lod_ready(level, pos).is_some())
        {
            return start.elapsed();
        }
        assert!(Instant::now() < deadline, "the rings never became ready");
    }
}

#[test]
fn stored_lod_reloads_without_sampling_and_unchanged() {
    // The default rings around spawn: ready from worldgen once, then —
    // after a flush and reopen — ready from storage alone, with zero
    // samples, and cube for cube the same. Both times and the database
    // size are printed for the task report.
    let dir = tempfile::TempDir::new().unwrap();
    let rings = {
        let generator = Generator::new(DEFAULT_SEED, blocks());
        ring_positions(&generator)
    };
    assert!(rings.len() > 1_000, "{} ring cubes", rings.len());

    let before: std::collections::HashMap<_, _> = {
        let mut server = lod_server(dir.path());
        let from_worldgen = request_until_ready(&mut server, &rings);
        println!(
            "rings ready from worldgen: {from_worldgen:?} ({} cubes)",
            rings.len()
        );
        assert!(server.lod_samples() > 0, "the first pass must sample");
        let snapshot: std::collections::HashMap<_, _> = rings
            .iter()
            .map(|&(level, pos)| {
                (
                    (level, pos),
                    server.lod_ready(level, pos).expect("ready").clone(),
                )
            })
            .collect();
        server.flush();
        snapshot
    }; // dropped: the flush above already persisted everything

    let mut server = lod_server(dir.path());
    let from_storage = request_until_ready(&mut server, &rings);
    println!("rings ready from storage: {from_storage:?}");
    assert_eq!(server.lod_samples(), 0, "a reload must not sample worldgen");
    for &(level, pos) in &rings {
        assert_eq!(
            server.lod_ready(level, pos),
            before
                .get(&(level, pos))
                .map(|cube| cube as &jbm_core::cube::Cube),
            "the reloaded cube changed at ({level}, {pos:?})"
        );
    }
    let storage = server.storage().expect("persistence on");
    println!(
        "database: {} bytes, {} LOD records, {} records total",
        std::fs::metadata(storage.path()).unwrap().len(),
        storage.lod_record_count().unwrap(),
        storage.cube_count().unwrap()
    );
}

#[test]
fn default_rings_ready_times_at_game_scale() {
    // The report's timing numbers, threaded and at game scale: the default
    // rings (a `lod_view_radius`/`lod_vertical_radius` disk at every
    // level) around spawn ready from worldgen, then — after a flush and
    // reopen — from storage alone.
    let dir = tempfile::TempDir::new().unwrap();
    let rings = ring_positions(&Generator::new(DEFAULT_SEED, blocks()));
    assert!(rings.len() > 1_000, "{} ring cubes", rings.len());

    let threaded = || {
        Server::with_config(ServerConfig {
            world_dir: Some(dir.path().to_owned()),
            ..ServerConfig::default() // threaded pools, default rings
        })
    };
    let from_worldgen = {
        let mut server = threaded();
        let start = Instant::now();
        for &(level, pos) in &rings {
            server.request_lod(level, pos);
        }
        let deadline = Instant::now() + std::time::Duration::from_secs(120);
        loop {
            server.tick();
            // Yield: the workers, not this loop, do the filling.
            std::thread::sleep(std::time::Duration::from_millis(1));
            if rings
                .iter()
                .all(|&(level, pos)| server.lod_ready(level, pos).is_some())
            {
                break;
            }
            assert!(Instant::now() < deadline, "the rings never readied");
        }
        let ready = start.elapsed();
        println!(
            "rings ready from worldgen, threaded: {ready:?} ({} cubes, {} samples)",
            rings.len(),
            server.lod_samples()
        );
        server.flush();
        ready
    };
    let mut server = threaded();
    let start = Instant::now();
    for &(level, pos) in &rings {
        server.request_lod(level, pos);
    }
    let deadline = Instant::now() + std::time::Duration::from_secs(120);
    loop {
        server.tick();
        // Yield, as above: the workers do the loading.
        std::thread::sleep(std::time::Duration::from_millis(1));
        if rings
            .iter()
            .all(|&(level, pos)| server.lod_ready(level, pos).is_some())
        {
            break;
        }
        assert!(Instant::now() < deadline, "the stored rings never readied");
    }
    println!(
        "rings ready from storage, threaded: {:?} ({} samples), pyramid heap {} bytes ({:.1} MiB)",
        start.elapsed(),
        server.lod_samples(),
        server.lod_heap_bytes(),
        server.lod_heap_bytes() as f64 / (1024.0 * 1024.0)
    );
    assert_eq!(server.lod_samples(), 0, "a reload must not sample");
    assert!(from_worldgen > std::time::Duration::ZERO);
}

#[test]
fn edits_survive_in_lod_without_the_lod0_cube_loaded() {
    // Edit a pillar, flush, reopen with no player (nothing loads LOD0),
    // and ask for the ancestors: the edit shows from the stored LOD.
    let dir = tempfile::TempDir::new().unwrap();
    let ground = {
        let mut server = Server::with_config(ServerConfig {
            world_dir: Some(dir.path().to_owned()),
            gen_threads: 0,
            ..streaming_config()
        });
        let (mut client, server_end) = in_process_pair();
        client
            .send(ClientMsg::Hello {
                protocol_version: PROTOCOL_VERSION,
                name: "editor".to_owned(),
            })
            .unwrap();
        server.add_connection(server_end);
        for _ in 0..100 {
            server.tick();
            drain(&mut client);
        }
        let ground = BlockPos {
            x: 0,
            y: surface_y(&server),
            z: 0,
        };
        break_block(&mut server, &mut client, ground);
        server.flush();
        ground
    };

    let mut server = lod_server(dir.path());
    assert_eq!(server.world_cube_count(), 0, "no LOD0 cube is loaded");
    let (cube1, cell1) = lod_cell(1, ground.x, ground.y, ground.z);
    server.request_lod(1, cube1);
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    while server.lod_ready(1, cube1).is_none() {
        server.tick();
        assert!(Instant::now() < deadline, "the level-1 cube never readied");
    }
    // The stored LOD0 cube (with the edit) downsamples to exactly the
    // stored level-1 cell.
    let storage = server.storage().unwrap();
    let stored_lod0 = storage
        .load(ground.split().0)
        .unwrap()
        .expect("stored LOD0");
    let mut expected = Cube::filled(BlockStateId::AIR);
    jbm_core::lod::downsample_octant(&stored_lod0, &mut expected, parent(ground.split().0).1);
    let got = server.lod_ready(1, cube1).unwrap();
    assert_eq!(
        got.get(cell1),
        expected.get(cell1),
        "the edit must survive in the stored level-1 cube"
    );
}

#[test]
fn a_lod_cube_changed_while_its_save_is_in_flight_is_saved_again() {
    // The lost-save regression: a ready LOD cube whose content changes
    // again while the save of its previous content is still in flight must
    // stay dirty, so the interval after that save's ack stores the latest
    // content. Dropping it from the dirty set (it was already pending)
    // left the stale record on disk forever.
    let dir = tempfile::TempDir::new().unwrap();
    let glass = blocks().glass;
    let (mut server, mut client) = settled_server(ServerConfig {
        world_dir: Some(dir.path().to_owned()),
        save_interval: std::time::Duration::from_millis(50), // every tick
        gen_threads: 0,
        ..streaming_config()
    });
    // Two 2×2×2 glass groups, each aligned inside one LOD1 cell of the same
    // level-1 cube: placing a whole group flips its cell from air to glass
    // (8 of 8), so each group is one observable content change.
    let ground = surface_y(&server);
    let y0 = {
        let y = ground + 4;
        y + (y & 1) // even, above the surface and the player's box
    };
    let (l1, cell_a) = lod_cell(1, 2, y0, 0);
    let (l1_b, cell_b) = lod_cell(1, 3, y0 + 1, 3);
    assert_eq!(l1, l1_b, "both groups share one level-1 cube");
    assert_ne!(cell_a, cell_b);
    let group = |z0: i64| {
        (y0..y0 + 2)
            .flat_map(|y| (2..4).map(move |x| (x, y)))
            .flat_map(|(x, y)| (z0..z0 + 2).map(move |z| BlockPos { x, y, z }))
            .collect::<Vec<_>>()
    };
    // All eight placements land in one tick, so the interval's snapshot
    // holds the fully flipped cell (not a partial group).
    let place_group = |server: &mut Server, client: &mut ClientLink, blocks: &[BlockPos]| {
        for pos in blocks {
            client
                .send(ClientMsg::PlaceBlock {
                    pos: *pos,
                    state: glass,
                })
                .unwrap();
        }
        server.tick();
        drain(client);
        for pos in blocks {
            assert_eq!(server.block(*pos), Some(glass), "place at {pos:?} rejected");
        }
    };

    // v1: the cube over spawn is ready and its first save has committed.
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    while server
        .storage()
        .unwrap()
        .load_lod(1, l1)
        .expect("load_lod")
        .is_none()
    {
        server.tick();
        drain(&mut client);
        assert!(
            Instant::now() < deadline,
            "the first LOD save never committed"
        );
    }
    let stored_cell = |server: &Server, cell: LocalPos| {
        server
            .storage()
            .unwrap()
            .load_lod(1, l1)
            .expect("load_lod")
            .expect("the record exists")
            .0
            .get(cell)
    };

    // Hold the saver; v2 (group A) goes into a batch that never commits
    // until released; v3 (group B) changes the cube again while that save
    // is in flight.
    let held = server.storage().expect("persistence on").clone();
    held.hold_saves();
    place_group(&mut server, &mut client, &group(0));
    assert!(server.unanswered_batches() > 0, "the v2 batch is in flight");
    // Let the saver take the held batch, as 2.3's test does.
    std::thread::sleep(std::time::Duration::from_millis(10));
    place_group(&mut server, &mut client, &group(2));
    for _ in 0..3 {
        server.tick(); // intervals snapshot (and skip) the pending key
        drain(&mut client);
    }

    // Releasing one save commits v2 only; the record must show group A's
    // glass and not group B's.
    held.release_one_save();
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    while stored_cell(&server, cell_a) != glass {
        server.tick();
        drain(&mut client);
        assert!(
            Instant::now() < deadline,
            "the held v2 batch never committed"
        );
    }
    assert_ne!(
        stored_cell(&server, cell_b),
        glass,
        "the v2 record must predate group B"
    );

    // The rest of the saver runs; the cube was changed during the save, so
    // a later interval must store v3. With the lost-save bug the key was
    // gone from the dirty set and this never happened.
    held.release_saves();
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    while stored_cell(&server, cell_b) != glass {
        server.tick();
        drain(&mut client);
        assert!(
            Instant::now() < deadline,
            "the change made during the save was never stored"
        );
    }

    // And the stored record is exactly the live cube: flush, reopen, and
    // load it back without a single worldgen sample. The held storage Arc
    // goes first — it keeps the database file open.
    let live = server.lod_ready(1, l1).expect("ready").clone();
    drop(held);
    drop(client);
    server.flush();
    drop(server);
    let mut reopened = lod_server(dir.path());
    reopened.request_lod(1, l1);
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    while reopened.lod_ready(1, l1).is_none() {
        reopened.tick();
        assert!(Instant::now() < deadline, "the reopened cube never readied");
    }
    assert_eq!(reopened.lod_ready(1, l1), Some(&live));
    assert_eq!(reopened.lod_samples(), 0, "the reload must not sample");
}

#[test]
fn a_corrupt_lod_record_is_resampled_never_a_panic() {
    let dir = tempfile::TempDir::new().unwrap();
    let pos = {
        let generator = Generator::new(DEFAULT_SEED, blocks());
        let h = generator.surface_height(0, 0);
        CubePos {
            x: 0,
            y: (h >> 6) as i32,
            z: 0,
        }
    };
    {
        let mut server = lod_server(dir.path());
        server.request_lod(1, pos);
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while server.lod_ready(1, pos).is_none() {
            server.tick();
            assert!(Instant::now() < deadline);
        }
        server.flush();
    }
    // Tear the stored record, as a half-written batch would.
    lod_server(dir.path())
        .storage()
        .unwrap()
        .damage_lod_record(1, pos)
        .unwrap();
    let mut server = lod_server(dir.path());
    server.request_lod(1, pos);
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    while server.lod_ready(1, pos).is_none() {
        server.tick(); // the corrupt record is logged, the cube resampled
        assert!(
            Instant::now() < deadline,
            "the corrupt record stalled the fill"
        );
    }
    assert!(server.lod_ready(1, pos).is_some());
}

#[test]
fn far_terrain_needs_no_full_detail() {
    // With a slice of the rings ready around spawn (every eighth cube, all
    // levels — the full-ring pass is the reload test's job), the server
    // holds exactly the LOD0 cubes streaming alone loads: LOD jobs never
    // generate LOD0.
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let rings: Vec<_> = ring_positions(&generator).into_iter().step_by(8).collect();
    assert!(rings.len() > 500, "{} ring cubes", rings.len());

    let mut lodless = Server::with_config(ServerConfig {
        lod_levels: 0,
        gen_threads: 0,
        ..streaming_config()
    });
    let (mut client, server_end) = in_process_pair();
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "baseline".to_owned(),
        })
        .unwrap();
    lodless.add_connection(server_end);
    for _ in 0..300 {
        lodless.tick();
        drain(&mut client);
    }
    let baseline = lodless.world_cube_count();
    drop(client);
    drop(lodless);
    assert!(baseline > 1);

    let mut server = Server::with_config(ServerConfig {
        gen_threads: 0,
        ..streaming_config()
    });
    let (mut client, server_end) = in_process_pair();
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "far-view".to_owned(),
        })
        .unwrap();
    server.add_connection(server_end);
    for _ in 0..300 {
        server.tick();
        drain(&mut client);
    }
    let _ = request_until_ready(&mut server, &rings);
    assert_eq!(
        server.world_cube_count(),
        baseline,
        "the far terrain loaded extra LOD0 cubes"
    );
    assert!(
        server.lod_cube_count() > 0,
        "the pyramid holds the far view"
    );
}

#[test]
fn lod0_generation_is_not_delayed_by_a_lod_backlog() {
    // The LOD workers are held with a deep LOD backlog queued; LOD0
    // streaming (a separate pool) still settles within a few ticks.
    let mut server = Server::with_config(ServerConfig {
        gen_threads: 2,
        view_radius: 3,
        vertical_radius: 1,
        ..streaming_config()
    });
    server.lod_pyramid().hold_workers();
    let generator = Generator::new(DEFAULT_SEED, blocks());
    for (level, pos) in ring_positions(&generator).into_iter().take(600) {
        server.request_lod(level, pos);
    }
    assert!(server.lod_in_flight() > 500, "the backlog is queued");
    let (mut client, server_end) = in_process_pair();
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "walker".to_owned(),
        })
        .unwrap();
    server.add_connection(server_end);
    // The small view streams in full while the LOD jobs are stuck (the
    // workers are held, not busy): a few LOD0 generations' worth of
    // ticks, well inside what the backlog could ever delay.
    // The desired set is a disk (dx² + dz² ≤ r²), |dy| ≤ v: 29 columns
    // of 3 cubes for r = 3, v = 1.
    let columns = (-3i32..=3)
        .flat_map(|dx| (-3i32..=3).map(move |dz| (dx, dz)))
        .filter(|(dx, dz)| dx * dx + dz * dz <= 9)
        .count();
    let desired = columns * 3;
    let deadline = Instant::now() + std::time::Duration::from_secs(60);
    while server.world_cube_count() < desired {
        server.tick();
        drain(&mut client);
        // Yield: the generation workers, not this loop, do the work.
        std::thread::sleep(std::time::Duration::from_millis(2));
        assert!(
            Instant::now() < deadline,
            "LOD0 streaming starved by the LOD backlog: {} of {desired}",
            server.world_cube_count()
        );
    }
    assert!(server.lod_in_flight() > 0, "the backlog is still held");
    server.lod_pyramid().release_workers();
}

// ---------------------------------------------------------------------
// Edit while the child fills, after a reload (review finding 1)

/// Opens the world's storage exactly as the server does: register the
/// builtin blocks, open (creating) the database, declare the worldgen
/// fingerprint so stored LOD records are usable.
fn open_lod_storage(dir: &std::path::Path) -> Arc<Storage> {
    let mut registry = BlockRegistry::new();
    let _ = Blocks::register(&mut registry);
    let storage = Storage::open(dir, &registry, DEFAULT_SEED).unwrap();
    storage
        .set_worldgen_fingerprint(jbm_server::worldgen::overworld().fingerprint())
        .unwrap();
    Arc::new(storage)
}

#[test]
fn edit_while_child_fills_after_reload_keeps_the_reference_content() {
    // The lost-edit regression: a stored level-2 record loads with a real
    // (known) octant; the LOD0 cube below re-enters the world, so its
    // level-1 child gains real data but is not ready yet; an edit lands in
    // between. The parent's octant is already known, so the old defer path
    // set no pending bit — and when the child readied, the propagation took
    // the already-known branch and never recomputed, leaving the edit lost
    // from the parent forever. After the pipeline drains, every ready cube
    // must equal the reference.
    let dir = tempfile::TempDir::new().unwrap();
    let generator = Arc::new(Generator::new(DEFAULT_SEED, blocks()));
    let h = generator.surface_height(0, 0);
    let lod0_pos = CubePos {
        x: 0,
        y: (h >> 5) as i32,
        z: 0,
    };
    let p1 = parent(lod0_pos).0;
    let (p2, octant) = parent(p1);
    let cube_v1 = generator.generate(lod0_pos);

    // Run 1: build the chain, fill it, store it. One inline job per poll
    // keeps the interleaving below deterministic.
    {
        let storage = open_lod_storage(dir.path());
        let mut pyramid =
            LodPyramid::with_workers(2, Arc::clone(&generator), Some(Arc::clone(&storage)), 0, 1);
        pyramid.insert_lod0(lod0_pos, &cube_v1); // defers; queues the child's fill
        pyramid.request(2, p2);
        pyramid.poll(); // job 1: the level-1 child fills and readies
        pyramid.poll(); // job 2: the level-2 cube fills and readies
        assert!(pyramid.ready(1, p1).is_some());
        assert!(pyramid.ready(2, p2).is_some());
        let dirty = pyramid.take_store_dirty();
        assert_eq!(dirty, vec![(1, p1), (2, p2)], "both cubes became ready");
        let records: Vec<(u8, CubePos, Cube, u8)> = dirty
            .iter()
            .map(|&(level, pos)| {
                let lc = pyramid.get(level, pos).unwrap();
                (level, pos, lc.cube.clone(), lc.known)
            })
            .collect();
        let lod_refs: Vec<(u8, CubePos, &Cube, u8)> = records
            .iter()
            .map(|record| (record.0, record.1, &record.2, record.3))
            .collect();
        storage.save_lod_batch(&[], &lod_refs, &[]).unwrap();
    }

    // Run 2 (the restart): the stored level-2 cube loads first — octant
    // real — then the LOD0 cube re-enters the world and an edit lands
    // before the child's internal fill completes.
    let mut pyramid = LodPyramid::with_workers(
        2,
        Arc::clone(&generator),
        Some(open_lod_storage(dir.path())),
        0,
        1,
    );
    pyramid.request(2, p2);
    pyramid.poll(); // the stored record loads: octant real, cube ready
    assert!(pyramid.ready(2, p2).is_some());
    assert_eq!(
        pyramid.get(2, p2).unwrap().known,
        1 << octant,
        "the stored octant is real"
    );

    pyramid.insert_lod0(lod0_pos, &cube_v1);
    assert!(
        !pyramid.get(1, p1).unwrap().ready(),
        "the child is not ready yet"
    );
    assert_eq!(
        pyramid.in_flight_total(),
        1,
        "the child's internal fill is queued"
    );

    // The edit: a 4×4×4 planks platform, aligned to the level-2 cell grid
    // (local coordinates multiples of 4), so it flips the one level-2 cell
    // covering it — 8 of 8 level-1 children planks — not just level 1.
    // One edit call per 2×2×2 group, each with the full post-edit cube.
    let base = (16u8, 24u8, 16u8);
    let mut cube_v2 = cube_v1.clone();
    for dy in 0..4u8 {
        for dz in 0..4u8 {
            for dx in 0..4u8 {
                cube_v2.set(
                    LocalPos::new_unchecked(base.0 + dx, base.1 + dy, base.2 + dz),
                    blocks().oak_planks,
                );
            }
        }
    }
    for gy in 0..2u8 {
        for gz in 0..2u8 {
            for gx in 0..2u8 {
                let local =
                    LocalPos::new_unchecked(base.0 + 2 * gx, base.1 + 2 * gy, base.2 + 2 * gz);
                pyramid.edit_lod0(lod0_pos.block(local), &cube_v2);
            }
        }
    }

    // The pipeline drains: the child readies and the parent's real octant
    // must be recomputed from the child's full content.
    drain_pyramid(&mut pyramid);

    let rebuilt = LodPyramid::rebuild(2, &generator, std::iter::once((lod0_pos, &cube_v2)));
    assert_pyramids_equal(&pyramid, &rebuilt, true, "edit under a stored octant");
    assert!(
        pyramid.take_store_dirty().contains(&(2, p2)),
        "the healed parent must be re-saved"
    );
}

// ---------------------------------------------------------------------
// Kill during save (task 2.7's stored-LOD half, as 2.3's crash test)

/// The child role selector; without it the children do nothing.
const KILL_CHILD_VAR: &str = "JBM_LOD_KILL_CHILD";
const KILL_DIR_VAR: &str = "JBM_LOD_KILL_WORLD_DIR";

/// Runs this test binary as the kill child on `dir`, waits for its
/// "started" line, and returns it.
fn spawn_kill_child(dir: &std::path::Path) -> std::process::Child {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};
    let exe = std::env::current_exe().expect("current test binary");
    let mut child = Command::new(exe)
        .arg("lod_kill_child")
        .arg("--exact")
        .arg("--nocapture")
        .env(KILL_CHILD_VAR, "1")
        .env(KILL_DIR_VAR, dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // a child panic must be visible
        .spawn()
        .expect("spawning the kill child");
    {
        let stdout = child.stdout.take().expect("piped stdout");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            while let Ok(n) = reader.read_line(&mut line) {
                if n == 0 {
                    break;
                }
                if line.trim() == "started" {
                    let _ = tx.send(());
                    break;
                }
                line.clear();
            }
        });
        let deadline = Instant::now() + std::time::Duration::from_secs(20);
        while rx.try_recv().is_err() {
            assert!(Instant::now() < deadline, "child never reported startup");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
    child
}

#[test]
#[cfg(unix)]
fn kill_during_save_leaves_lod_records_consistent_with_lod0() {
    // As 2.3's crash test: several kills at different seeded delays, so
    // some land mid-save. After every kill the world reopens, every stored
    // LOD record decodes, and every real octant whose LOD0 cube is also
    // stored agrees with its downsample.
    let dir = tempfile::TempDir::new().unwrap();
    let mut rng = XorShift64(0x4C4F_4453_4156_4573);
    for round in 1..=10u32 {
        let mut child = spawn_kill_child(dir.path());
        let ms = 20 + rng.next() % 381;
        std::thread::sleep(std::time::Duration::from_millis(ms));
        child.kill().expect("SIGKILL the child");
        let _ = child.wait();

        let server = lod_server(dir.path());
        let storage = server.storage().expect("persistence on");
        let lod_keys = storage.lod_positions().expect("lod positions");
        assert!(
            !lod_keys.is_empty(),
            "round {round}: some LOD records were committed"
        );
        for (level, pos) in &lod_keys {
            let (cube, _known) = storage
                .load_lod(*level, *pos)
                .unwrap_or_else(|e| {
                    panic!("round {round}: LOD {level} {pos:?} failed to decode: {e}")
                })
                .unwrap_or_else(|| panic!("round {round}: LOD {level} {pos:?} vanished"));
            if *level == 1 {
                for octant in 0..8u8 {
                    let [dx, dy, dz] = octant_offset(octant).map(i32::from);
                    let child = CubePos {
                        x: pos.x * 2 + dx,
                        y: pos.y * 2 + dy,
                        z: pos.z * 2 + dz,
                    };
                    if let Some(lod0) = storage.load(child).expect("lod0 load") {
                        let mut expected = Cube::filled(BlockStateId::AIR);
                        jbm_core::lod::downsample_octant(&lod0, &mut expected, octant);
                        assert_octant_equals(
                            &cube,
                            octant,
                            |cell| expected.get(cell),
                            "round {round}: stored LOD disagrees with stored LOD0",
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn lod_kill_child() {
    // The child role: settle a streaming player, make an edit, ask for a
    // few LOD cubes, then tick (and save) until killed.
    if std::env::var(KILL_CHILD_VAR).as_deref() != Ok("1") {
        return;
    }
    use std::io::Write as _;
    let dir = std::env::var(KILL_DIR_VAR).expect("child gets the world dir");
    let mut server = Server::with_config(ServerConfig {
        world_dir: Some(std::path::PathBuf::from(dir)),
        save_interval: std::time::Duration::from_millis(50),
        view_radius: 2,
        vertical_radius: 1,
        gen_threads: 0,
        gen_budget_per_tick: 512,
        ..ServerConfig::default()
    });
    let (mut client, server_end) = in_process_pair();
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "killer".to_owned(),
        })
        .unwrap();
    server.add_connection(server_end);
    for _ in 0..100 {
        server.tick();
        drain(&mut client);
    }
    // One fresh edit per round, always within reach of the restored
    // player: alternating break/place on the spawn surface block (its y
    // from the generator, since the stored column top moves with the
    // edits). A fixed deeper dig would walk out of reach round by round.
    let ground = BlockPos {
        x: 0,
        y: Generator::new(DEFAULT_SEED, blocks()).surface_height(0, 0),
        z: 0,
    };
    if server.block(ground) == Some(BlockStateId::AIR) {
        place(&mut server, &mut client, ground, blocks().glass);
    } else {
        break_block(&mut server, &mut client, ground);
    }
    let generator = Generator::new(DEFAULT_SEED, blocks());
    let h = generator.surface_height(0, 0);
    for level in 1..=4u8 {
        let shift = 5 + level as i64;
        server.request_lod(
            level,
            CubePos {
                x: 0,
                y: (h >> shift) as i32,
                z: 0,
            },
        );
    }
    println!("started");
    std::io::stdout().flush().unwrap();
    loop {
        server.tick();
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}
