//! LOD pyramid tests (task 2.4): thin features surviving the levels, edit
//! propagation through the real `Server` paths, incremental-vs-rebuild
//! equality, and the ingest/eviction rules.

use std::collections::HashMap;

use jbm_core::block::{BlockRegistry, BlockStateId, builtin::Blocks};
use jbm_core::coords::{BlockPos, CUBE_SIZE, CubePos, LocalPos};
use jbm_core::cube::Cube;
use jbm_core::link::{ClientLink, in_process_pair};
use jbm_core::lod::parent;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION, ServerMsg};
use jbm_server::lod::LodPyramid;
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

fn assert_pyramids_equal(incremental: &LodPyramid, rebuilt: &LodPyramid, what: &str) {
    assert_eq!(
        incremental.max_levels(),
        rebuilt.max_levels(),
        "{what}: level counts differ"
    );
    for level in 1..=incremental.max_levels() {
        let a: HashMap<CubePos, (u8, Cube)> = incremental
            .iter(level)
            .map(|(&pos, lc)| (pos, (lc.known, lc.cube.clone())))
            .collect();
        let b: HashMap<CubePos, (u8, Cube)> = rebuilt
            .iter(level)
            .map(|(&pos, lc)| (pos, (lc.known, lc.cube.clone())))
            .collect();
        assert_eq!(a.len(), b.len(), "{what}: level {level} cube counts differ");
        for (pos, (known, cube)) in &a {
            let other = b
                .get(pos)
                .unwrap_or_else(|| panic!("{what}: level {level} misses {pos:?}"));
            assert_eq!(known, &other.0, "{what}: level {level} {pos:?} known mask");
            assert_eq!(cube, &other.1, "{what}: level {level} {pos:?} content");
        }
    }
}

// ---------------------------------------------------------------------
// Thin features

#[test]
fn one_block_wall_survives_lod1_and_lod2_at_both_parities() {
    let planks = blocks().oak_planks;
    for wall_x in [7u8, 8u8] {
        let mut pyramid = LodPyramid::new(4);
        // A wall 1 block thick, 32 long and 16 tall, otherwise air.
        let cube = lod0_cube(|x, _y, _z| {
            if x == wall_x {
                planks
            } else {
                BlockStateId::AIR
            }
        });
        pyramid.insert_lod0(CubePos { x: 0, y: 0, z: 0 }, &cube);

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
    // A 1×1 trunk, 8 tall, surrounded by air.
    let cube = lod0_cube(|x, y, z| {
        if (x, z) == (7, 7) && y < 8 {
            log
        } else {
            BlockStateId::AIR
        }
    });
    pyramid.insert_lod0(CubePos { x: 0, y: 0, z: 0 }, &cube);

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
    // z = 0 cube borders.
    let planks = blocks().oak_planks;
    let mut pyramid = LodPyramid::new(2);
    for (cube_x, wall_x) in [(-1i32, 31u8), (0, 0u8)] {
        for cube_z in [-1i32, 0] {
            let cube = lod0_cube(|x, _y, z| {
                // Cube -1 covers world z -32..0: its z >= 16 half is world
                // z -16..0. Cube 0 covers 0..32: its z < 16 half is 0..16.
                // Together the wall spans world z -16..16.
                let in_wall_z = if cube_z == -1 { z >= 16 } else { z < 16 };
                if x == wall_x && in_wall_z {
                    planks
                } else {
                    BlockStateId::AIR
                }
            });
            pyramid.insert_lod0(
                CubePos {
                    x: cube_x,
                    y: 0,
                    z: cube_z,
                },
                &cube,
            );
        }
    }
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
fn inserting_a_cube_marks_its_whole_ancestor_chain_known() {
    let stone = blocks().stone;
    let mut pyramid = LodPyramid::new(4);
    let pos = CubePos { x: 1, y: 2, z: 3 };
    pyramid.insert_lod0(pos, &Cube::filled(stone));

    let mut child = pos;
    for level in 1..=4u8 {
        let (ancestor, octant) = parent(child);
        let lod = pyramid.get(level, ancestor).expect("chain cube exists");
        assert_eq!(lod.known, 1 << octant, "level {level} known mask");
        child = ancestor;
    }
    assert_eq!(pyramid.total_cubes(), 4);

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
    let mut pyramid = LodPyramid::new(4);
    let first = pyramid.insert_lod0(pos, &cube);
    assert!(!first.is_empty(), "the first insert changes content");
    let second = pyramid.insert_lod0(pos, &cube);
    assert!(second.is_empty(), "re-inserting changed {second:?}");
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
    for level in 1..=3u8 {
        assert_eq!(pyramid.len(level), 1, "one chain cube at level {level}");
    }
    assert_eq!(pyramid.total_cubes(), 3);
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
fn incremental_edits_equal_rebuild_exactly() {
    // Seeded from JBM_LOD_TEST_SEED (hex) so acceptance runs can vary it;
    // fixed by default so the suite is deterministic.
    let seed = std::env::var("JBM_LOD_TEST_SEED")
        .ok()
        .and_then(|s| u64::from_str_radix(&s, 16).ok())
        .unwrap_or(0x10D5_EED1);
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
    let h = generator.surface_height(0, 0);
    let y_base = (h >> 5) as i32 - 1;

    // A 4×2×4 region of generated cubes straddling the origin, so edits
    // land on cube borders and at negative coordinates.
    let region: Vec<CubePos> = (-2..2)
        .flat_map(|x| (y_base..y_base + 2).map(move |y| (x, y)))
        .flat_map(|(x, y)| (-2..2).map(move |z| CubePos { x, y, z }))
        .collect();
    assert_eq!(region.len(), 32);
    let mut lod0: HashMap<CubePos, Cube> = region
        .iter()
        .map(|&pos| (pos, generator.generate(pos)))
        .collect();

    let mut pyramid = LodPyramid::new(4);
    for &pos in &region {
        pyramid.insert_lod0(pos, &lod0[&pos]);
    }

    for edit in 0..2_000u64 {
        let pos = region[rng.below(region.len() as u64) as usize];
        let local = LocalPos::new_unchecked(
            rng.below(CUBE_SIZE as u64) as u8,
            rng.below(CUBE_SIZE as u64) as u8,
            rng.below(CUBE_SIZE as u64) as u8,
        );
        let state = states[rng.below(states.len() as u64) as usize];
        let cube = lod0.get_mut(&pos).expect("region cube");
        cube.set(local, state);
        pyramid.edit_lod0(pos.block(local), cube);

        if edit % 100 == 99 {
            let rebuilt = LodPyramid::rebuild(4, region.iter().map(|&pos| (pos, &lod0[&pos])));
            assert_pyramids_equal(&pyramid, &rebuilt, "after edit {edit}");
        }
    }
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
