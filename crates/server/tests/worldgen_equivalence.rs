//! Task 2.1 acceptance core: the graph generator must reproduce the task
//! 1.5 reference terrain exactly — every cube, every surface height — and
//! `sample_grid` must agree with the solidity those cubes encode.

use std::collections::BTreeMap;

use jbm_core::block::BlockRegistry;
use jbm_core::block::BlockStateId;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CubePos, LocalPos};
use jbm_server::DEFAULT_SEED;
use jbm_server::terrain::{Generator, MAX_HEIGHT, MIN_HEIGHT};
use jbm_server::worldgen::reference::ReferenceGenerator;
use jbm_server::worldgen::{GRID_SIDE, Graph, Node, NodeValue, Outputs, Preset, Salt, compile};

fn blocks() -> Blocks {
    let mut registry = BlockRegistry::new();
    Blocks::register(&mut registry)
}

/// The graph side of every equivalence test here is the **classic** preset:
/// it is the graph that re-expresses the task 1.5 reference terrain
/// bit-for-bit (the overworld is a different world, asserted in
/// tests/overworld.rs).
fn graph_generator(seed: u64) -> Generator {
    Generator::with_preset(seed, blocks(), Preset::Classic)
}

fn reference_generator(seed: u64) -> ReferenceGenerator {
    ReferenceGenerator::new(seed, blocks())
}

fn generator_for(seed: u64, outputs: Outputs, nodes: BTreeMap<String, NodeValue>) -> Generator {
    let graph = compile(&Graph {
        version: 1,
        nodes,
        outputs,
    })
    .expect("test graph compiles");
    Generator::from_graph(seed, blocks(), &graph)
}

fn final_node(node: Node) -> Outputs {
    Outputs {
        final_density: NodeValue::Inline(Box::new(node)),
    }
}

/// The block at world `(x, y, z)`, through the cube that contains it.
fn block_at(terrain: &Generator, x: i64, y: i64, z: i64) -> BlockStateId {
    let cube = terrain.generate(CubePos {
        x: (x >> 5) as i32,
        y: (y >> 5) as i32,
        z: (z >> 5) as i32,
    });
    cube.get(LocalPos::new((x & 31) as u8, (y & 31) as u8, (z & 31) as u8).unwrap())
}

#[test]
fn graph_cubes_equal_reference_cubes_exactly() {
    for seed in [DEFAULT_SEED, 42, 0x0BAD_C0DE] {
        let graph = graph_generator(seed);
        let reference = reference_generator(seed);
        for cx in -6..6 {
            for cz in -6..6 {
                for cy in 0..4 {
                    let pos = CubePos {
                        x: cx,
                        y: cy,
                        z: cz,
                    };
                    assert_eq!(
                        graph.generate(pos),
                        reference.generate(pos),
                        "graph cube differs at {pos:?} (seed {seed})"
                    );
                }
            }
        }
    }
}

#[test]
fn graph_surface_heights_equal_reference_heights_exactly() {
    let graph = graph_generator(DEFAULT_SEED);
    let reference = reference_generator(DEFAULT_SEED);
    // Every block of a 512×512 area centred on the origin.
    for x in -256..256 {
        for z in -256..256 {
            assert_eq!(
                graph.surface_height(x, z),
                reference.surface_height(x, z),
                "at ({x}, {z})"
            );
        }
    }
}

#[test]
fn graph_surface_heights_equal_reference_heights_far_away() {
    let graph = graph_generator(DEFAULT_SEED);
    let reference = reference_generator(DEFAULT_SEED);
    // 1,000 pseudo-random columns out to |x|, |z| ≈ 1e9 (deterministic LCG).
    let mut state = 0x5EED_2026_0000_0001u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };
    for _ in 0..1000 {
        let x = ((next() % 2_000_000_001) as i64) - 1_000_000_000;
        let z = ((next() % 2_000_000_001) as i64) - 1_000_000_000;
        assert_eq!(
            graph.surface_height(x, z),
            reference.surface_height(x, z),
            "at ({x}, {z})"
        );
    }
}

#[test]
fn sample_grid_step_one_signs_match_generated_solidity() {
    let compiled = Preset::Classic.compiled();
    let mut out = Box::new([0.0; GRID_SIDE * GRID_SIDE * GRID_SIDE]);
    for seed in [DEFAULT_SEED, 42] {
        let terrain = graph_generator(seed);
        for (cx, cy, cz) in [(0, 2, 0), (-3, 1, 2), (5, 2, -5)] {
            let origin = [i64::from(cx) * 32, i64::from(cy) * 32, i64::from(cz) * 32];
            compiled.sample_grid(seed, origin, 1, &mut out);
            let cube = terrain.generate(CubePos {
                x: cx,
                y: cy,
                z: cz,
            });
            for k in 0..GRID_SIDE {
                for j in 0..GRID_SIDE {
                    for i in 0..GRID_SIDE {
                        let density = out[(k * GRID_SIDE + j) * GRID_SIDE + i];
                        let local = LocalPos::new(i as u8, j as u8, k as u8).unwrap();
                        let solid = cube.get(local) != BlockStateId::AIR;
                        assert_eq!(
                            density >= 0.0,
                            solid,
                            "sign mismatch at {origin:?} + ({i}, {j}, {k}) (seed {seed})"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn a_flat_graph_makes_a_flat_world_with_the_right_layering() {
    let blocks = blocks();
    // The spec's literal `y - 70`: solid at and above 70, air below — flat,
    // every solid cell has a saturated run above it, so it is all stone.
    let inverted = generator_for(
        7,
        final_node(Node::Sub {
            a: y_node(),
            b: NodeValue::Const(70.0),
        }),
        BTreeMap::new(),
    );
    assert_eq!(inverted.surface_height(10, -10), MAX_HEIGHT);
    assert_eq!(block_at(&inverted, 10, 71, -10), blocks.stone);
    assert_eq!(block_at(&inverted, 10, 70, -10), blocks.stone);
    assert_eq!(block_at(&inverted, 10, 69, -10), BlockStateId::AIR);

    // The heightmap-shaped reading `70 - y`: a classic flat world with its
    // surface at 70, above the sand level, so grass over dirt over stone.
    let terrain = generator_for(
        7,
        final_node(Node::Sub {
            a: NodeValue::Const(70.0),
            b: y_node(),
        }),
        BTreeMap::new(),
    );
    assert_eq!(terrain.surface_height(10, -10), 70);
    assert_eq!(block_at(&terrain, 10, 71, -10), BlockStateId::AIR);
    assert_eq!(block_at(&terrain, 10, 70, -10), blocks.grass_block);
    assert_eq!(block_at(&terrain, 10, 69, -10), blocks.dirt);
    assert_eq!(block_at(&terrain, 10, 67, -10), blocks.dirt);
    assert_eq!(block_at(&terrain, 10, 66, -10), blocks.stone);
}

#[test]
fn surface_height_follows_the_graph_band_outside_the_shipped_range() {
    // Flat worlds far above and below the shipped graph's [36, 92]: a banded
    // graph's surface comes from its own band, not MIN/MAX_HEIGHT.
    for level in [150.0, 10.0, -300.0] {
        let terrain = generator_for(
            3,
            final_node(Node::Sub {
                a: NodeValue::Const(level),
                b: y_node(),
            }),
            BTreeMap::new(),
        );
        assert_eq!(terrain.surface_height(5, -5), level as i64, "level {level}");
    }
}

fn y_node() -> NodeValue {
    NodeValue::Inline(Box::new(Node::Y))
}

#[test]
fn an_overhang_graph_gets_grass_on_top_and_stone_underneath() {
    // A spline over y: ground to 49, a gap from 50 to 59, a floating slab
    // from 60 to 69, air above 70.
    let outputs = final_node(Node::Spline {
        input: y_node(),
        points: vec![
            [0.0, 10.0],
            [49.5, 10.0],
            [50.0, -10.0],
            [59.0, -10.0],
            [60.0, 10.0],
            [69.0, 10.0],
            [69.5, -10.0],
            [128.0, -10.0],
        ],
    });
    let terrain = generator_for(7, outputs, BTreeMap::new());
    let blocks = blocks();
    // The gap is really empty and the slab is really floating.
    assert_eq!(block_at(&terrain, 4, 55, 4), BlockStateId::AIR);
    // Grass on top of the overhang (69 > 61, so no sand).
    assert_eq!(block_at(&terrain, 4, 69, 4), blocks.grass_block);
    assert_eq!(block_at(&terrain, 4, 68, 4), blocks.dirt);
    // Stone under the slab: the run above 60 is longer than 4.
    assert_eq!(block_at(&terrain, 4, 60, 4), blocks.stone);
    // The ground's own surface (49 <= 61) is sand, like a beach column.
    assert_eq!(block_at(&terrain, 4, 49, 4), blocks.sand);
    assert_eq!(block_at(&terrain, 4, 46, 4), blocks.sand);
    assert_eq!(block_at(&terrain, 4, 45, 4), blocks.stone);
    // And the surface scan finds the slab, not the ground, plus stays in
    // range.
    assert_eq!(terrain.surface_height(4, 4), 69);
    assert!(terrain.surface_height(4, 4) <= MAX_HEIGHT);
}

#[test]
fn an_unbanded_graph_still_generates_consistently() {
    // final_density = noise2 - 0.3: no y dependence anywhere, so the
    // interval bounds cannot prove a solidity band and the generator must
    // fall back to direct per-row evaluation.
    let outputs = final_node(Node::Sub {
        a: NodeValue::Inline(Box::new(Node::Noise2 {
            salt: Salt(1),
            x_scale: 0.05,
            z_scale: 0.05,
            x_offset: None,
            z_offset: None,
        })),
        b: NodeValue::Const(0.3),
    });
    let graph = compile(&Graph {
        version: 1,
        nodes: BTreeMap::new(),
        outputs,
    })
    .unwrap();
    assert!(
        graph.band().is_none(),
        "noise cannot be banded by intervals"
    );
    let terrain = Generator::from_graph(9, blocks(), &graph);
    for (x, z) in [(0i64, 0i64), (7, -3), (-20, 11)] {
        let solid = graph.sample(9, [x, 64, z]) >= 0.0;
        // The column is y-independent: every row in the window is solid or
        // none is, so the surface sits at one end of the documented range.
        let expected_height = if solid { MAX_HEIGHT } else { MIN_HEIGHT - 1 };
        assert_eq!(
            terrain.surface_height(x, z),
            expected_height,
            "at ({x}, {z})"
        );
        for y in 36..=92i64 {
            let expected = graph.sample(9, [x, y, z]) >= 0.0;
            assert_eq!(
                block_at(&terrain, x, y, z) != BlockStateId::AIR,
                expected,
                "solidity at ({x}, {y}, {z})"
            );
        }
    }
}
