//! Criterion benches for the LOD pyramid (task 2.4): inserting a generated
//! surface cube (and an air cube) into an existing pyramid through all 4
//! levels, and one edit through all levels. The setup also reports, once,
//! the pyramid's size after streaming the default view around spawn.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use jbm_core::block::BlockRegistry;
use jbm_core::block::builtin::Blocks;
use jbm_core::coords::{CubePos, LocalPos};
use jbm_core::link::in_process_pair;
use jbm_core::protocol::{ClientMsg, PROTOCOL_VERSION};
use jbm_server::lod::LodPyramid;
use jbm_server::terrain::Generator;
use jbm_server::{DEFAULT_SEED, Server, ServerConfig};
use std::hint::black_box;

fn generator() -> Generator {
    let mut registry = BlockRegistry::new();
    let blocks = Blocks::register(&mut registry);
    Generator::new(DEFAULT_SEED, blocks)
}

/// A cube position whose column intersects the surface near the origin.
fn surface_pos(generator: &Generator) -> CubePos {
    let h = generator.surface_height(0, 0);
    CubePos {
        x: 0,
        y: (h >> 5) as i32,
        z: 0,
    }
}

/// The pyramid holding the seven siblings of `target`'s parent octant (plus
/// their chains), so the measured insert lands in an existing pyramid with
/// a non-uniform level-1 parent.
fn sibling_pyramid(generator: &Generator, target: CubePos) -> LodPyramid {
    let mut pyramid = LodPyramid::new(4);
    for dx in 0..2i32 {
        for dy in 0..2i32 {
            for dz in 0..2i32 {
                if (dx, dy, dz) == (target.x & 1, target.y & 1, target.z & 1) {
                    continue;
                }
                let pos = CubePos {
                    x: target.x - (target.x & 1) + dx,
                    y: target.y - (target.y & 1) + dy,
                    z: target.z - (target.z & 1) + dz,
                };
                let cube = generator.generate(pos);
                pyramid.insert_lod0(pos, &cube);
            }
        }
    }
    pyramid
}

/// Prints the LOD heap bytes after streaming the default view around spawn
/// (next to the LOD0 world's cube count) — the size half of the task's
/// bench report, measured once outside any timing.
fn report_default_view_heap() {
    let mut server = Server::with_config(ServerConfig {
        gen_threads: 0,
        ..ServerConfig::default()
    });
    let (mut client, server_end) = in_process_pair();
    client
        .send(ClientMsg::Hello {
            protocol_version: PROTOCOL_VERSION,
            name: "lod-heap-report".to_owned(),
        })
        .unwrap();
    server.add_connection(server_end);
    // Load the whole default view (R = 8, V = 4: 1773 cubes), then settle.
    for _ in 0..300 {
        server.tick();
        while client.try_recv().unwrap().is_some() {}
    }
    eprintln!(
        "lod/default_view: lod0 cubes = {}, lod cubes = {}, lod heap = {} bytes ({:.1} KiB)",
        server.world_cube_count(),
        server.lod_cube_count(),
        server.lod_heap_bytes(),
        server.lod_heap_bytes() as f64 / 1024.0
    );
}

fn bench(c: &mut Criterion) {
    report_default_view_heap();

    let generator = generator();
    let pos = surface_pos(&generator);

    let mut group = c.benchmark_group("lod");

    group.bench_function("insert_lod0/surface", |b| {
        // A generated surface cube, mixed by construction (asserted), into
        // an existing pyramid whose level-1 parent already holds the seven
        // sibling octants — through all 4 levels.
        let cube = generator.generate(pos);
        assert!(cube.uniform().is_none(), "the surface case must be mixed");
        b.iter_batched(
            || sibling_pyramid(&generator, pos),
            |mut pyramid| {
                black_box(pyramid.insert_lod0(pos, &cube));
                pyramid
            },
            BatchSize::PerIteration,
        )
    });

    group.bench_function("insert_lod0/air", |b| {
        // A uniform-air cube far above the surface: chain marking only, no
        // content changes. The pyramid starts empty so the chain is built.
        let air_pos = CubePos { x: 0, y: 100, z: 0 };
        let cube = generator.generate(air_pos);
        assert_eq!(cube.uniform(), Some(jbm_core::block::BlockStateId::AIR));
        b.iter_batched(
            || LodPyramid::new(4),
            |mut pyramid| {
                black_box(pyramid.insert_lod0(air_pos, &cube));
                pyramid
            },
            BatchSize::PerIteration,
        )
    });

    group.bench_function("edit", |b| {
        // One edit through all levels, against a pyramid that already
        // holds the cube's downsample. The edit (breaking one block) is
        // picked so that LOD1 really changes; a no-op edit would stop at
        // the first level and measure nothing.
        let cube = generator.generate(pos);
        // The pyramid owns worker plumbing and is not `Clone`; a bare one
        // is rebuilt per batch in the untimed setup instead.
        let base = || {
            let mut pyramid = LodPyramid::new(4);
            pyramid.insert_lod0(pos, &cube);
            pyramid
        };
        let (edited, local) = (0..jbm_core::coords::CUBE_VOLUME)
            .map(LocalPos::from_index)
            .filter(|&local| cube.get(local) != jbm_core::block::BlockStateId::AIR)
            .find_map(|local| {
                let mut edited = cube.clone();
                edited.set(local, jbm_core::block::BlockStateId::AIR);
                let mut pyramid = base();
                (!pyramid.edit_lod0(pos.block(local), &edited).is_empty())
                    .then_some((edited, local))
            })
            .expect("some single break changes LOD1");
        let block = pos.block(local);
        b.iter_batched(
            base,
            |mut pyramid| {
                black_box(pyramid.edit_lod0(block, &edited));
                pyramid
            },
            BatchSize::PerIteration,
        )
    });

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
