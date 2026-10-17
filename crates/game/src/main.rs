//! The `jbm` binary: starts an in-process server on its own thread,
//! connects the client to it over an in-process link, and runs the client
//! window. The server thread is joined on exit — never detached.

use std::path::PathBuf;
use std::sync::Mutex;

use jbm_client::app::ViewSetup;
use jbm_client::debug_ui::DebugHooks;
use jbm_core::link::in_process_pair;
use jbm_server::run::{self, ServerHandle};
use jbm_server::server::{Server, ServerConfig, ServerStats};
use jbm_server::worldgen::Preset;

/// The far-look view distances (task 2.9) derived from the server config:
/// the LOD0 streaming disk radius and the outermost LOD ring's radius, in
/// blocks. The client cannot learn these from the link, so the game —
/// which built the config — passes them along. Task 2.12 adds the debug
/// hooks (server stats and live radius steering) to the same bundle.
fn view_setup(lod0_ring_radius: f32, furthest_ring_radius: f32, debug: DebugHooks) -> ViewSetup {
    ViewSetup {
        lod0_ring_radius,
        furthest_ring_radius,
        debug: Some(debug),
    }
}

/// The world directory the game saves to, unless `JBM_WORLD_DIR` overrides
/// it. `saves/default` under the current directory. The playground never
/// saves, whatever `JBM_NO_SAVE` says (task 2.12): it is a deterministic
/// debug layout — a save directory would only cache it.
fn world_dir(preset: Preset) -> Option<PathBuf> {
    if preset == Preset::Playground {
        return None;
    }
    if std::env::var("JBM_NO_SAVE").is_ok_and(|v| v == "1") {
        return None;
    }
    Some(match std::env::var("JBM_WORLD_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => PathBuf::from("saves/default"),
    })
}

/// The world preset `JBM_WORLD` selects: `overworld` (the default),
/// `classic` or `playground` (task 2.12's fixed debug layout). An unknown
/// value is a one-line exit, not a panic.
fn preset_from_env() -> Result<Preset, String> {
    let value = std::env::var("JBM_WORLD").unwrap_or_default();
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "overworld" => Ok(Preset::Overworld),
        "classic" => Ok(Preset::Classic),
        "playground" => Ok(Preset::Playground),
        other => Err(format!(
            "jbm: unknown JBM_WORLD '{other}' (expected overworld, classic or playground)"
        )),
    }
}

/// The F3 panel's server bridge (task 2.12): plain closures over the
/// shared stats slot and the control sender, so the client shows server
/// numbers and steers the view radius without depending on `jbm-server`.
fn debug_hooks(handle: &ServerHandle) -> DebugHooks {
    let (stats_slot, control) = handle.debug_bridge();
    DebugHooks {
        server_lines: Box::new(move || server_lines(&stats_slot)),
        set_view_radius: Box::new(move |view_radius, lod_view_radius| {
            let _ = control.send((view_radius, lod_view_radius));
        }),
    }
}

/// Formats the latest [`ServerStats`] snapshot into the panel's label /
/// value rows. A missing snapshot (before the first 0.5 s publication)
/// says so.
fn server_lines(stats_slot: &Mutex<Option<ServerStats>>) -> Vec<(String, String)> {
    // A poisoned lock means the server thread died mid-publication; the
    // panel should show the last values, not crash the render loop.
    let snapshot = stats_slot
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let Some(s) = snapshot else {
        return vec![("server".to_owned(), "starting…".to_owned())];
    };
    let mut lines = vec![
        ("tick".to_owned(), s.tick.to_string()),
        ("world cubes".to_owned(), s.world_cube_count.to_string()),
        ("gen in flight".to_owned(), s.gen_in_flight.to_string()),
        (
            "lod cubes".to_owned(),
            format!("{} ({} jobs)", s.lod.cube_count, s.lod.in_flight),
        ),
        (
            "spans".to_owned(),
            format!(
                "{} + {} done, {} queued",
                s.spans.computed_lod0, s.spans.computed_l1, s.spans.pending
            ),
        ),
        (
            "saves".to_owned(),
            format!(
                "{} ok, {} failed, {} pending",
                s.saves.saved_batches, s.saves.failed_batches, s.saves.pending_save_count
            ),
        ),
    ];
    for player in &s.per_player {
        lines.push((
            format!("pending p{}", player.id.0),
            format!(
                "lod0 {} ring {}",
                player.lod0_pending, player.lod_ring_pending
            ),
        ));
    }
    lines
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let preset = match preset_from_env() {
        Ok(preset) => preset,
        Err(line) => {
            eprintln!("{line}");
            std::process::exit(1);
        }
    };
    let config = ServerConfig {
        world_dir: world_dir(preset),
        worldgen: preset,
        ..ServerConfig::default()
    };
    // The far-look view distances, read off the config before it moves
    // into the server (task 2.9).
    let (lod0_ring_radius, furthest_ring_radius) = config.ring_radii();
    // A storage error (locked world, newer format, …) is a one-line exit,
    // not a panic.
    let server = match Server::try_with_config(config) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("jbm: cannot open the world: {error}");
            std::process::exit(1);
        }
    };
    let handle = run::spawn(server);
    let view = view_setup(lod0_ring_radius, furthest_ring_radius, debug_hooks(&handle));
    let (client_end, server_end) = in_process_pair();
    handle.connect(server_end);
    let result = jbm_client::app::run_with_view(client_end, view);
    // The window closed (or the link died); stop and join the server. That
    // flushes every accepted edit to disk before the process exits.
    handle.stop();
    result
}
