//! Shared types for the engine: cubes, palettes, block states, registries,
//! and the protocol messages the client and server exchange.

pub mod block;
pub mod coords;
pub mod cube;

/// The engine's canonical name, reported by the game binary and in logs.
pub const ENGINE_NAME: &str = "just-better-minecraft";

#[cfg(test)]
mod tests {
    use super::ENGINE_NAME;

    #[test]
    fn engine_name_is_set() {
        assert_eq!(ENGINE_NAME, "just-better-minecraft");
        assert!(!ENGINE_NAME.is_empty());
    }
}
