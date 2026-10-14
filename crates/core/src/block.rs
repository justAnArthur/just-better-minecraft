//! Block states and the startup block registry.
//!
//! A block is declared once as a [`BlockDef`]; every combination of property
//! values gets a consecutive [`BlockStateId`]. The registry is built once at
//! startup and then shared read-only — no interior mutability.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// An interned `(block, properties)` pair. Id `0` is always air.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockStateId(pub u32);

impl BlockStateId {
    /// The air state; every registry assigns it id 0.
    pub const AIR: Self = Self(0);
}

/// Handle to a registered block: one `BlockDef` and all of its states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

/// Declaration used to register a block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockDef {
    /// Namespaced id, `"namespace:path"` (lowercase `[a-z0-9_.]` on both
    /// sides of exactly one `:`).
    pub name: String,
    pub properties: Vec<Property>,
    pub solid: bool,
    pub opaque: bool,
}

/// A named property with its possible values. The first value is the default;
/// declaration order defines the state id layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Property {
    pub name: String,
    pub values: Vec<String>,
}

/// Everything known about one block state, looked up by id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateInfo {
    pub block: BlockId,
    /// Full block name, `"namespace:path"`.
    pub name: String,
    /// This state's property values, in declaration order.
    pub properties: Vec<(String, String)>,
    pub solid: bool,
    pub opaque: bool,
}

/// Ways a [`BlockDef`] can fail to register.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateName(String),
    MalformedName(String),
    /// A property was declared with no values; carries the property name.
    EmptyPropertyValues(String),
    DuplicateProperty(String),
    /// Registering the block would push total state count past `u32::MAX`.
    TooManyStates,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::DuplicateName(name) => write!(f, "duplicate block name {name:?}"),
            RegistryError::MalformedName(name) => write!(f, "malformed block name {name:?}"),
            RegistryError::EmptyPropertyValues(prop) => {
                write!(f, "property {prop:?} has no values")
            }
            RegistryError::DuplicateProperty(prop) => write!(f, "duplicate property {prop:?}"),
            RegistryError::TooManyStates => write!(f, "too many states in total"),
        }
    }
}

impl std::error::Error for RegistryError {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BlockInfo {
    def: BlockDef,
    /// State ids of this block are `first_state..first_state + state_count`.
    first_state: u32,
    state_count: u32,
}

/// Interns block states to `u32` ids. Built once at startup, then read-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockRegistry {
    blocks: Vec<BlockInfo>,
    by_name: HashMap<String, BlockId>,
    /// One entry per state id; hot paths index this flat table in O(1).
    states: Vec<StateInfo>,
}

impl Default for BlockRegistry {
    /// An empty registry except `base:air` at state 0 (not solid, not opaque).
    fn default() -> Self {
        Self::new()
    }
}

impl BlockRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            blocks: Vec::new(),
            by_name: HashMap::new(),
            states: Vec::new(),
        };
        registry
            .register(BlockDef {
                name: "base:air".into(),
                properties: Vec::new(),
                solid: false,
                opaque: false,
            })
            .expect("base:air always registers cleanly");
        registry
    }

    /// Registers every combination of property values as a consecutive run of
    /// state ids starting at [`BlockStateId::AIR`]-offset `state_count()`.
    /// The first value of each property is the default (offset 0).
    pub fn register(&mut self, def: BlockDef) -> Result<BlockId, RegistryError> {
        validate_name(&def.name)?;
        if self.by_name.contains_key(&def.name) {
            return Err(RegistryError::DuplicateName(def.name));
        }
        let mut seen = HashSet::new();
        for prop in &def.properties {
            if prop.values.is_empty() {
                return Err(RegistryError::EmptyPropertyValues(prop.name.clone()));
            }
            if !seen.insert(prop.name.as_str()) {
                return Err(RegistryError::DuplicateProperty(prop.name.clone()));
            }
        }
        let count: u128 = def
            .properties
            .iter()
            .map(|p| p.values.len() as u128)
            .product();
        if self.states.len() as u128 + count > u128::from(u32::MAX) {
            return Err(RegistryError::TooManyStates);
        }

        let block = BlockId(self.blocks.len() as u32);
        let first_state = self.states.len() as u32;
        let name = def.name.clone();
        // Odometer over value indices; the last property varies fastest.
        let mut combo = vec![0u32; def.properties.len()];
        for _ in 0..count {
            let properties = def
                .properties
                .iter()
                .zip(&combo)
                .map(|(prop, &value)| (prop.name.clone(), prop.values[value as usize].clone()))
                .collect();
            self.states.push(StateInfo {
                block,
                name: def.name.clone(),
                properties,
                solid: def.solid,
                opaque: def.opaque,
            });
            for i in (0..combo.len()).rev() {
                combo[i] += 1;
                if combo[i] < def.properties[i].values.len() as u32 {
                    break;
                }
                combo[i] = 0;
            }
        }
        self.blocks.push(BlockInfo {
            def,
            first_state,
            state_count: count as u32,
        });
        self.by_name.insert(name, block);
        Ok(block)
    }

    /// Looks up a block by `"namespace:path"` name.
    pub fn block(&self, name: &str) -> Option<BlockId> {
        self.by_name.get(name).copied()
    }

    /// The state a freshly placed block of this kind gets. Panics on an
    /// unregistered block id.
    pub fn default_state(&self, block: BlockId) -> BlockStateId {
        BlockStateId(self.block_info(block).first_state)
    }

    /// The state of `name` with `props` set; unspecified properties take
    /// their defaults. `None` for an unknown name, property or value.
    pub fn state(&self, name: &str, props: &[(&str, &str)]) -> Option<BlockStateId> {
        let info = self.block_info(self.block(name)?);
        let mut combo = vec![0u32; info.def.properties.len()];
        for (prop, value) in props {
            let pi = info.def.properties.iter().position(|p| p.name == *prop)?;
            let vi = info.def.properties[pi]
                .values
                .iter()
                .position(|v| v == *value)?;
            combo[pi] = vi as u32;
        }
        Some(BlockStateId(
            info.first_state + Self::offset(&info.def.properties, &combo),
        ))
    }

    /// The state that differs from `state` only in `prop` being `value`.
    /// `None` if any of the three is unknown.
    pub fn with_property(
        &self,
        state: BlockStateId,
        prop: &str,
        value: &str,
    ) -> Option<BlockStateId> {
        let info = self.states.get(state.0 as usize)?;
        let block = self.block_info(info.block);
        let mut combo = Self::decode(block, state)?;
        let pi = block.def.properties.iter().position(|p| p.name == prop)?;
        let vi = block.def.properties[pi]
            .values
            .iter()
            .position(|v| v == value)?;
        combo[pi] = vi as u32;
        Some(BlockStateId(
            block.first_state + Self::offset(&block.def.properties, &combo),
        ))
    }

    /// O(1) lookup of everything about a state. Panics on an unregistered id.
    pub fn info(&self, state: BlockStateId) -> &StateInfo {
        self.states
            .get(state.0 as usize)
            .expect("unknown BlockStateId")
    }

    /// Total number of registered states; valid ids are `0..state_count()`.
    pub fn state_count(&self) -> u32 {
        self.states.len() as u32
    }

    fn block_info(&self, block: BlockId) -> &BlockInfo {
        self.blocks.get(block.0 as usize).expect("unknown BlockId")
    }

    /// Mixed-radix offset of `combo` within a block's state run, where the
    /// last property varies fastest.
    fn offset(properties: &[Property], combo: &[u32]) -> u32 {
        let mut acc = 0u32;
        for (prop, &value) in properties.iter().zip(combo) {
            acc = acc * prop.values.len() as u32 + value;
        }
        acc
    }

    /// Inverse of `offset` for a valid state id.
    fn decode(block: &BlockInfo, state: BlockStateId) -> Option<Vec<u32>> {
        let mut offset = state.0.checked_sub(block.first_state)?;
        if offset >= block.state_count {
            return None;
        }
        let mut combo = vec![0u32; block.def.properties.len()];
        for i in (0..combo.len()).rev() {
            let len = block.def.properties[i].values.len() as u32;
            combo[i] = offset % len;
            offset /= len;
        }
        Some(combo)
    }
}

fn validate_name(name: &str) -> Result<(), RegistryError> {
    let valid = |part: &str| {
        !part.is_empty()
            && part
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'.')
    };
    let mut parts = name.split(':');
    let well_formed = matches!(
        (parts.next(), parts.next(), parts.next()),
        (Some(ns), Some(path), None) if valid(ns) && valid(path)
    );
    if well_formed {
        Ok(())
    } else {
        Err(RegistryError::MalformedName(name.to_owned()))
    }
}

/// Hardcoded Stage 1 blocks in the `base:` namespace. Stage 3 moves them into
/// a mod.
pub mod builtin {
    use super::{BlockDef, BlockRegistry, BlockStateId, Property};

    /// Default state ids of the builtin blocks, for convenience.
    #[derive(Debug, Clone, Copy)]
    pub struct Blocks {
        pub stone: BlockStateId,
        pub dirt: BlockStateId,
        pub grass_block: BlockStateId,
        pub sand: BlockStateId,
        pub water: BlockStateId,
        pub glass: BlockStateId,
        pub oak_planks: BlockStateId,
        pub oak_log: BlockStateId,
        pub leaves: BlockStateId,
    }

    impl Blocks {
        /// Registers all builtin blocks and returns their default state ids.
        /// Panics if any name is already taken (e.g. builtins registered
        /// twice).
        pub fn register(registry: &mut BlockRegistry) -> Self {
            fn simple(name: &str, solid: bool, opaque: bool) -> BlockDef {
                BlockDef {
                    name: format!("base:{name}"),
                    properties: Vec::new(),
                    solid,
                    opaque,
                }
            }
            let stone = registry.register(simple("stone", true, true)).unwrap();
            let dirt = registry.register(simple("dirt", true, true)).unwrap();
            let grass_block = registry
                .register(simple("grass_block", true, true))
                .unwrap();
            let sand = registry.register(simple("sand", true, true)).unwrap();
            let water = registry.register(simple("water", false, false)).unwrap();
            let glass = registry.register(simple("glass", true, false)).unwrap();
            let oak_planks = registry.register(simple("oak_planks", true, true)).unwrap();
            let oak_log = registry
                .register(BlockDef {
                    name: "base:oak_log".into(),
                    properties: vec![Property {
                        name: "axis".into(),
                        values: ["y", "x", "z"].map(String::from).into(),
                    }],
                    solid: true,
                    opaque: true,
                })
                .unwrap();
            let leaves = registry.register(simple("leaves", true, false)).unwrap();
            Blocks {
                stone: registry.default_state(stone),
                dirt: registry.default_state(dirt),
                grass_block: registry.default_state(grass_block),
                sand: registry.default_state(sand),
                water: registry.default_state(water),
                glass: registry.default_state(glass),
                oak_planks: registry.default_state(oak_planks),
                oak_log: registry.default_state(oak_log),
                leaves: registry.default_state(leaves),
            }
        }
    }

    /// Registers the builtin blocks, ignoring their ids.
    pub fn register(registry: &mut BlockRegistry) {
        Blocks::register(registry);
    }
}

#[cfg(test)]
mod tests {
    use super::builtin;
    use super::*;

    fn demo_registry() -> (BlockRegistry, BlockId) {
        let mut registry = BlockRegistry::new();
        let id = registry
            .register(BlockDef {
                name: "test:thing".into(),
                properties: vec![
                    Property {
                        name: "a".into(),
                        values: ["x", "y"].map(String::from).into(),
                    },
                    Property {
                        name: "b".into(),
                        values: ["1", "2", "3"].map(String::from).into(),
                    },
                ],
                solid: true,
                opaque: false,
            })
            .unwrap();
        (registry, id)
    }

    #[test]
    fn air_is_state_zero() {
        let registry = BlockRegistry::new();
        assert_eq!(BlockStateId::AIR, BlockStateId(0));
        let air = registry.info(BlockStateId::AIR);
        assert_eq!(air.name, "base:air");
        assert_eq!(air.block, BlockId(0));
        assert!(!air.solid && !air.opaque);
        assert!(air.properties.is_empty());
        assert_eq!(registry.state_count(), 1);
        assert_eq!(registry.default_state(BlockId(0)), BlockStateId::AIR);
    }

    #[test]
    fn state_ids_are_consecutive_per_block() {
        let (registry, id) = demo_registry();
        let first = registry.default_state(id);
        // air (1 state) then 2 x 3 = 6 states of `test:thing`.
        assert_eq!(first, BlockStateId(1));
        assert_eq!(registry.state_count(), 7);
        for k in 0..6 {
            assert_eq!(registry.info(BlockStateId(1 + k)).block, id);
        }
        // Default state: first value of each property, offset 0.
        let info = registry.info(first);
        assert_eq!(
            info.properties,
            vec![
                ("a".to_string(), "x".to_string()),
                ("b".to_string(), "1".to_string())
            ]
        );
        assert!(info.solid && !info.opaque);
    }

    #[test]
    fn state_lookup_partial_and_full() {
        let (registry, id) = demo_registry();
        let first = registry.default_state(id);
        // Last property varies fastest: offset = a_idx * 3 + b_idx.
        assert_eq!(registry.state("test:thing", &[]), Some(first));
        assert_eq!(
            registry.state("test:thing", &[("b", "3")]),
            Some(BlockStateId(3))
        );
        assert_eq!(
            registry.state("test:thing", &[("a", "y"), ("b", "2")]),
            Some(BlockStateId(5))
        );
        assert_eq!(registry.state("test:missing", &[]), None);
        assert_eq!(registry.state("test:thing", &[("nope", "1")]), None);
        assert_eq!(registry.state("test:thing", &[("a", "nope")]), None);
    }

    #[test]
    fn with_property_moves_within_the_block() {
        let (registry, id) = demo_registry();
        let first = registry.default_state(id);
        assert_eq!(
            registry.with_property(first, "a", "y"),
            Some(BlockStateId(4))
        );
        assert_eq!(
            registry.with_property(BlockStateId(5), "b", "1"),
            Some(BlockStateId(4))
        );
        // Idempotence: same value stays put.
        assert_eq!(registry.with_property(first, "a", "x"), Some(first));
        assert_eq!(registry.with_property(first, "nope", "x"), None);
        assert_eq!(registry.with_property(first, "a", "nope"), None);
        assert_eq!(registry.with_property(BlockStateId(99), "a", "x"), None);
    }

    #[test]
    fn every_error_case() {
        let mut registry = BlockRegistry::new();
        registry
            .register(BlockDef {
                name: "test:dup".into(),
                properties: Vec::new(),
                solid: false,
                opaque: false,
            })
            .unwrap();

        let simple = |name: &str| BlockDef {
            name: name.into(),
            properties: Vec::new(),
            solid: false,
            opaque: false,
        };
        assert_eq!(
            registry.register(simple("test:dup")),
            Err(RegistryError::DuplicateName("test:dup".into()))
        );
        for bad in [
            "dup", "a:b:c", ":path", "ns:", "Ns:path", "ns:Path", "ns:pa-th", "",
        ] {
            assert_eq!(
                registry.register(simple(bad)),
                Err(RegistryError::MalformedName(bad.to_string())),
                "name {bad:?} should be rejected"
            );
        }
        assert_eq!(
            registry.register(BlockDef {
                name: "test:empty".into(),
                properties: vec![Property {
                    name: "axis".into(),
                    values: Vec::new(),
                }],
                solid: false,
                opaque: false,
            }),
            Err(RegistryError::EmptyPropertyValues("axis".into()))
        );
        assert_eq!(
            registry.register(BlockDef {
                name: "test:twice".into(),
                properties: vec![
                    Property {
                        name: "axis".into(),
                        values: ["x"].map(String::from).into(),
                    },
                    Property {
                        name: "axis".into(),
                        values: ["x"].map(String::from).into(),
                    },
                ],
                solid: false,
                opaque: false,
            }),
            Err(RegistryError::DuplicateProperty("axis".into()))
        );

        // 65536 x 65536 = 2^32 states; even without any existing block that
        // exceeds u32::MAX.
        let big = |n: u32| (0..n).map(|i| format!("v{i}")).collect::<Vec<_>>();
        assert_eq!(
            registry.register(BlockDef {
                name: "test:huge".into(),
                properties: vec![
                    Property {
                        name: "p0".into(),
                        values: big(65536),
                    },
                    Property {
                        name: "p1".into(),
                        values: big(65536),
                    },
                ],
                solid: false,
                opaque: false,
            }),
            Err(RegistryError::TooManyStates)
        );
    }

    #[test]
    fn builtins_register_cleanly() {
        let mut registry = BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);

        assert_eq!(
            registry.state("base:oak_log", &[("axis", "y")]),
            Some(blocks.oak_log)
        );
        assert!(registry.info(blocks.stone).solid && registry.info(blocks.stone).opaque);
        assert!(registry.info(blocks.dirt).solid && registry.info(blocks.dirt).opaque);
        assert!(registry.info(blocks.grass_block).solid);
        assert!(registry.info(blocks.sand).opaque);
        let water = registry.info(blocks.water);
        assert!(!water.solid && !water.opaque);
        let glass = registry.info(blocks.glass);
        assert!(glass.solid && !glass.opaque);
        let leaves = registry.info(blocks.leaves);
        assert!(leaves.solid && !leaves.opaque);
        // air + 8 single-state blocks + oak_log with 3 axis states
        assert_eq!(registry.state_count(), 1 + 8 + 3);
    }

    #[test]
    #[should_panic(expected = "DuplicateName")]
    fn registering_builtins_twice_panics() {
        let mut registry = BlockRegistry::new();
        builtin::register(&mut registry);
        builtin::register(&mut registry);
    }

    #[test]
    fn builtin_blocks_ids_match_the_registry() {
        let mut registry = BlockRegistry::new();
        let blocks = builtin::Blocks::register(&mut registry);
        assert_eq!(
            registry
                .block("base:stone")
                .map(|b| registry.default_state(b)),
            Some(blocks.stone)
        );
        assert_eq!(
            registry.state("base:oak_log", &[("axis", "z")]),
            registry.with_property(blocks.oak_log, "axis", "z")
        );
        assert_eq!(registry.info(blocks.oak_log).name, "base:oak_log");
    }
}
