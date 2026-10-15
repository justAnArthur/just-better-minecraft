//! The density-graph format, its compiler and its evaluator. Format docs
//! live in the crate README ("Worldgen graph"); the determinism rule is
//! restated in the [`super`] module docs.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::reference;

// ---------------------------------------------------------------------------
// Graph format
// ---------------------------------------------------------------------------

/// A parsed worldgen graph. Build one with [`Graph::from_json`] (or
/// literally, in tests) and hand it to [`compile`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Graph {
    /// Format version; only `1` is supported.
    pub version: u32,
    /// Named, shareable subexpressions. A named node is compiled once per
    /// sample however often it is referenced.
    pub nodes: BTreeMap<String, NodeValue>,
    /// Graph outputs. `final_density` decides solidity (a cell is solid
    /// when it is `>= 0`); further outputs may appear in later versions and
    /// are ignored for now.
    pub outputs: Outputs,
}

/// The `outputs` map of a [`Graph`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Outputs {
    pub final_density: NodeValue,
}

/// An operand inside the graph: a constant, a reference to a named node, or
/// an inline node.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum NodeValue {
    Const(f64),
    Ref {
        #[serde(rename = "ref")]
        name: String,
    },
    Inline(Box<Node>),
}

/// One node of the density graph. All values are `f64`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Node {
    X,
    Y,
    Z,
    Add {
        a: NodeValue,
        b: NodeValue,
    },
    Sub {
        a: NodeValue,
        b: NodeValue,
    },
    Mul {
        a: NodeValue,
        b: NodeValue,
    },
    Neg {
        a: NodeValue,
    },
    Min {
        a: NodeValue,
        b: NodeValue,
    },
    Max {
        a: NodeValue,
        b: NodeValue,
    },
    Abs {
        a: NodeValue,
    },
    Floor {
        a: NodeValue,
    },
    Clamp {
        input: NodeValue,
        min: NodeValue,
        max: NodeValue,
    },
    /// The reference 2D gradient noise ([`reference::noise2`]) sampled at
    /// `(x · x_scale, z · z_scale)` with seed `world_seed ^ salt`.
    Noise2 {
        salt: Salt,
        x_scale: f64,
        z_scale: f64,
    },
    /// The reference fractal sum ([`reference::fbm`]) sampled at
    /// `(x · scale, z · scale)` with seed `world_seed ^ salt`.
    Fbm2 {
        salt: Salt,
        scale: f64,
        octaves: u32,
    },
    /// Piecewise-linear spline over `input`, clamped at both ends. Points
    /// must be strictly increasing in `x`.
    Spline {
        input: NodeValue,
        points: Vec<[f64; 2]>,
    },
}

/// A noise salt. Serialized as a hex string (`"0x0D1B54A32D168ED5"`) so
/// integers above 2^53 — every interesting salt — round-trip through JSON
/// exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Salt(pub u64);

impl Salt {
    fn parse(s: &str) -> Option<u64> {
        let hex = s
            .strip_prefix("0x")
            .or_else(|| s.strip_prefix("0X"))?
            .trim_start_matches('_');
        if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        u64::from_str_radix(hex, 16).ok()
    }
}

impl fmt::Display for Salt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:016X}", self.0)
    }
}

impl Serialize for Salt {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Everything that can go wrong loading or compiling a graph. Variants that
/// point at a node name the offending node path, e.g. `nodes.hills.a` or
/// `outputs.final_density`. Loading never panics on bad input.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// The JSON did not parse, or the top-level shape was wrong.
    Json(String),
    UnknownVersion {
        found: u64,
    },
    /// `outputs` has no `final_density` entry.
    MissingFinalDensity,
    UnknownNodeType {
        kind: String,
        path: String,
    },
    MissingField {
        field: &'static str,
        path: String,
    },
    /// A structurally bad node: bad salt string, malformed spline point,
    /// zero octaves, a stray field, a non-number where one was required.
    BadNode {
        message: String,
        path: String,
    },
    DanglingRef {
        name: String,
        path: String,
    },
    Cycle {
        path: String,
    },
    /// A spline with fewer than two points, or points not strictly
    /// increasing in `x`.
    BadSpline {
        path: String,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::Json(message) => write!(f, "invalid graph JSON: {message}"),
            GraphError::UnknownVersion { found } => {
                write!(f, "unsupported graph version {found}, expected 1")
            }
            GraphError::MissingFinalDensity => {
                write!(f, "outputs.final_density: missing `final_density` output")
            }
            GraphError::UnknownNodeType { kind, path } => {
                write!(f, "{path}: unknown node type `{kind}`")
            }
            GraphError::MissingField { field, path } => {
                write!(f, "{path}: missing field `{field}`")
            }
            GraphError::BadNode { message, path } => write!(f, "{path}: {message}"),
            GraphError::DanglingRef { name, path } => {
                write!(f, "{path}: reference to unknown node `{name}`")
            }
            GraphError::Cycle { path } => write!(f, "{path}: named-node cycle"),
            GraphError::BadSpline { path } => {
                write!(
                    f,
                    "{path}: spline points must be at least two and strictly increasing in x"
                )
            }
        }
    }
}

impl std::error::Error for GraphError {}

// ---------------------------------------------------------------------------
// JSON loading with node-path errors
// ---------------------------------------------------------------------------

/// The loose scaffold `serde` fills in; node bodies stay `Value` so the
/// walk below can attach node paths to every error.
#[derive(Deserialize)]
struct RawGraph {
    version: u64,
    #[serde(default)]
    nodes: HashMap<String, Value>,
    #[serde(default)]
    outputs: HashMap<String, Value>,
}

impl Graph {
    /// Parses and validates a graph from its JSON text. Structural errors
    /// name the offending node path; this never panics on bad input.
    pub fn from_json(json: &str) -> Result<Self, GraphError> {
        let raw: RawGraph =
            serde_json::from_str(json).map_err(|e| GraphError::Json(e.to_string()))?;
        if raw.version != 1 {
            return Err(GraphError::UnknownVersion { found: raw.version });
        }
        let Some(final_value) = raw.outputs.get("final_density") else {
            return Err(GraphError::MissingFinalDensity);
        };
        let final_density = parse_node_value(final_value, "outputs.final_density")?;
        let mut nodes = BTreeMap::new();
        for (name, value) in &raw.nodes {
            nodes.insert(
                name.clone(),
                parse_node_value(value, &format!("nodes.{name}"))?,
            );
        }
        Ok(Self {
            version: 1,
            nodes,
            outputs: Outputs { final_density },
        })
    }
}

fn parse_node_value(value: &Value, path: &str) -> Result<NodeValue, GraphError> {
    match value {
        Value::Number(n) => match n.as_f64() {
            Some(v) if v.is_finite() => Ok(NodeValue::Const(v)),
            _ => Err(GraphError::BadNode {
                message: "number is not a finite f64".into(),
                path: path.into(),
            }),
        },
        Value::Object(map) => {
            if map.contains_key("ref") {
                if map.len() != 1 {
                    return Err(GraphError::BadNode {
                        message: "a `ref` object must have exactly one key".into(),
                        path: path.into(),
                    });
                }
                let name = map["ref"].as_str().ok_or_else(|| GraphError::BadNode {
                    message: "`ref` must be a string".into(),
                    path: path.into(),
                })?;
                Ok(NodeValue::Ref {
                    name: name.to_owned(),
                })
            } else if map.contains_key("type") {
                Ok(NodeValue::Inline(Box::new(parse_node(map, path)?)))
            } else {
                Err(GraphError::BadNode {
                    message: "expected a number, a `{\"ref\": …}` or a `{\"type\": …}` object"
                        .into(),
                    path: path.into(),
                })
            }
        }
        _ => Err(GraphError::BadNode {
            message: "expected a number, a `{\"ref\": …}` or a `{\"type\": …}` object".into(),
            path: path.into(),
        }),
    }
}

fn parse_node(map: &serde_json::Map<String, Value>, path: &str) -> Result<Node, GraphError> {
    let kind = map["type"].as_str().ok_or_else(|| GraphError::BadNode {
        message: "`type` must be a string".into(),
        path: path.into(),
    })?;
    let operand = |key: &'static str| -> Result<NodeValue, GraphError> {
        let value = map.get(key).ok_or_else(|| GraphError::MissingField {
            field: key,
            path: path.into(),
        })?;
        parse_node_value(value, &format!("{path}.{key}"))
    };
    let number = |key: &'static str| -> Result<f64, GraphError> {
        let value = map.get(key).ok_or_else(|| GraphError::MissingField {
            field: key,
            path: path.into(),
        })?;
        value
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| GraphError::BadNode {
                message: format!("`{key}` must be a finite number"),
                path: path.into(),
            })
    };
    let salt = || -> Result<u64, GraphError> {
        let value = map.get("salt").ok_or_else(|| GraphError::MissingField {
            field: "salt",
            path: path.into(),
        })?;
        let text = value.as_str().ok_or_else(|| GraphError::BadNode {
            message: "`salt` must be a hex string like \"0x0D1B54A32D168ED5\" (JSON numbers do not round-trip above 2^53)".into(),
            path: path.into(),
        })?;
        Salt::parse(text).ok_or_else(|| GraphError::BadNode {
            message: format!("`salt` is not a hex integer: `{text}`"),
            path: path.into(),
        })
    };
    let node = match kind {
        "x" => Node::X,
        "y" => Node::Y,
        "z" => Node::Z,
        "add" => Node::Add {
            a: operand("a")?,
            b: operand("b")?,
        },
        "sub" => Node::Sub {
            a: operand("a")?,
            b: operand("b")?,
        },
        "mul" => Node::Mul {
            a: operand("a")?,
            b: operand("b")?,
        },
        "neg" => Node::Neg { a: operand("a")? },
        "min" => Node::Min {
            a: operand("a")?,
            b: operand("b")?,
        },
        "max" => Node::Max {
            a: operand("a")?,
            b: operand("b")?,
        },
        "abs" => Node::Abs { a: operand("a")? },
        "floor" => Node::Floor { a: operand("a")? },
        "clamp" => Node::Clamp {
            input: operand("input")?,
            min: operand("min")?,
            max: operand("max")?,
        },
        "noise2" => Node::Noise2 {
            salt: Salt(salt()?),
            x_scale: number("x_scale")?,
            z_scale: number("z_scale")?,
        },
        "fbm2" => {
            let octaves = map.get("octaves").ok_or_else(|| GraphError::MissingField {
                field: "octaves",
                path: path.into(),
            })?;
            let octaves = octaves
                .as_u64()
                .filter(|&o| (1..=u64::from(u32::MAX)).contains(&o))
                .ok_or_else(|| GraphError::BadNode {
                    message: "`octaves` must be an integer between 1 and 2^32-1".into(),
                    path: path.into(),
                })?;
            Node::Fbm2 {
                salt: Salt(salt()?),
                scale: number("scale")?,
                octaves: octaves as u32,
            }
        }
        "spline" => {
            let points_value = map.get("points").ok_or_else(|| GraphError::MissingField {
                field: "points",
                path: path.into(),
            })?;
            let points_json = points_value.as_array().ok_or_else(|| GraphError::BadNode {
                message: "`points` must be an array of `[x, y]` pairs".into(),
                path: path.into(),
            })?;
            let mut points = Vec::with_capacity(points_json.len());
            for (i, pair) in points_json.iter().enumerate() {
                let pair = pair.as_array().ok_or_else(|| GraphError::BadNode {
                    message: format!("points[{i}] must be an `[x, y]` pair"),
                    path: path.into(),
                })?;
                if pair.len() != 2 {
                    return Err(GraphError::BadNode {
                        message: format!("points[{i}] must have exactly two entries"),
                        path: path.into(),
                    });
                }
                let coord = |v: &Value| {
                    v.as_f64()
                        .filter(|c| c.is_finite())
                        .ok_or_else(|| GraphError::BadNode {
                            message: format!("points[{i}] must hold finite numbers"),
                            path: path.into(),
                        })
                };
                points.push([coord(&pair[0])?, coord(&pair[1])?]);
            }
            Node::Spline {
                input: operand("input")?,
                points,
            }
        }
        other => {
            return Err(GraphError::UnknownNodeType {
                kind: other.to_owned(),
                path: path.into(),
            });
        }
    };
    let allowed: &[&str] = match &node {
        Node::X | Node::Y | Node::Z => &[],
        Node::Add { .. }
        | Node::Sub { .. }
        | Node::Mul { .. }
        | Node::Min { .. }
        | Node::Max { .. } => &["a", "b"],
        Node::Neg { .. } | Node::Abs { .. } | Node::Floor { .. } => &["a"],
        Node::Clamp { .. } => &["input", "min", "max"],
        Node::Noise2 { .. } => &["salt", "x_scale", "z_scale"],
        Node::Fbm2 { .. } => &["salt", "scale", "octaves"],
        Node::Spline { .. } => &["input", "points"],
    };
    for key in map.keys() {
        if key != "type" && !allowed.contains(&key.as_str()) {
            return Err(GraphError::BadNode {
                message: format!("unknown field `{key}` on `{kind}` node"),
                path: path.into(),
            });
        }
    }
    Ok(node)
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

/// Which coordinate inputs an instruction depends on. An instruction only
/// ever reads operands of a lower-or-equal dimension, so the stages can run
/// in order constant → column → cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Dim {
    Const,
    Xz,
    Xyz,
}

/// One compiled instruction. Operands are register indices into the
/// evaluator's register array; constants are not instructions, they are
/// preloaded register values.
#[derive(Clone, Debug)]
enum Op {
    X,
    Y,
    Z,
    Add(u32, u32),
    Sub(u32, u32),
    Mul(u32, u32),
    Neg(u32),
    Min(u32, u32),
    Max(u32, u32),
    Abs(u32),
    Floor(u32),
    Clamp {
        input: u32,
        min: u32,
        max: u32,
    },
    Noise2 {
        salt: u64,
        x_scale: f64,
        z_scale: f64,
    },
    Fbm2 {
        salt: u64,
        scale: f64,
        octaves: u32,
    },
    Spline {
        input: u32,
        points: Arc<[(f64, f64)]>,
    },
}

impl Op {
    /// Feeds the op's discriminant, operands and float payloads (as bit
    /// patterns) to `hasher` — the fingerprint's per-instruction half.
    /// Every variant carries its own tag, so two different op kinds never
    /// hash alike.
    fn feed(&self, mut hasher: impl std::hash::Hasher) {
        use std::hash::Hash;
        macro_rules! tag {
            ($n:literal) => {
                $n.hash(&mut hasher)
            };
        }
        match self {
            Op::X => tag!(0),
            Op::Y => tag!(1),
            Op::Z => tag!(2),
            Op::Add(a, b) => {
                tag!(3);
                a.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            Op::Sub(a, b) => {
                tag!(4);
                a.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            Op::Mul(a, b) => {
                tag!(5);
                a.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            Op::Neg(a) => {
                tag!(6);
                a.hash(&mut hasher);
            }
            Op::Min(a, b) => {
                tag!(7);
                a.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            Op::Max(a, b) => {
                tag!(8);
                a.hash(&mut hasher);
                b.hash(&mut hasher);
            }
            Op::Abs(a) => {
                tag!(9);
                a.hash(&mut hasher);
            }
            Op::Floor(a) => {
                tag!(10);
                a.hash(&mut hasher);
            }
            Op::Clamp { input, min, max } => {
                tag!(11);
                input.hash(&mut hasher);
                min.hash(&mut hasher);
                max.hash(&mut hasher);
            }
            Op::Noise2 {
                salt,
                x_scale,
                z_scale,
            } => {
                tag!(12);
                salt.hash(&mut hasher);
                x_scale.to_bits().hash(&mut hasher);
                z_scale.to_bits().hash(&mut hasher);
            }
            Op::Fbm2 {
                salt,
                scale,
                octaves,
            } => {
                tag!(13);
                salt.hash(&mut hasher);
                scale.to_bits().hash(&mut hasher);
                octaves.hash(&mut hasher);
            }
            Op::Spline { input, points } => {
                tag!(14);
                input.hash(&mut hasher);
                points.len().hash(&mut hasher);
                for &(x, y) in points.iter() {
                    x.to_bits().hash(&mut hasher);
                    y.to_bits().hash(&mut hasher);
                }
            }
        }
    }

    /// Register operands, in evaluation-irrelevant order. Compile-time only.
    fn operands(&self) -> Vec<u32> {
        match self {
            Op::X | Op::Y | Op::Z | Op::Noise2 { .. } | Op::Fbm2 { .. } => vec![],
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Min(a, b) | Op::Max(a, b) => {
                vec![*a, *b]
            }
            Op::Neg(a) | Op::Abs(a) | Op::Floor(a) => vec![*a],
            Op::Clamp { input, min, max } => vec![*input, *min, *max],
            Op::Spline { input, .. } => vec![*input],
        }
    }

    /// Rewrites every operand through an old→new register index map.
    /// Compile-time only.
    fn remap(&self, map: &[u32]) -> Self {
        let r = |i: &u32| map[*i as usize];
        match self {
            Op::X | Op::Y | Op::Z | Op::Noise2 { .. } | Op::Fbm2 { .. } => self.clone(),
            Op::Add(a, b) => Op::Add(r(a), r(b)),
            Op::Sub(a, b) => Op::Sub(r(a), r(b)),
            Op::Mul(a, b) => Op::Mul(r(a), r(b)),
            Op::Min(a, b) => Op::Min(r(a), r(b)),
            Op::Max(a, b) => Op::Max(r(a), r(b)),
            Op::Neg(a) => Op::Neg(r(a)),
            Op::Abs(a) => Op::Abs(r(a)),
            Op::Floor(a) => Op::Floor(r(a)),
            Op::Clamp { input, min, max } => Op::Clamp {
                input: r(input),
                min: r(min),
                max: r(max),
            },
            Op::Spline { input, points } => Op::Spline {
                input: r(input),
                points: Arc::clone(points),
            },
        }
    }
}

enum Emitted {
    Const(f64),
    Op(Op),
}

struct Compiler<'g> {
    graph: &'g Graph,
    /// Named node → emission id, so every shared node is emitted once.
    resolved: HashMap<&'g str, u32>,
    /// Named nodes currently being expanded, for cycle detection.
    visiting: Vec<&'g str>,
    emitted: Vec<Emitted>,
    dims: Vec<Dim>,
}

impl<'g> Compiler<'g> {
    fn new(graph: &'g Graph) -> Self {
        Self {
            graph,
            resolved: HashMap::new(),
            visiting: Vec::new(),
            emitted: Vec::new(),
            dims: Vec::new(),
        }
    }

    fn push_const(&mut self, value: f64) -> u32 {
        let id = self.emitted.len() as u32;
        self.emitted.push(Emitted::Const(value));
        self.dims.push(Dim::Const);
        id
    }

    fn push_op(&mut self, op: Op, dim: Dim) -> u32 {
        let id = self.emitted.len() as u32;
        self.emitted.push(Emitted::Op(op));
        self.dims.push(dim);
        id
    }

    fn const_of(&self, id: u32) -> Option<f64> {
        match self.emitted[id as usize] {
            Emitted::Const(v) => Some(v),
            Emitted::Op(_) => None,
        }
    }

    fn dim_of(&self, id: u32) -> Dim {
        self.dims[id as usize]
    }

    fn emit_value(&mut self, value: &'g NodeValue, path: &str) -> Result<u32, GraphError> {
        match value {
            NodeValue::Const(c) => Ok(self.push_const(*c)),
            NodeValue::Ref { name } => {
                if let Some(&id) = self.resolved.get(name.as_str()) {
                    return Ok(id);
                }
                if self.visiting.contains(&name.as_str()) {
                    return Err(GraphError::Cycle { path: path.into() });
                }
                let node = match self.graph.nodes.get(name) {
                    Some(NodeValue::Inline(node)) => node.as_ref(),
                    Some(_) => {
                        return Err(GraphError::BadNode {
                            message: format!(
                                "named node `{name}` must be an inline node object, not a constant or ref"
                            ),
                            path: format!("nodes.{name}"),
                        });
                    }
                    None => {
                        return Err(GraphError::DanglingRef {
                            name: name.clone(),
                            path: path.into(),
                        });
                    }
                };
                self.visiting.push(name.as_str());
                let id = self.emit_node(node, &format!("nodes.{name}"))?;
                self.visiting.pop();
                self.resolved.insert(name.as_str(), id);
                Ok(id)
            }
            NodeValue::Inline(node) => self.emit_node(node, path),
        }
    }

    fn emit_node(&mut self, node: &'g Node, path: &str) -> Result<u32, GraphError> {
        match node {
            Node::X => Ok(self.push_op(Op::X, Dim::Xz)),
            Node::Y => Ok(self.push_op(Op::Y, Dim::Xyz)),
            Node::Z => Ok(self.push_op(Op::Z, Dim::Xz)),
            Node::Add { a, b } => self.emit_binary(a, b, path, |x, y| x + y, Op::Add),
            Node::Sub { a, b } => self.emit_binary(a, b, path, |x, y| x - y, Op::Sub),
            Node::Mul { a, b } => self.emit_binary(a, b, path, |x, y| x * y, Op::Mul),
            Node::Min { a, b } => self.emit_binary(a, b, path, f64::min, Op::Min),
            Node::Max { a, b } => self.emit_binary(a, b, path, f64::max, Op::Max),
            Node::Neg { a } => self.emit_unary(a, path, |x| -x, Op::Neg),
            Node::Abs { a } => self.emit_unary(a, path, f64::abs, Op::Abs),
            Node::Floor { a } => self.emit_unary(a, path, f64::floor, Op::Floor),
            Node::Clamp { input, min, max } => {
                let ii = self.emit_value(input, &format!("{path}.input"))?;
                let im = self.emit_value(min, &format!("{path}.min"))?;
                let ix = self.emit_value(max, &format!("{path}.max"))?;
                if let (Some(v), Some(lo), Some(hi)) =
                    (self.const_of(ii), self.const_of(im), self.const_of(ix))
                {
                    if lo > hi {
                        return Err(GraphError::BadNode {
                            message: "`min` must not exceed `max`".into(),
                            path: path.into(),
                        });
                    }
                    return Ok(self.push_const(v.clamp(lo, hi)));
                }
                let dim = self.dim_of(ii).max(self.dim_of(im)).max(self.dim_of(ix));
                Ok(self.push_op(
                    Op::Clamp {
                        input: ii,
                        min: im,
                        max: ix,
                    },
                    dim,
                ))
            }
            Node::Noise2 {
                salt,
                x_scale,
                z_scale,
            } => Ok(self.push_op(
                Op::Noise2 {
                    salt: salt.0,
                    x_scale: *x_scale,
                    z_scale: *z_scale,
                },
                Dim::Xz,
            )),
            Node::Fbm2 {
                salt,
                scale,
                octaves,
            } => {
                if *octaves == 0 {
                    return Err(GraphError::BadNode {
                        message: "`octaves` must be at least 1".into(),
                        path: path.into(),
                    });
                }
                Ok(self.push_op(
                    Op::Fbm2 {
                        salt: salt.0,
                        scale: *scale,
                        octaves: *octaves,
                    },
                    Dim::Xz,
                ))
            }
            Node::Spline { input, points } => {
                if points.len() < 2 {
                    return Err(GraphError::BadSpline { path: path.into() });
                }
                for w in points.windows(2) {
                    if w[0][0] >= w[1][0] {
                        return Err(GraphError::BadSpline { path: path.into() });
                    }
                }
                let ii = self.emit_value(input, &format!("{path}.input"))?;
                let points: Vec<(f64, f64)> = points.iter().map(|&[x, y]| (x, y)).collect();
                if let Some(v) = self.const_of(ii) {
                    return Ok(self.push_const(spline_eval(&points, v)));
                }
                Ok(self.push_op(
                    Op::Spline {
                        input: ii,
                        points: points.into(),
                    },
                    self.dim_of(ii),
                ))
            }
        }
    }

    fn emit_binary(
        &mut self,
        a: &'g NodeValue,
        b: &'g NodeValue,
        path: &str,
        scalar: fn(f64, f64) -> f64,
        mk: fn(u32, u32) -> Op,
    ) -> Result<u32, GraphError> {
        let ia = self.emit_value(a, &format!("{path}.a"))?;
        let ib = self.emit_value(b, &format!("{path}.b"))?;
        if let (Some(x), Some(y)) = (self.const_of(ia), self.const_of(ib)) {
            return Ok(self.push_const(scalar(x, y)));
        }
        Ok(self.push_op(mk(ia, ib), self.dim_of(ia).max(self.dim_of(ib))))
    }

    fn emit_unary(
        &mut self,
        a: &'g NodeValue,
        path: &str,
        scalar: fn(f64) -> f64,
        mk: fn(u32) -> Op,
    ) -> Result<u32, GraphError> {
        let ia = self.emit_value(a, &format!("{path}.a"))?;
        if let Some(x) = self.const_of(ia) {
            return Ok(self.push_const(scalar(x)));
        }
        Ok(self.push_op(mk(ia), self.dim_of(ia)))
    }
}

/// A graph compiled into a topologically ordered instruction list over a
/// flat register array: evaluation is a loop over instructions, with no
/// recursion and no allocation per sample. Registers are laid out in three
/// stages — folded constants, then column-stage (xz) instructions that run
/// once per `(x, z)` column, then cell-stage (xyz) instructions that run
/// per block. The cell stage never writes a column register, so one column
/// state can serve every `y` of the column.
#[derive(Debug)]
pub struct CompiledGraph {
    const_values: Vec<f64>,
    xz_ops: Vec<Op>,
    xyz_ops: Vec<Op>,
    /// Register holding `final_density`.
    output: u32,
    /// Column-stage registers read by cell-stage instructions, ascending.
    /// These are the only per-column values worth caching between cubes.
    live_xz: Vec<u32>,
    /// The solidity band (see [`Band`]); `None` when the interval bounds
    /// cannot bound it.
    band: Option<Band>,
    /// Test-only count of `eval_column` calls; see the once-per-column and
    /// column-cache tests.
    #[cfg(test)]
    pub(crate) column_evals: AtomicU64,
}

/// The band of rows the interval bounds cannot decide: every row `y <=
/// low()` is provably solid, every row `y >= high()` provably air (over
/// `|x|, |z| <= 2^40`), and the `rows()` rows in between need per-column
/// evaluation. Caching one bit per uncertain row per column — 56 bits for
/// the shipped graph, exactly today's heightmap generalized — makes
/// vertically adjacent cubes nearly free.
#[derive(Debug, Clone, Copy)]
pub struct Band {
    low: i64,
    high: i64,
    rows: usize,
}

/// Widest band worth caching bits for; wider graphs fall back to direct
/// evaluation.
const MAX_BAND_ROWS: usize = 512;

impl Band {
    /// Greatest row provably solid at every `(x, z)` within `|coord| <= 2^40`.
    pub fn low(&self) -> i64 {
        self.low
    }

    /// Least row provably air at every `(x, z)` within `|coord| <= 2^40`.
    pub fn high(&self) -> i64 {
        self.high
    }

    /// How many rows in `(low, high)` need per-column evaluation.
    pub fn rows(&self) -> usize {
        self.rows
    }
}

impl Clone for CompiledGraph {
    fn clone(&self) -> Self {
        Self {
            const_values: self.const_values.clone(),
            xz_ops: self.xz_ops.clone(),
            xyz_ops: self.xyz_ops.clone(),
            output: self.output,
            live_xz: self.live_xz.clone(),
            band: self.band,
            #[cfg(test)]
            column_evals: AtomicU64::new(self.column_evals.load(Ordering::Relaxed)),
        }
    }
}

/// Compiles a graph into a [`CompiledGraph`]. Validates what parsing cannot
/// (dangling refs into cycles, spline monotonicity, clamp order), folding
/// constants along the way. Never panics on bad input.
pub fn compile(graph: &Graph) -> Result<CompiledGraph, GraphError> {
    let mut compiler = Compiler::new(graph);
    let output = compiler.emit_value(&graph.outputs.final_density, "outputs.final_density")?;

    let n_const = compiler.dims.iter().filter(|&d| *d == Dim::Const).count();
    let n_xz = compiler.dims.iter().filter(|&d| *d == Dim::Xz).count();
    // Old emission id → final register index, staged const | xz | xyz.
    let mut final_index = vec![0u32; compiler.emitted.len()];
    {
        let (mut c, mut x, mut y) = (0usize, n_const, n_const + n_xz);
        for (i, &dim) in compiler.dims.iter().enumerate() {
            final_index[i] = match dim {
                Dim::Const => {
                    let t = c;
                    c += 1;
                    t
                }
                Dim::Xz => {
                    let t = x;
                    x += 1;
                    t
                }
                Dim::Xyz => {
                    let t = y;
                    y += 1;
                    t
                }
            } as u32;
        }
    }

    let mut const_values = Vec::with_capacity(n_const);
    let mut xz_ops = Vec::with_capacity(n_xz);
    let mut xyz_ops = Vec::new();
    for (emitted, dim) in compiler.emitted.into_iter().zip(compiler.dims) {
        match emitted {
            Emitted::Const(v) => const_values.push(v),
            Emitted::Op(op) => {
                let op = op.remap(&final_index);
                match dim {
                    // Folded constants are `Emitted::Const`, never this arm.
                    Dim::Const => unreachable!("const-dim op"),
                    Dim::Xz => xz_ops.push(op),
                    Dim::Xyz => xyz_ops.push(op),
                }
            }
        }
    }

    // Dead-code elimination: folding a parent leaves its constant leaves
    // dead. Keep only registers reachable from the output so constant
    // subgraphs really do collapse to a single folded value.
    let output = final_index[output as usize];
    let total = const_values.len() + xz_ops.len() + xyz_ops.len();
    let mut keep = vec![false; total];
    keep[output as usize] = true;
    for (i, op) in xyz_ops.iter().enumerate().rev() {
        if keep[n_const + n_xz + i] {
            for operand in op.operands() {
                keep[operand as usize] = true;
            }
        }
    }
    for (i, op) in xz_ops.iter().enumerate().rev() {
        if keep[n_const + i] {
            for operand in op.operands() {
                keep[operand as usize] = true;
            }
        }
    }
    let mut new_index = vec![0u32; total];
    let mut next = 0u32;
    for r in 0..total {
        if keep[r] {
            new_index[r] = next;
            next += 1;
        }
    }
    let const_values = (0..n_const)
        .filter(|&r| keep[r])
        .map(|r| const_values[r])
        .collect::<Vec<_>>();
    let xz_ops = xz_ops
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep[n_const + i])
        .map(|(_, op)| op.remap(&new_index))
        .collect::<Vec<_>>();
    let xyz_ops = xyz_ops
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep[n_const + n_xz + i])
        .map(|(_, op)| op.remap(&new_index))
        .collect::<Vec<_>>();
    let output = new_index[output as usize];

    let mut live_xz: Vec<u32> = xyz_ops
        .iter()
        .flat_map(|op| op.operands())
        .filter(|&r| (n_const as u32..(n_const + n_xz) as u32).contains(&r))
        .collect();
    // A column-stage output (a graph with no cell stage at all) must stay
    // live too, or a restored column would read a stale register.
    if (n_const as u32..(n_const + n_xz) as u32).contains(&output) {
        live_xz.push(output);
    }
    live_xz.sort_unstable();
    live_xz.dedup();

    let compiled = CompiledGraph {
        const_values,
        xz_ops,
        xyz_ops,
        output,
        live_xz,
        band: None,
        #[cfg(test)]
        column_evals: AtomicU64::new(0),
    };
    Ok(CompiledGraph {
        band: compute_band(&compiled),
        ..compiled
    })
}

/// Bisects the solidity band of a compiled graph. Both searches rely on
/// interval bounds growing monotonically as the y interval widens.
fn compute_band(graph: &CompiledGraph) -> Option<Band> {
    const COORD: i64 = 1 << 40;
    // Greatest Y such that every row in (-2^40, Y] is provably solid.
    let solid_below = {
        let provable = |y: i64| graph.bounds([-COORD, -COORD, -COORD], [COORD, y, COORD]).0 >= 0.0;
        if !provable(-COORD) {
            None
        } else if provable(COORD) {
            Some(COORD)
        } else {
            let (mut lo, mut hi) = (-COORD, COORD);
            while lo + 1 < hi {
                let mid = lo.midpoint(hi);
                if provable(mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Some(lo)
        }
    };
    // Least Y such that every row in [Y, 2^40) is provably air.
    let air_above = {
        let provable = |y: i64| graph.bounds([-COORD, y, -COORD], [COORD, COORD, COORD]).1 < 0.0;
        if !provable(COORD) {
            None
        } else if provable(-COORD) {
            Some(-COORD)
        } else {
            let (mut lo, mut hi) = (-COORD, COORD);
            while lo + 1 < hi {
                let mid = lo.midpoint(hi);
                if provable(mid) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            Some(hi)
        }
    };
    let (low, high) = (solid_below?, air_above?);
    let rows = (high - low - 1).max(0) as usize;
    if rows > MAX_BAND_ROWS {
        return None;
    }
    Some(Band { low, high, rows })
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

/// Side length of the grid [`CompiledGraph::sample_grid`] fills.
pub const GRID_SIDE: usize = 32;
/// Number of samples [`CompiledGraph::sample_grid`] fills.
pub const GRID_VOLUME: usize = GRID_SIDE * GRID_SIDE * GRID_SIDE;

impl CompiledGraph {
    /// Instruction counts per stage: `(constant, column, cell)`. Constant
    /// instructions are folded values, not evaluated instructions.
    pub fn instruction_counts(&self) -> (usize, usize, usize) {
        (
            self.const_values.len(),
            self.xz_ops.len(),
            self.xyz_ops.len(),
        )
    }

    /// Total instructions across all stages.
    pub fn instruction_count(&self) -> usize {
        self.const_values.len() + self.xz_ops.len() + self.xyz_ops.len()
    }

    /// Registers the evaluator uses; size your scratch arrays with this.
    pub fn register_count(&self) -> usize {
        self.const_values.len() + self.xz_ops.len() + self.xyz_ops.len()
    }

    /// Column-stage registers the cell stage reads. A per-column cache
    /// storing exactly these values (in this order) lets vertically
    /// adjacent cubes skip the whole column stage; see `terrain::Generator`.
    pub fn live_xz(&self) -> &[u32] {
        &self.live_xz
    }

    /// The solidity band ([`Band`]) if the interval bounds prove one.
    pub fn band(&self) -> Option<&Band> {
        self.band.as_ref()
    }

    /// A deterministic fingerprint of the compiled graph: equal graphs
    /// (same instructions, constants, salts and spline points, bit for bit)
    /// hash equal, and any edit to the graph changes it with overwhelming
    /// probability. Persistence (task 2.7) stores it mixed with the world
    /// seed so a changed worldgen invalidates stored LOD without touching
    /// stored LOD0 records. `DefaultHasher` is seeded with fixed keys, so
    /// the value is stable across runs and platforms (f64s enter as their
    /// bit patterns).
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.const_values.len().hash(&mut hasher);
        for &value in &self.const_values {
            value.to_bits().hash(&mut hasher);
        }
        self.xz_ops.len().hash(&mut hasher);
        for op in &self.xz_ops {
            op.feed(&mut hasher);
        }
        self.xyz_ops.len().hash(&mut hasher);
        for op in &self.xyz_ops {
            op.feed(&mut hasher);
        }
        self.output.hash(&mut hasher);
        self.live_xz.hash(&mut hasher);
        match &self.band {
            None => 0u8.hash(&mut hasher),
            Some(band) => (band.low(), band.high(), band.rows()).hash(&mut hasher),
        }
        hasher.finish()
    }

    /// Prepares `regs` with the folded constants and, optionally, cached
    /// values for [`Self::live_xz`] (a slice of `live_xz().len()` values for
    /// one column). Afterwards [`Self::eval_cell_density`] can run for any
    /// `y` of that column.
    pub fn restore_column(&self, regs: &mut [f64], live: &[f64]) {
        debug_assert_eq!(live.len(), self.live_xz.len());
        regs[..self.const_values.len()].copy_from_slice(&self.const_values);
        for (slot, &reg) in self.live_xz.iter().enumerate() {
            regs[reg as usize] = live[slot];
        }
    }

    /// Runs the constant and column stages for `(x, z)`: every instruction
    /// that does not depend on `y`. Afterwards the cell stage can run for
    /// any `y` of the column.
    pub fn eval_column(&self, seed: u64, x: i64, z: i64, regs: &mut [f64]) {
        #[cfg(test)]
        self.column_evals.fetch_add(1, Ordering::Relaxed);
        regs[..self.const_values.len()].copy_from_slice(&self.const_values);
        let (xf, zf) = (x as f64, z as f64);
        let base = self.const_values.len();
        for (i, op) in self.xz_ops.iter().enumerate() {
            regs[base + i] = eval_op(op, regs, xf, 0.0, zf, seed);
        }
    }

    /// Runs the cell stage at `(x, y, z)` on top of an existing column
    /// state and returns `final_density`. Only cell registers are written,
    /// so the column state survives for the next `y` of the same column.
    pub fn eval_cell_density(&self, seed: u64, x: i64, y: i64, z: i64, regs: &mut [f64]) -> f64 {
        let (xf, yf, zf) = (x as f64, y as f64, z as f64);
        let base = self.const_values.len() + self.xz_ops.len();
        for (i, op) in self.xyz_ops.iter().enumerate() {
            regs[base + i] = eval_op(op, regs, xf, yf, zf, seed);
        }
        regs[self.output as usize]
    }

    /// Density at a single block position. Allocates a scratch register
    /// array; when sampling many points prefer [`Self::sample_grid`] or the
    /// staged `eval_column` / `eval_cell_density` pair.
    pub fn sample(&self, seed: u64, at: [i64; 3]) -> f64 {
        let mut regs = vec![0.0; self.register_count()];
        self.eval_column(seed, at[0], at[2], &mut regs);
        self.eval_cell_density(seed, at[0], at[1], at[2], &mut regs)
    }

    /// Fills `out` with the densities of a 32×32×32 grid at
    /// `origin + step · (i, j, k)` over `(x = i, y = j, z = k)`, laid out
    /// as `(k · 32 + j) · 32 + i`. `step` must be a power of two in
    /// `1..=64`. The column stage runs once per grid column, whatever
    /// `step` is. Step 1 at a cube origin gives exactly the densities
    /// `terrain::Generator::generate` solidity decisions use — the entry
    /// point later LOD tasks sample at coarser steps.
    pub fn sample_grid(
        &self,
        seed: u64,
        origin: [i64; 3],
        step: u32,
        out: &mut [f64; GRID_VOLUME],
    ) {
        assert!(
            step.is_power_of_two() && (1..=64).contains(&step),
            "step must be a power of two in 1..=64, got {step}"
        );
        let step = i64::from(step);
        let mut regs = vec![0.0; self.register_count()];
        for k in 0..GRID_SIDE as i64 {
            for i in 0..GRID_SIDE as i64 {
                let (x, z) = (origin[0] + step * i, origin[2] + step * k);
                self.eval_column(seed, x, z, &mut regs);
                for j in 0..GRID_SIDE as i64 {
                    let density =
                        self.eval_cell_density(seed, x, origin[1] + step * j, z, &mut regs);
                    out[(k * GRID_SIDE as i64 + j) as usize * GRID_SIDE + i as usize] = density;
                }
            }
        }
    }

    /// Sound interval-arithmetic bounds `(min, max)` of `final_density`
    /// over the inclusive axis-aligned box `min..=max` (per axis). Every
    /// actual density in the box lies within; the bounds are not tight.
    /// Seed-independent: noise contributes its documented worst case.
    pub fn bounds(&self, min: [i64; 3], max: [i64; 3]) -> (f64, f64) {
        let x = (cast_lo(min[0]), cast_hi(max[0]));
        let y = (cast_lo(min[1]), cast_hi(max[1]));
        let z = (cast_lo(min[2]), cast_hi(max[2]));
        let mut intervals = vec![(0.0, 0.0); self.register_count()];
        for (i, &v) in self.const_values.iter().enumerate() {
            intervals[i] = (v, v);
        }
        let base = self.const_values.len();
        for (i, op) in self.xz_ops.iter().enumerate() {
            intervals[base + i] = interval_op(op, &intervals, x, y, z);
        }
        let base = base + self.xz_ops.len();
        for (i, op) in self.xyz_ops.iter().enumerate() {
            intervals[base + i] = interval_op(op, &intervals, x, y, z);
        }
        intervals[self.output as usize]
    }
}

/// Evaluates one instruction over the register array. `y` is ignored by
/// construction for column-stage instructions.
#[inline]
fn eval_op(op: &Op, r: &[f64], x: f64, y: f64, z: f64, seed: u64) -> f64 {
    let v = |i: u32| r[i as usize];
    match op {
        Op::X => x,
        Op::Y => y,
        Op::Z => z,
        Op::Add(a, b) => v(*a) + v(*b),
        Op::Sub(a, b) => v(*a) - v(*b),
        Op::Mul(a, b) => v(*a) * v(*b),
        Op::Neg(a) => -v(*a),
        Op::Min(a, b) => v(*a).min(v(*b)),
        Op::Max(a, b) => v(*a).max(v(*b)),
        Op::Abs(a) => v(*a).abs(),
        Op::Floor(a) => v(*a).floor(),
        Op::Clamp { input, min, max } => v(*input).clamp(v(*min), v(*max)),
        Op::Noise2 {
            salt,
            x_scale,
            z_scale,
        } => reference::noise2(seed ^ salt, x * x_scale, z * z_scale),
        Op::Fbm2 {
            salt,
            scale,
            octaves,
        } => reference::fbm(seed ^ salt, x * scale, z * scale, *octaves),
        Op::Spline { input, points } => spline_eval(points, v(*input)),
    }
}

/// Piecewise-linear spline, clamped at both ends. Determinism: only
/// `+ - * /`, and division is IEEE-exact.
fn spline_eval(points: &[(f64, f64)], x: f64) -> f64 {
    let &(first_x, first_y) = &points[0];
    if x <= first_x {
        return first_y;
    }
    let &(last_x, last_y) = &points[points.len() - 1];
    if x >= last_x {
        return last_y;
    }
    // Segment i holds points[i].0 <= x < points[i+1].0.
    let i = points.partition_point(|&(px, _)| px <= x) - 1;
    let (x0, y0) = points[i];
    let (x1, y1) = points[i + 1];
    y0 + (x - x0) * (y1 - y0) / (x1 - x0)
}

type Interval = (f64, f64);

/// Documented worst-case range of one `noise2` — and, after normalization,
/// `fbm2` — sample. `noise2` blends the four corner dot products `g·f`
/// (unit gradient `g`, per-axis offset `f` below 1) with fade weights that
/// are non-negative and sum to 1, so the value is a convex combination
/// bounded by the largest corner dot. That bound peaks at the cell centre,
/// where every corner offset is (±0.5, ±0.5) with `|f| = √2/2`; hence
/// ±√2/2 for unit gradients (the frozen table's observed range is ≈ ±0.7).
/// `fbm2` divides the octave sum by the sum of the (positive, total 1 in
/// the limit) amplitudes, which is the same convex combination of `noise2`
/// samples, so the bound carries over. A documented bound the cube fast
/// paths rely on, not a tight one.
const NOISE_RANGE: Interval = (
    -(std::f64::consts::SQRT_2 / 2.0),
    std::f64::consts::SQRT_2 / 2.0,
);

#[inline]
fn interval_op(op: &Op, r: &[Interval], x: Interval, y: Interval, z: Interval) -> Interval {
    let v = |i: u32| r[i as usize];
    match op {
        Op::X => x,
        Op::Y => y,
        Op::Z => z,
        Op::Add(a, b) => (v(*a).0 + v(*b).0, v(*a).1 + v(*b).1),
        Op::Sub(a, b) => (v(*a).0 - v(*b).1, v(*a).1 - v(*b).0),
        Op::Mul(a, b) => {
            let products = [
                v(*a).0 * v(*b).0,
                v(*a).0 * v(*b).1,
                v(*a).1 * v(*b).0,
                v(*a).1 * v(*b).1,
            ];
            let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (lo, hi)
        }
        Op::Neg(a) => (-v(*a).1, -v(*a).0),
        Op::Min(a, b) => (v(*a).0.min(v(*b).0), v(*a).1.min(v(*b).1)),
        Op::Max(a, b) => (v(*a).0.max(v(*b).0), v(*a).1.max(v(*b).1)),
        Op::Abs(a) => {
            let (lo, hi) = v(*a);
            if lo >= 0.0 {
                (lo, hi)
            } else if hi <= 0.0 {
                (-hi, -lo)
            } else {
                (0.0, (-lo).max(hi))
            }
        }
        Op::Floor(a) => (v(*a).0.floor(), v(*a).1.floor()),
        // clamp(v, m, M) == min(max(v, m), M); interval min/max of
        // monotone combinations keeps this sound for any operand intervals.
        Op::Clamp { input, min, max } => {
            let inner = (v(*input).0.max(v(*min).0), v(*input).1.max(v(*min).1));
            (inner.0.min(v(*max).0), inner.1.min(v(*max).1))
        }
        Op::Noise2 { .. } | Op::Fbm2 { .. } => NOISE_RANGE,
        Op::Spline { input, points } => spline_interval(points, v(*input)),
    }
}

/// Interval of a spline over an input interval: piecewise linear, so the
/// extrema sit at the interval ends or at knots strictly inside.
fn spline_interval(points: &[(f64, f64)], input: Interval) -> Interval {
    let a = spline_eval(points, input.0);
    let b = spline_eval(points, input.1);
    let mut lo = a.min(b);
    let mut hi = a.max(b);
    for &(px, py) in points {
        if px > input.0 && px < input.1 {
            lo = lo.min(py);
            hi = hi.max(py);
        }
    }
    (lo, hi)
}

/// `v` rounded toward −∞ far enough to contain `v` after the i64→f64 cast.
fn cast_lo(v: i64) -> f64 {
    let f = v as f64;
    if f as i64 > v { f.next_down() } else { f }
}

/// `v` rounded toward +∞ far enough to contain `v` after the i64→f64 cast.
fn cast_hi(v: i64) -> f64 {
    let f = v as f64;
    if (f as i64) < v { f.next_up() } else { f }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile_json(json: &str) -> Result<CompiledGraph, GraphError> {
        compile(&Graph::from_json(json)?)
    }

    fn final_only(node: &str) -> String {
        format!("{{\"version\": 1, \"nodes\": {{}}, \"outputs\": {{\"final_density\": {node}}}}}")
    }

    #[test]
    fn the_shipped_graph_loads_and_compiles() {
        let graph = Graph::from_json(super::super::OVERWORLD_JSON).expect("shipped graph parses");
        let compiled = compile(&graph).expect("shipped graph compiles");
        // 9 folded constants, 11 column-stage instructions (the noise,
        // clamps and rounding), and just 2 cell-stage instructions: `y` and
        // the final subtraction. That last stage is what 3D evaluation
        // costs beyond a heightmap.
        assert_eq!(compiled.instruction_counts(), (9, 11, 2));
        assert_eq!(compiled.live_xz().len(), 1);
    }

    #[test]
    fn salts_round_trip_through_json() {
        let graph = Graph::from_json(super::super::OVERWORLD_JSON).unwrap();
        let NodeValue::Inline(mask) = &graph.nodes["mask"] else {
            panic!("mask is an inline node");
        };
        let Node::Noise2 { salt, .. } = mask.as_ref() else {
            panic!("mask is a noise2 node");
        };
        assert_eq!(salt.0, 0x0D1B_54A3_2D16_8ED5);
        let text = serde_json::to_string(&graph).unwrap();
        assert!(
            text.contains("\"0x0D1B54A32D168ED5\""),
            "salt stays a hex string: {text}"
        );
        let reparsed = Graph::from_json(&text).unwrap();
        assert_eq!(graph, reparsed);
    }

    #[test]
    fn unknown_node_type_names_the_node() {
        let err = Graph::from_json(&final_only("{\"type\": \"sin\", \"a\": 1}")).unwrap_err();
        assert!(
            matches!(err, GraphError::UnknownNodeType { .. }),
            "got {err:?}"
        );
        assert!(
            err.to_string().contains("outputs.final_density"),
            "error names the node: {err}"
        );
        let err = compile_json(
            "{\"version\": 1, \"nodes\": {\"bad\": {\"type\": \"cos\"}}, \"outputs\": {\"final_density\": {\"ref\": \"bad\"}}}",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("nodes.bad"),
            "error names the node: {err}"
        );
    }

    #[test]
    fn missing_field_names_the_node() {
        let err = compile_json(
            "{\"version\": 1, \"nodes\": {\"hills\": {\"type\": \"noise2\", \"salt\": \"0x1\"}}, \"outputs\": {\"final_density\": {\"ref\": \"hills\"}}}",
        )
        .unwrap_err();
        assert!(
            matches!(err, GraphError::MissingField { .. }),
            "got {err:?}"
        );
        assert!(
            err.to_string().contains("nodes.hills"),
            "error names the node: {err}"
        );
    }

    #[test]
    fn dangling_ref_names_the_node() {
        let err = compile_json(&final_only("{\"ref\": \"nope\"}")).unwrap_err();
        assert!(matches!(err, GraphError::DanglingRef { .. }), "got {err:?}");
        assert!(
            err.to_string().contains("outputs.final_density"),
            "error names the node: {err}"
        );
    }

    #[test]
    fn a_cycle_names_the_node() {
        let err = compile_json(
            "{\"version\": 1, \"nodes\": {\"a\": {\"type\": \"add\", \"a\": {\"ref\": \"b\"}, \"b\": 1}, \"b\": {\"type\": \"mul\", \"a\": {\"ref\": \"a\"}, \"b\": 2}}, \"outputs\": {\"final_density\": {\"ref\": \"a\"}}}",
        )
        .unwrap_err();
        assert!(matches!(err, GraphError::Cycle { .. }), "got {err:?}");
        assert!(
            err.to_string().contains("nodes."),
            "error names a node: {err}"
        );
    }

    #[test]
    fn a_bad_spline_names_the_node() {
        for points in [
            "[[0.0, 1.0]]",
            "[[0.0, 1.0], [0.0, 2.0]]",
            "[[1.0, 1.0], [0.0, 2.0]]",
        ] {
            let err = compile_json(&format!(
                "{{\"version\": 1, \"nodes\": {{\"s\": {{\"type\": \"spline\", \"input\": 0.5, \"points\": {points}}}}}, \"outputs\": {{\"final_density\": {{\"ref\": \"s\"}}}}}}"
            ))
            .unwrap_err();
            assert!(matches!(err, GraphError::BadSpline { .. }), "got {err:?}");
            assert!(
                err.to_string().contains("nodes.s"),
                "error names the node: {err}"
            );
        }
    }

    #[test]
    fn unknown_version_is_rejected() {
        let err = Graph::from_json(&final_only("1").replace("\"version\": 1", "\"version\": 2"))
            .unwrap_err();
        assert!(
            matches!(err, GraphError::UnknownVersion { found: 2, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn missing_final_density_is_rejected() {
        let err = Graph::from_json("{\"version\": 1, \"nodes\": {}}").unwrap_err();
        assert!(
            matches!(err, GraphError::MissingFinalDensity),
            "got {err:?}"
        );
        let err = Graph::from_json("{\"version\": 1, \"nodes\": {}, \"outputs\": {}}").unwrap_err();
        assert!(
            matches!(err, GraphError::MissingFinalDensity),
            "got {err:?}"
        );
    }

    #[test]
    fn malformed_input_never_panics() {
        let bad = [
            "",
            "{",
            "[]",
            "null",
            "{\"version\": 1, \"nodes\": {\"n\": true}, \"outputs\": {\"final_density\": {\"ref\": \"n\"}}}",
            "{\"version\": 1, \"nodes\": {\"n\": \"text\"}, \"outputs\": {\"final_density\": {\"ref\": \"n\"}}}",
            "{\"version\": 1, \"nodes\": {\"n\": {\"ref\": 3}}, \"outputs\": {\"final_density\": {\"ref\": \"n\"}}}",
            "{\"version\": 1, \"nodes\": {\"n\": {\"type\": 4}}, \"outputs\": {\"final_density\": {\"ref\": \"n\"}}}",
            "{\"version\": 1, \"nodes\": {\"n\": {\"type\": \"noise2\", \"salt\": 12, \"x_scale\": 1, \"z_scale\": 1}}, \"outputs\": {\"final_density\": {\"ref\": \"n\"}}}",
            "{\"version\": 1, \"nodes\": {\"n\": {\"type\": \"noise2\", \"salt\": \"0xZZ\", \"x_scale\": 1, \"z_scale\": 1}}, \"outputs\": {\"final_density\": {\"ref\": \"n\"}}}",
            "{\"version\": 1, \"nodes\": {\"n\": {\"type\": \"fbm2\", \"salt\": \"0x1\", \"scale\": 1, \"octaves\": 0}}, \"outputs\": {\"final_density\": {\"ref\": \"n\"}}}",
            "{\"version\": 1, \"nodes\": {\"n\": {\"type\": \"clamp\", \"input\": 1, \"min\": 5, \"max\": 1}}, \"outputs\": {\"final_density\": {\"ref\": \"n\"}}}",
            "{\"version\": 1, \"nodes\": {\"n\": {\"type\": \"spline\", \"input\": 1, \"points\": [[1, 2, 3]]}}, \"outputs\": {\"final_density\": {\"ref\": \"n\"}}}",
            "{\"version\": 1, \"nodes\": {\"n\": {\"type\": \"add\", \"a\": 1, \"typo\": 2}}, \"outputs\": {\"final_density\": {\"ref\": \"n\"}}}",
            "{\"version\": 1, \"nodes\": {}, \"outputs\": {\"final_density\": {\"type\": \"add\", \"a\": {}}}}",
        ];
        for json in bad {
            assert!(compile_json(json).is_err(), "expected an error for {json}");
        }
    }

    #[test]
    fn constants_fold_at_compile_time() {
        let compiled = compile_json(&final_only(
            "{\"type\": \"add\", \"a\": {\"type\": \"mul\", \"a\": 2, \"b\": 3}, \"b\": 0.5}",
        ))
        .unwrap();
        // One folded constant (6.5), no column or cell instructions.
        assert_eq!(compiled.instruction_counts(), (1, 0, 0));
        let mut regs = vec![0.0; compiled.register_count()];
        compiled.eval_column(0, 0, 0, &mut regs);
        assert_eq!(compiled.eval_cell_density(0, 0, 0, 0, &mut regs), 6.5);
    }

    #[test]
    fn shared_named_nodes_are_compiled_once() {
        let compiled = compile_json(
            "{\"version\": 1, \"nodes\": {\"n\": {\"type\": \"noise2\", \"salt\": \"0x1\", \"x_scale\": 0.01, \"z_scale\": 0.01}}, \"outputs\": {\"final_density\": {\"type\": \"add\", \"a\": {\"ref\": \"n\"}, \"b\": {\"ref\": \"n\"}}}}",
        )
        .unwrap();
        // One noise2 plus one add, not two noise evaluations.
        assert_eq!(compiled.instruction_counts(), (0, 2, 0));
        // ... and one column evaluation per sampled column, not one per use.
        let before = compiled.column_evals.load(Ordering::Relaxed);
        let _ = compiled.sample(7, [10, 20, 30]);
        assert_eq!(compiled.column_evals.load(Ordering::Relaxed) - before, 1);
    }

    #[test]
    fn column_instructions_run_once_per_column() {
        let compiled = compile_json(
            "{\"version\": 1, \"nodes\": {}, \"outputs\": {\"final_density\": {\"type\": \"add\", \"a\": {\"type\": \"noise2\", \"salt\": \"0x1\", \"x_scale\": 0.01, \"z_scale\": 0.01}, \"b\": {\"type\": \"y\"}}}}",
        )
        .unwrap();
        let mut out = [0.0; GRID_VOLUME];
        let before = compiled.column_evals.load(Ordering::Relaxed);
        compiled.sample_grid(1, [0, 0, 0], 1, &mut out);
        // Exactly one column stage per (x, z) column of the 32×32×32 grid.
        assert_eq!(
            compiled.column_evals.load(Ordering::Relaxed) - before,
            (GRID_SIDE * GRID_SIDE) as u64
        );
    }

    #[test]
    fn sample_grid_at_steps_2_4_8_equals_point_evaluation() {
        let graph = Graph::from_json(super::super::OVERWORLD_JSON).unwrap();
        let compiled = compile(&graph).unwrap();
        let mut out = [0.0; GRID_VOLUME];
        for step in [2u32, 4, 8] {
            let origin = [1234i64, 0, -5678];
            compiled.sample_grid(42, origin, step, &mut out);
            for k in 0..GRID_SIDE as i64 {
                for j in 0..GRID_SIDE as i64 {
                    for i in 0..GRID_SIDE as i64 {
                        let at = [
                            origin[0] + i64::from(step) * i,
                            origin[1] + i64::from(step) * j,
                            origin[2] + i64::from(step) * k,
                        ];
                        let expected = compiled.sample(42, at);
                        let got = out[(k * GRID_SIDE as i64 + j) as usize * GRID_SIDE + i as usize];
                        assert_eq!(expected.to_bits(), got.to_bits(), "step {step} at {at:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn bounds_contain_sampled_values_over_random_boxes() {
        let graph = Graph::from_json(super::super::OVERWORLD_JSON).unwrap();
        let compiled = compile(&graph).unwrap();
        // Deterministic LCG so failures reproduce.
        let mut state = 0x5EED_2026_0000_0001u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for _ in 0..50 {
            let span = |v: u64| 1 + (v % 64) as i64;
            let lo = [
                ((next() % 20001) as i64) - 10000,
                ((next() % 129) as i64),
                ((next() % 20001) as i64) - 10000,
            ];
            let hi = [
                lo[0] + span(next()),
                lo[1] + span(next()),
                lo[2] + span(next()),
            ];
            let (min, max) = compiled.bounds(lo, hi);
            for _ in 0..20 {
                let at = [
                    lo[0] + (next() % (hi[0] - lo[0] + 1) as u64) as i64,
                    lo[1] + (next() % (hi[1] - lo[1] + 1) as u64) as i64,
                    lo[2] + (next() % (hi[2] - lo[2] + 1) as u64) as i64,
                ];
                let density = compiled.sample(99, at);
                assert!(
                    density >= min && density <= max,
                    "density {density} at {at:?} outside bounds [{min}, {max}]"
                );
            }
        }
    }

    #[test]
    fn shipped_graph_bounds_prove_the_documented_height_range() {
        let compiled = compile(&Graph::from_json(super::super::OVERWORLD_JSON).unwrap()).unwrap();
        for y in (crate::terrain::MAX_HEIGHT + 1)..=128i64 {
            let (_, max) = compiled.bounds([-64, y, -64], [64, y, 64]);
            assert!(max < 0.0, "air not proven at y = {y}: max = {max}");
        }
        for y in 0..=(crate::terrain::MIN_HEIGHT - 1) {
            let (min, _) = compiled.bounds([-64, y, -64], [64, y, 64]);
            assert!(min >= 0.0, "solid not proven at y = {y}: min = {min}");
        }
    }
}
