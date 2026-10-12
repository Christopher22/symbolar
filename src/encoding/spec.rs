#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The complete, data-independent description of an encoder.
///
/// All data-dependent decisions (category lists, level boundaries, scaling of continuous values) are expected to be
/// resolved beforehand, so that the same specification always yields the same codebook for a given architecture seed.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EncoderSpec {
    /// The dimensionality of the vectors.
    pub size: usize,
    /// How the bindings of a record are combined.
    pub structure: Structure,
    /// For [`Structure::Hierarchical`]: whether the bundle of each module is normalized before it is bound to its module vector.
    #[cfg_attr(feature = "serde", serde(default = "default_true"))]
    pub normalize_modules: bool,
    /// The variables, in the order of the columns passed during encoding.
    pub variables: Vec<VariableSpec>,
}

#[cfg(feature = "serde")]
fn default_true() -> bool {
    true
}

/// How the bindings of a record are combined into a single vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Structure {
    /// `P = Σ_v R_v ⊛ Φ_v(x_v)`; modules are ignored.
    Flat,
    /// `P = Σ_m M_m ⊛ norm(Σ_{v ∈ m} R_v ⊛ Φ_v(x_v))`; variables without a module are added directly.
    Hierarchical,
}

/// A variable, i.e. a role together with all fillers it may be bound to.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VariableSpec {
    /// The name of the variable, which is also the name of its role vector.
    pub name: String,
    /// The module the variable belongs to. Its vector is named `@<module>`.
    #[cfg_attr(feature = "serde", serde(default))]
    pub module: Option<String>,
    /// The finite fillers of the variable, addressed by their index during encoding.
    #[cfg_attr(feature = "serde", serde(default))]
    pub keys: Vec<KeySpec>,
    /// An optional continuous filler, used for values which are not addressed by a key.
    #[cfg_attr(feature = "serde", serde(default))]
    pub continuous: Option<ContinuousSpec>,
}

/// A labelled filler of a variable.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct KeySpec {
    /// The label used to refer to the filler in expressions (`variable:label`).
    pub label: String,
    /// The construction of the filler.
    pub filler: Filler,
}

/// The construction of a filler vector.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Filler {
    /// A random vector derived from its name.
    Symbol(String),
    /// Level `index` out of `count` levels interpolated between two random vectors, named `<pool>#<index>`.
    /// Neighbouring levels are similar; the similarity decreases linearly until the first and last level are unrelated.
    Level {
        /// The name of the level series. Variables sharing a pool share their level vectors.
        pool: String,
        /// The level, `0 <= index < count`.
        index: usize,
        /// The number of levels in the series.
        count: usize,
    },
    /// The normalized sum of fillers, e.g. a code together with its ancestors in a hierarchy.
    Bundle(Vec<Filler>),
    /// The binding of fillers, e.g. an event bound to the time since it happened.
    Bind(Vec<Filler>),
}

/// A filler for continuous values using fractional power encoding: `B^x`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ContinuousSpec {
    /// The name of the base vector `B`.
    pub base: String,
    /// The similarity kernel `⟨B^x, B^y⟩ ≈ k(x - y)`. Values are expected to be scaled by the bandwidth beforehand.
    #[cfg_attr(feature = "serde", serde(default))]
    pub kernel: Kernel,
}

/// The similarity kernel of a fractional power encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Kernel {
    /// `sinc(Δ)`, from phases uniformly distributed on `(-π, π)`.
    #[default]
    Sinc,
    /// `exp(-Δ² / 2)`, from normally distributed phases.
    Gaussian,
}
