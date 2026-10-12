//! Encoding of tabular records into hypervectors.
//!
//! Each record is encoded as a superposition of role-filler bindings `R_v ⊛ Φ_v(x_v)`, optionally grouped into
//! modules (`M_m ⊛ norm(Σ_{v ∈ m} R_v ⊛ Φ_v(x_v))`). The [`EncoderSpec`] fully determines the codebook: role, module
//! and symbol vectors are derived from their names and the seed of the architecture, while levels, bundles and
//! bindings are derived from those. Data-dependent decisions (categories, level boundaries, scaling of continuous
//! values) are resolved before the specification is created, e.g. on the training data only.
//!
//! Values are passed as indices into the keys of each variable ([`Column`]), with [`OMITTED`] marking cells that do
//! not contribute to the record (e.g. missing values).
//!
//! Since binding distributes over bundling, the similarity of a record to a full key `M_m ⊛ R_v ⊛ Φ_v(k)` equals
//! (up to a constant factor) the similarity of the unbound record `P ⊘ M_m ⊘ R_v` to `Φ_v(k)` for all architectures,
//! as their (approximate) inverses are the adjoints of the binding. Hence, decoding, querying and explaining reduce
//! to dot products with the vectors returned by [`Encoder::keys`].

mod primitives;
mod spec;

use std::{borrow::Cow, collections::HashMap, num::NonZero};

use rustfft::FftPlanner;

use self::primitives::{FractionalPower, level, to_dense};
pub use self::spec::{
    ContinuousSpec, EncoderSpec, Filler, Kernel, KeySpec, Structure, VariableSpec,
};
use crate::{
    Dynamic, Expression, Normalized, Size, Storage, UnknownValue, Vector,
    architectures::VectorSymbolicArchitecture,
};

/// The code marking a cell which does not contribute to the record.
pub const OMITTED: u32 = u32::MAX;

/// The values of a variable for all records.
#[derive(Debug, Clone, Copy)]
pub enum Column<'a> {
    /// Indices into the keys of the variable, or [`OMITTED`].
    Codes(&'a [u32]),
    /// Indices into the keys of the variable, falling back to the continuous filler at the (scaled) value where the
    /// code is [`OMITTED`]. A `NaN` value is omitted as well.
    Continuous {
        /// Indices into the keys of the variable, or [`OMITTED`].
        codes: &'a [u32],
        /// The values passed to the continuous filler.
        values: &'a [f64],
    },
}

impl Column<'_> {
    fn len(&self) -> usize {
        match self {
            Column::Codes(codes) | Column::Continuous { codes, .. } => codes.len(),
        }
    }

    fn codes(&self) -> &[u32] {
        match self {
            Column::Codes(codes) | Column::Continuous { codes, .. } => codes,
        }
    }
}

/// An element type the dense representation of records can be written to.
pub trait DenseElement: Copy + Send + Sync {
    /// Whether the element type can only represent normalized vectors of bipolar architectures.
    const BIPOLAR_ONLY: bool;

    /// Convert a dense value.
    fn from_f32(value: f32) -> Self;
}

impl DenseElement for f32 {
    const BIPOLAR_ONLY: bool = false;

    fn from_f32(value: f32) -> Self {
        value
    }
}

impl DenseElement for i8 {
    const BIPOLAR_ONLY: bool = true;

    fn from_f32(value: f32) -> Self {
        value as i8
    }
}

/// An error during the construction or use of an [`Encoder`].
#[derive(Debug, Clone, PartialEq)]
pub enum EncodingError {
    /// The size is invalid for the architecture.
    InvalidSize(usize),
    /// Fractional power encoding requires a real-valued architecture.
    ContinuousUnsupported {
        /// The variable with the continuous filler.
        variable: String,
    },
    /// A level is out of range or the pool was used with different numbers of levels.
    InvalidLevel {
        /// The pool of the level.
        pool: String,
    },
    /// A variable name was used twice.
    DuplicateVariable(String),
    /// A label was used twice within a variable.
    DuplicateLabel {
        /// The variable.
        variable: String,
        /// The duplicated label.
        label: String,
    },
    /// A variable is not known.
    UnknownVariable(String),
    /// A label is not known for a variable.
    UnknownLabel {
        /// The variable.
        variable: String,
        /// The unknown label.
        label: String,
    },
    /// The variable has no continuous filler.
    NotContinuous(String),
    /// The number of columns does not match the number of variables.
    ColumnCount {
        /// The number of variables.
        expected: usize,
        /// The number of columns.
        found: usize,
    },
    /// The columns or the output do not match in their number of records.
    RowCount {
        /// The expected number of records.
        expected: usize,
        /// The number of records found.
        found: usize,
    },
    /// A code exceeds the number of keys of a variable.
    CodeOutOfRange {
        /// The variable.
        variable: String,
        /// The invalid code.
        code: u32,
    },
    /// The requested output is incompatible with the architecture or the normalization.
    InvalidOutput(&'static str),
    /// A name in an expression could not be resolved.
    UnknownValue(UnknownValue),
}

impl std::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSize(size) => write!(f, "invalid vector size {size} for the architecture"),
            Self::ContinuousUnsupported { variable } => write!(
                f,
                "variable '{variable}': fractional power encoding requires a real-valued architecture (HRR, VTB); use levels instead"
            ),
            Self::InvalidLevel { pool } => write!(
                f,
                "level pool '{pool}': level out of range or used with different numbers of levels"
            ),
            Self::DuplicateVariable(name) => write!(f, "variable '{name}' is defined twice"),
            Self::DuplicateLabel { variable, label } => {
                write!(f, "variable '{variable}': label '{label}' is defined twice")
            }
            Self::UnknownVariable(name) => write!(f, "unknown variable '{name}'"),
            Self::UnknownLabel { variable, label } => {
                write!(f, "variable '{variable}': unknown label '{label}'")
            }
            Self::NotContinuous(name) => write!(f, "variable '{name}' has no continuous filler"),
            Self::ColumnCount { expected, found } => {
                write!(
                    f,
                    "expected {expected} columns (one per variable), found {found}"
                )
            }
            Self::RowCount { expected, found } => {
                write!(f, "expected {expected} records, found {found}")
            }
            Self::CodeOutOfRange { variable, code } => {
                write!(
                    f,
                    "variable '{variable}': code {code} exceeds the number of keys"
                )
            }
            Self::InvalidOutput(reason) => f.write_str(reason),
            Self::UnknownValue(value) => write!(f, "unknown value '{value}'"),
        }
    }
}

impl std::error::Error for EncodingError {}

impl From<UnknownValue> for EncodingError {
    fn from(value: UnknownValue) -> Self {
        Self::UnknownValue(value)
    }
}

#[derive(Debug, Clone)]
struct PreparedVariable<V: VectorSymbolicArchitecture> {
    module: Option<usize>,
    /// `R_v`, or `M_m ⊛ R_v` if the module is bound within the keys.
    base: V::Storage,
    /// `base ⊛ Φ_v(k)` for all keys.
    keys: Vec<V::Storage>,
    labels: Vec<String>,
    label_index: HashMap<String, usize>,
    continuous: Option<FractionalPower>,
}

/// Encodes records into hypervectors according to an [`EncoderSpec`].
#[derive(Debug, Clone)]
pub struct Encoder<V: VectorSymbolicArchitecture> {
    spec: EncoderSpec,
    codebook: Storage<Dynamic, V>,
    variables: Vec<PreparedVariable<V>>,
    variable_index: HashMap<String, usize>,
    modules: Vec<V::Storage>,
}

impl<V: VectorSymbolicArchitecture> Encoder<V> {
    /// Create the encoder, generating the codebook and caching all bound keys.
    pub fn new(vsa: V, spec: EncoderSpec) -> Result<Self, EncodingError> {
        let size = Dynamic::from(NonZero::new(spec.size).ok_or(EncodingError::InvalidSize(0))?);
        let mut codebook =
            Storage::new(vsa.clone(), size).ok_or(EncodingError::InvalidSize(spec.size))?;
        let inverse_fft = FftPlanner::<f64>::new().plan_fft_inverse(spec.size);

        let mut module_index: HashMap<String, usize> = HashMap::new();
        let mut modules = Vec::new();
        let mut level_counts: HashMap<String, usize> = HashMap::new();
        let mut variables = Vec::with_capacity(spec.variables.len());
        let mut variable_index = HashMap::with_capacity(spec.variables.len());
        let module_in_keys = Self::module_in_keys(&spec);

        for variable in &spec.variables {
            if variable_index
                .insert(variable.name.clone(), variables.len())
                .is_some()
            {
                return Err(EncodingError::DuplicateVariable(variable.name.clone()));
            }

            let module = match (&variable.module, spec.structure) {
                (Some(name), Structure::Hierarchical) => {
                    Some(*module_index.entry(name.clone()).or_insert_with(|| {
                        modules.push(codebook.push(format!("@{name}")).data.0.clone());
                        modules.len() - 1
                    }))
                }
                _ => None,
            };

            let mut base = codebook.push(variable.name.as_str()).data.0.clone();
            if let (true, Some(module)) = (module_in_keys, module) {
                V::bind(&mut base, &modules[module]);
            }

            let mut keys = Vec::with_capacity(variable.keys.len());
            let mut labels = Vec::with_capacity(variable.keys.len());
            let mut label_index = HashMap::with_capacity(variable.keys.len());
            for key in &variable.keys {
                if label_index.insert(key.label.clone(), keys.len()).is_some() {
                    return Err(EncodingError::DuplicateLabel {
                        variable: variable.name.clone(),
                        label: key.label.clone(),
                    });
                }
                let mut bound = base.clone();
                V::bind(
                    &mut bound,
                    &Self::build_filler(&mut codebook, &mut level_counts, &key.filler)?,
                );
                keys.push(bound);
                labels.push(key.label.clone());
            }

            let continuous = match &variable.continuous {
                Some(_) if !V::REAL_VALUED => {
                    return Err(EncodingError::ContinuousUnsupported {
                        variable: variable.name.clone(),
                    });
                }
                Some(continuous) => Some(FractionalPower::new(
                    vsa.seed(),
                    &continuous.base,
                    continuous.kernel,
                    inverse_fft.clone(),
                )),
                None => None,
            };

            variables.push(PreparedVariable {
                module,
                base,
                keys,
                labels,
                label_index,
                continuous,
            });
        }

        Ok(Self {
            spec,
            codebook,
            variables,
            variable_index,
            modules,
        })
    }

    fn module_in_keys(spec: &EncoderSpec) -> bool {
        spec.structure == Structure::Hierarchical && !spec.normalize_modules
    }

    fn build_filler(
        codebook: &mut Storage<Dynamic, V>,
        level_counts: &mut HashMap<String, usize>,
        filler: &Filler,
    ) -> Result<V::Storage, EncodingError> {
        let size = codebook.size();
        Ok(match filler {
            Filler::Symbol(name) => codebook.push(name.as_str()).data.0.clone(),
            Filler::Level { pool, index, count } => {
                let known_count = *level_counts.entry(pool.clone()).or_insert(*count);
                if *index >= *count || known_count != *count {
                    return Err(EncodingError::InvalidLevel { pool: pool.clone() });
                }

                let name = format!("{pool}#{index}");
                if let Some(vector) = codebook.get(&name.as_str()) {
                    vector.data.0.clone()
                } else {
                    let vsa = codebook.architecture().clone();
                    let storage = level(&vsa, size.size(), pool, *index, *count);
                    codebook
                        .insert(name, Vector::from_normalized(vsa, size, storage.clone()))
                        .expect("same size");
                    storage
                }
            }
            Filler::Bundle(children) => {
                let mut dense = vec![0.0; size.size()];
                for child in children {
                    V::accumulate_dense(
                        &Self::build_filler(codebook, level_counts, child)?,
                        &mut dense,
                        1.0,
                    );
                }
                codebook.architecture().normalize_dense(&dense)
            }
            Filler::Bind(children) => {
                let mut result: Option<V::Storage> = None;
                for child in children {
                    let child = Self::build_filler(codebook, level_counts, child)?;
                    match result.as_mut() {
                        Some(result) => V::bind(result, &child),
                        None => result = Some(child),
                    }
                }
                result.unwrap_or_else(|| {
                    codebook
                        .architecture()
                        .normalize_dense(&vec![0.0; size.size()])
                })
            }
        })
    }

    /// The specification of the encoder.
    pub fn spec(&self) -> &EncoderSpec {
        &self.spec
    }

    /// The dimensionality of the vectors.
    pub fn size(&self) -> usize {
        self.spec.size
    }

    /// The codebook with all named vectors: roles (`<variable>`), modules (`@<module>`), symbols and levels (`<pool>#<index>`).
    pub fn codebook(&self) -> &Storage<Dynamic, V> {
        &self.codebook
    }

    fn variable(&self, name: &str) -> Result<&PreparedVariable<V>, EncodingError> {
        self.variable_index
            .get(name)
            .map(|index| &self.variables[*index])
            .ok_or_else(|| EncodingError::UnknownVariable(name.to_string()))
    }

    /// The labels of the keys of a variable, in the order of their codes.
    pub fn labels(&self, variable: &str) -> Result<&[String], EncodingError> {
        Ok(&self.variable(variable)?.labels)
    }

    /// Bind the module vector to a key, if it is not already part of it.
    fn complete(&self, variable: &PreparedVariable<V>, mut storage: V::Storage) -> V::Storage {
        if let (false, Some(module)) = (Self::module_in_keys(&self.spec), variable.module) {
            V::bind(&mut storage, &self.modules[module]);
        }
        storage
    }

    /// The full keys `M_m ⊛ R_v ⊛ Φ_v(k)` of a variable, in the order of their codes.
    pub fn keys(&self, variable: &str) -> Result<Vec<V::Storage>, EncodingError> {
        let prepared = self.variable(variable)?;
        Ok(prepared
            .keys
            .iter()
            .map(|key| self.complete(prepared, key.clone()))
            .collect())
    }

    /// The full key of a single label.
    pub fn key(&self, variable: &str, label: &str) -> Result<V::Storage, EncodingError> {
        let prepared = self.variable(variable)?;
        let index = prepared
            .label_index
            .get(label)
            .ok_or_else(|| EncodingError::UnknownLabel {
                variable: variable.to_string(),
                label: label.to_string(),
            })?;
        Ok(self.complete(prepared, prepared.keys[*index].clone()))
    }

    /// The full keys `M_m ⊛ R_v ⊛ B^x` of a continuous variable for (scaled) values `x`.
    pub fn continuous_keys(
        &self,
        variable: &str,
        values: &[f64],
    ) -> Result<Vec<V::Storage>, EncodingError> {
        let prepared = self.variable(variable)?;
        let continuous = prepared
            .continuous
            .as_ref()
            .ok_or_else(|| EncodingError::NotContinuous(variable.to_string()))?;
        let vsa = self.codebook.architecture();
        Ok(values
            .iter()
            .map(|value| {
                let mut key = prepared.base.clone();
                V::bind(&mut key, &vsa.normalize_dense(&continuous.dense(*value)));
                self.complete(prepared, key)
            })
            .collect())
    }

    /// The dense (bipolar or real-valued) representation of a vector.
    pub fn to_dense(&self, storage: &V::Storage) -> Vec<f32> {
        to_dense::<V>(storage, self.spec.size)
    }

    /// Evaluate an expression. Besides the names in the [codebook](Self::codebook), it resolves `<variable>:<label>`
    /// to the full key of the label and `<variable>:=<value>` to the full continuous key of a (scaled) value.
    pub fn evaluate(
        &self,
        expression: &Expression,
    ) -> Result<Vector<Dynamic, V, Normalized<V>>, EncodingError> {
        let vsa = self.codebook.architecture();
        let size = self.codebook.size();
        let resolved = expression.evaluate(|name| {
            if let Some(vector) = self.codebook.get(&name) {
                return Some(Cow::Borrowed(vector));
            }

            let (variable, label) = name.split_once(':')?;
            let storage = match label.strip_prefix('=') {
                Some(value) => self
                    .continuous_keys(variable, &[value.parse().ok()?])
                    .ok()?
                    .pop()?,
                None => self.key(variable, label).ok()?,
            };
            Some(Cow::Owned(Vector::from_normalized(
                vsa.clone(),
                size,
                storage,
            )))
        })?;
        Ok(resolved.into_owned())
    }

    /// Encode records into their dense representation, writing one row of `size` elements per record into `out`.
    ///
    /// Records without any contributing cell are written as zeros. With `normalize`, the rows are normalized like
    /// vectors of the architecture (sign for bipolar architectures, L2 for real-valued ones).
    pub fn encode_into<T: DenseElement>(
        &self,
        columns: &[Column<'_>],
        out: &mut [T],
        normalize: bool,
    ) -> Result<(), EncodingError>
    where
        V: Send + Sync,
        V::Storage: Send + Sync,
    {
        if T::BIPOLAR_ONLY && (V::REAL_VALUED || !normalize) {
            return Err(EncodingError::InvalidOutput(
                "integer output requires normalized vectors of a bipolar architecture (BSC, MAP)",
            ));
        }
        let rows = self.validate(columns)?;
        if out.len() != rows * self.spec.size {
            return Err(EncodingError::RowCount {
                expected: rows,
                found: out.len() / self.spec.size,
            });
        }

        let encode = |scratch: &mut Scratch, row: usize, target: &mut [T]| {
            self.encode_row(columns, row, normalize, scratch);
            for (target, value) in target.iter_mut().zip(scratch.record.iter()) {
                *target = T::from_f32(*value);
            }
        };

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(self.spec.size)
                .enumerate()
                .for_each_init(
                    || Scratch::new(self.spec.size, self.modules.len()),
                    |scratch, (row, target)| encode(scratch, row, target),
                );
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut scratch = Scratch::new(self.spec.size, self.modules.len());
            for (row, target) in out.chunks_mut(self.spec.size).enumerate() {
                encode(&mut scratch, row, target);
            }
        }
        Ok(())
    }

    /// Encode a single record into a normalized vector. Returns `None` if no cell contributes.
    pub fn encode_record(
        &self,
        columns: &[Column<'_>],
        row: usize,
    ) -> Result<Option<Vector<Dynamic, V, Normalized<V>>>, EncodingError> {
        let rows = self.validate(columns)?;
        if row >= rows {
            return Err(EncodingError::RowCount {
                expected: row + 1,
                found: rows,
            });
        }
        let mut scratch = Scratch::new(self.spec.size, self.modules.len());
        if !self.encode_row(columns, row, false, &mut scratch) {
            return Ok(None);
        }
        let vsa = self.codebook.architecture();
        Ok(Some(Vector::from_normalized(
            vsa.clone(),
            self.codebook.size(),
            vsa.normalize_dense(&scratch.record),
        )))
    }

    /// Validate the columns, returning the number of records.
    fn validate(&self, columns: &[Column<'_>]) -> Result<usize, EncodingError> {
        if columns.len() != self.variables.len() {
            return Err(EncodingError::ColumnCount {
                expected: self.variables.len(),
                found: columns.len(),
            });
        }

        let rows = columns.first().map(Column::len).unwrap_or(0);
        for (column, (variable, spec)) in columns
            .iter()
            .zip(self.variables.iter().zip(self.spec.variables.iter()))
        {
            if column.len() != rows {
                return Err(EncodingError::RowCount {
                    expected: rows,
                    found: column.len(),
                });
            }
            if let Column::Continuous { values, .. } = column {
                if variable.continuous.is_none() {
                    return Err(EncodingError::NotContinuous(spec.name.clone()));
                }
                if values.len() != rows {
                    return Err(EncodingError::RowCount {
                        expected: rows,
                        found: values.len(),
                    });
                }
            }
            let num_keys = variable.keys.len();
            if let Some(code) = column
                .codes()
                .iter()
                .find(|code| **code != OMITTED && **code as usize >= num_keys)
            {
                return Err(EncodingError::CodeOutOfRange {
                    variable: spec.name.clone(),
                    code: *code,
                });
            }
        }
        Ok(rows)
    }

    /// Encode a validated record into `scratch.record`, returning whether any cell contributed.
    fn encode_row(
        &self,
        columns: &[Column<'_>],
        row: usize,
        normalize: bool,
        scratch: &mut Scratch,
    ) -> bool {
        let normalize_modules =
            self.spec.structure == Structure::Hierarchical && self.spec.normalize_modules;
        let vsa = self.codebook.architecture();
        scratch.record.fill(0.0);
        let mut contributed = false;

        for (column, variable) in columns.iter().zip(self.variables.iter()) {
            let code = column.codes()[row];
            let continuous_term;
            let term = if code != OMITTED {
                &variable.keys[code as usize]
            } else if let (Column::Continuous { values, .. }, Some(continuous)) =
                (column, &variable.continuous)
            {
                let value = values[row];
                if value.is_nan() {
                    continue;
                }
                let mut key = variable.base.clone();
                V::bind(&mut key, &vsa.normalize_dense(&continuous.dense(value)));
                continuous_term = key;
                &continuous_term
            } else {
                continue;
            };

            contributed = true;
            match (normalize_modules, variable.module) {
                (true, Some(module)) => {
                    scratch.touched[module] = true;
                    V::accumulate_dense(term, &mut scratch.modules[module], 1.0);
                }
                _ => V::accumulate_dense(term, &mut scratch.record, 1.0),
            }
        }

        if normalize_modules {
            for (module, vector) in self.modules.iter().enumerate() {
                if !std::mem::take(&mut scratch.touched[module]) {
                    continue;
                }
                let mut bundle = vsa.normalize_dense(&scratch.modules[module]);
                V::bind(&mut bundle, vector);
                V::accumulate_dense(&bundle, &mut scratch.record, 1.0);
                scratch.modules[module].fill(0.0);
            }
        }

        if normalize && contributed {
            let normalized = vsa.normalize_dense(&scratch.record);
            scratch.record.fill(0.0);
            V::accumulate_dense(&normalized, &mut scratch.record, 1.0);
        }
        contributed
    }

    /// Dense random vectors derived from `<name>#<index>`, e.g. to derive the null distribution of similarities.
    pub fn random_dense(&self, name: &str, count: usize) -> Vec<f32> {
        let vsa = self.codebook.architecture();
        let mut out = vec![0.0; count * self.spec.size];
        for (index, row) in out.chunks_mut(self.spec.size).enumerate() {
            V::accumulate_dense(
                &vsa.random_named(self.spec.size, &format!("{name}#{index}")),
                row,
                1.0,
            );
        }
        out
    }
}

struct Scratch {
    record: Vec<f32>,
    modules: Vec<Vec<f32>>,
    touched: Vec<bool>,
}

impl Scratch {
    fn new(size: usize, modules: usize) -> Self {
        Self {
            record: vec![0.0; size],
            modules: vec![vec![0.0; size]; modules],
            touched: vec![false; modules],
        }
    }
}

#[cfg(test)]
mod tests;
