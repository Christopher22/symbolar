use std::num::NonZero;

use ::symbolar::{
    Dynamic, Expression, Normalized, NotNormalized, SamplesWithColumnVector, SamplesWithPosition,
    Size, Storage, Subset, Vector,
    architectures::{
        BinarySpatterCode, HolographicReducedRepresentation, MultiplyAddPermute,
        VectorDerivedTransformationBinding, VectorSymbolicArchitecture,
    },
    encoding::{Column, DenseElement, Encoder, EncoderSpec, OMITTED},
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, ndarray::Array2};
use polars::prelude::{ChunkFillNullValue, DataFrame, DataType, Float64Chunked, UInt32Chunked};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::types::PyDataFrame;

fn to_dynamic_size(size: usize) -> PyResult<Dynamic> {
    NonZero::new(size)
        .map(Dynamic::from)
        .ok_or_else(|| PyValueError::new_err("size must be > 0"))
}

/// An error for an invalid size, suggesting valid alternatives for architectures restricted to perfect squares.
fn invalid_size<V: VectorSymbolicArchitecture>(size: usize) -> PyErr {
    if size > 0 && !V::valid_size(Dynamic::from(NonZero::new(size).expect("non-zero"))) {
        let root = (size as f64).sqrt().floor() as usize;
        return PyValueError::new_err(format!(
            "invalid vector size {size} for architecture: the size must be a perfect square, e.g. {} or {}",
            root * root,
            (root + 1) * (root + 1)
        ));
    }
    PyValueError::new_err("invalid vector size for architecture")
}

/// The codes and optional values of all variables, read from a DataFrame with the columns `<variable>` (codes) and
/// `<variable>#value` (scaled continuous values).
struct EncodingColumns {
    codes: Vec<UInt32Chunked>,
    values: Vec<Option<Float64Chunked>>,
}

impl EncodingColumns {
    fn from_dataframe(spec: &EncoderSpec, dataframe: &DataFrame) -> PyResult<Self> {
        let mut codes = Vec::with_capacity(spec.variables.len());
        let mut values = Vec::with_capacity(spec.variables.len());
        for variable in &spec.variables {
            let column = dataframe.column(&variable.name).map_err(map_error)?;
            let column = column
                .as_materialized_series()
                .cast(&DataType::UInt32)
                .map_err(map_error)?;
            let column = column.u32().map_err(map_error)?;
            codes.push(
                column
                    .fill_null_with_values(OMITTED)
                    .map_err(map_error)?
                    .rechunk()
                    .into_owned(),
            );

            values.push(
                match dataframe.column(&format!("{}#value", variable.name)) {
                    Ok(column) => {
                        let column = column
                            .as_materialized_series()
                            .cast(&DataType::Float64)
                            .map_err(map_error)?;
                        Some(
                            column
                                .f64()
                                .map_err(map_error)?
                                .fill_null_with_values(f64::NAN)
                                .map_err(map_error)?
                                .rechunk()
                                .into_owned(),
                        )
                    }
                    Err(_) => None,
                },
            );
        }
        Ok(Self { codes, values })
    }

    fn columns(&self) -> PyResult<Vec<Column<'_>>> {
        self.codes
            .iter()
            .zip(self.values.iter())
            .map(|(codes, values)| {
                let codes = codes.cont_slice().map_err(map_error)?;
                Ok(match values {
                    Some(values) => Column::Continuous {
                        codes,
                        values: values.cont_slice().map_err(map_error)?,
                    },
                    None => Column::Codes(codes),
                })
            })
            .collect()
    }

    fn rows(&self) -> usize {
        self.codes.first().map(|codes| codes.len()).unwrap_or(0)
    }
}

/// Encode all records into a new NumPy array, releasing the GIL meanwhile.
fn encode_to_numpy<'py, V, T>(
    py: Python<'py>,
    encoder: &Encoder<V>,
    dataframe: PyDataFrame,
    normalize: bool,
) -> PyResult<Bound<'py, PyArray2<T>>>
where
    V: VectorSymbolicArchitecture + Send + Sync,
    V::Storage: Send + Sync,
    T: DenseElement + numpy::Element + Default,
{
    let dataframe: DataFrame = dataframe.into();
    let columns = EncodingColumns::from_dataframe(encoder.spec(), &dataframe)?;
    let output = PyArray2::<T>::zeros(py, [columns.rows(), encoder.size()], false);
    {
        let mut guard = output.readwrite();
        let out = guard.as_slice_mut().map_err(map_error)?;
        let columns = columns.columns()?;
        py.detach(|| encoder.encode_into(&columns, out, normalize))
            .map_err(map_error)?;
    }
    Ok(output)
}

/// Stack dense vectors into a two-dimensional NumPy array.
fn dense_matrix<'py, V: VectorSymbolicArchitecture>(
    py: Python<'py>,
    encoder: &Encoder<V>,
    vectors: &[V::Storage],
) -> Bound<'py, PyArray2<f32>> {
    let size = encoder.size();
    let mut flat = Vec::with_capacity(vectors.len() * size);
    for vector in vectors {
        flat.extend(encoder.to_dense(vector));
    }
    Array2::from_shape_vec((vectors.len(), size), flat)
        .expect("consistent shape")
        .into_pyarray(py)
}

fn map_error<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyclass(name = "VSA", subclass)]
#[derive(Clone, Debug)]
struct PyVsa {
    architecture: String,
}

impl PyVsa {
    fn new(architecture: &'static str) -> Self {
        Self {
            architecture: architecture.to_string(),
        }
    }
}

#[pymethods]
impl PyVsa {
    fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Abstract method — concrete architecture subclasses must override this.
    fn create_storage(&self, _py: Python<'_>, _num_elements: usize) -> PyResult<Py<PyAny>> {
        Err(PyNotImplementedError::new_err(
            "create_storage must be implemented by a concrete architecture subclass",
        ))
    }

    fn __repr__(&self) -> String {
        format!("VSA(architecture={})", self.architecture)
    }
}

#[pyclass(name = "Vector", subclass)]
#[derive(Clone, Debug)]
struct PyVector {
    architecture: String,
    size: usize,
}

impl PyVector {
    fn new(architecture: &'static str, size: usize) -> Self {
        Self {
            architecture: architecture.to_string(),
            size,
        }
    }
}

#[pymethods]
impl PyVector {
    fn architecture(&self) -> &str {
        &self.architecture
    }

    fn size(&self) -> usize {
        self.size
    }

    fn __repr__(&self) -> String {
        format!(
            "Vector(architecture={}, size={})",
            self.architecture, self.size
        )
    }
}

#[pyclass(name = "Storage", subclass)]
#[derive(Clone, Debug)]
struct PyStorage {
    architecture: String,
    size: usize,
}

impl PyStorage {
    fn new(architecture: &'static str, size: usize) -> Self {
        Self {
            architecture: architecture.to_string(),
            size,
        }
    }
}

#[pymethods]
impl PyStorage {
    fn architecture(&self) -> &str {
        &self.architecture
    }

    fn size(&self) -> usize {
        self.size
    }

    fn __repr__(&self) -> String {
        format!(
            "Storage(architecture={}, size={})",
            self.architecture, self.size
        )
    }
}

macro_rules! define_architecture_bindings {
    (
        architecture_py = $architecture_py:ident,
        architecture_name = $architecture_name:literal,
        architecture_inner = $architecture_inner:ty,
        vector_py = $vector_py:ident,
        vector_name = $vector_name:literal,
        vector_inner = $vector_inner:ty,
        vector_unnormalized_py = $vector_unnormalized_py:ident,
        vector_unnormalized_name = $vector_unnormalized_name:literal,
        vector_unnormalized_inner = $vector_unnormalized_inner:ty,
        inverse_methods = { $($inverse_methods:tt)* },
        storage_py = $storage_py:ident,
        storage_name = $storage_name:literal,
        storage_inner = $storage_inner:ty,
        subset_py = $subset_py:ident,
        subset_name = $subset_name:literal,
        encoder_py = $encoder_py:ident,
        encoder_name = $encoder_name:literal,
    ) => {
        #[pyclass(name = $architecture_name, extends = PyVsa)]
        #[derive(Clone, Debug)]
        struct $architecture_py {
            inner: $architecture_inner,
        }

        #[pymethods]
        impl $architecture_py {
            #[new]
            fn new(seed: u64) -> (Self, PyVsa) {
                (
                    Self {
                        inner: <$architecture_inner>::new(seed),
                    },
                    PyVsa::new($architecture_name),
                )
            }

            fn random_vector(&self, py: Python<'_>, size: usize) -> PyResult<Py<$vector_py>> {
                let size = to_dynamic_size(size)?;
                let inner = Vector::random(&self.inner, size)
                    .ok_or_else(|| PyValueError::new_err("invalid vector size for architecture"))?;
                $vector_py::from_inner(py, inner)
            }

            fn create_storage(
                &self,
                py: Python<'_>,
                num_elements: usize,
            ) -> PyResult<Py<$storage_py>> {
                let inner = <$storage_inner>::new(self.inner.clone(), to_dynamic_size(num_elements)?)
                    .ok_or_else(|| invalid_size::<$architecture_inner>(num_elements))?;
                $storage_py::from_inner(py, num_elements, inner)
            }

            /// The seed of the architecture, from which all named vectors are derived.
            fn seed(&self) -> u64 {
                self.inner.seed()
            }

            /// Create an encoder from its JSON specification.
            fn create_encoder(&self, spec: &str) -> PyResult<$encoder_py> {
                let spec: EncoderSpec = serde_json::from_str(spec).map_err(map_error)?;
                if !<$architecture_inner>::valid_size(to_dynamic_size(spec.size)?) {
                    return Err(invalid_size::<$architecture_inner>(spec.size));
                }
                Ok($encoder_py {
                    inner: Encoder::new(self.inner.clone(), spec).map_err(map_error)?,
                })
            }

            /// Normalize a dense (bipolar or real-valued) vector, e.g. a row of an encoded matrix, into a vector.
            fn vector_from_numpy(
                &self,
                py: Python<'_>,
                dense: PyReadonlyArray1<'_, f32>,
            ) -> PyResult<Py<$vector_py>> {
                let dense = dense.as_slice().map_err(map_error)?;
                let size = to_dynamic_size(dense.len())?;
                if !<$architecture_inner>::valid_size(size) {
                    return Err(invalid_size::<$architecture_inner>(dense.len()));
                }
                $vector_py::from_inner(
                    py,
                    Vector::from_normalized(self.inner.clone(), size, self.inner.normalize_dense(dense)),
                )
            }
        }

        /// Encodes records into hypervectors. Created by `create_encoder` of an architecture.
        #[pyclass(name = $encoder_name)]
        #[derive(Clone, Debug)]
        struct $encoder_py {
            inner: Encoder<$architecture_inner>,
        }

        #[pymethods]
        impl $encoder_py {
            /// The dimensionality of the vectors.
            fn size(&self) -> usize {
                self.inner.size()
            }

            /// The names of all vectors in the codebook.
            fn symbols(&self) -> Vec<String> {
                self.inner.codebook().values().map(|value| value.to_string()).collect()
            }

            /// A stable fingerprint of the codebook as a hexadecimal string.
            fn digest(&self) -> String {
                format!("{:016x}", self.inner.codebook().digest())
            }

            /// Encode the records given as codes (`<variable>`) and scaled values (`<variable>#value`).
            #[pyo3(signature = (codes, normalize = true, dtype = "float32"))]
            fn encode<'py>(
                &self,
                py: Python<'py>,
                codes: PyDataFrame,
                normalize: bool,
                dtype: &str,
            ) -> PyResult<Bound<'py, PyAny>> {
                match dtype {
                    "float32" => Ok(encode_to_numpy::<_, f32>(py, &self.inner, codes, normalize)?.into_any()),
                    "int8" => Ok(encode_to_numpy::<_, i8>(py, &self.inner, codes, normalize)?.into_any()),
                    _ => Err(PyValueError::new_err("dtype must be 'float32' or 'int8'")),
                }
            }

            /// The labels of the keys of a variable, in the order of their codes.
            fn labels(&self, variable: &str) -> PyResult<Vec<String>> {
                Ok(self.inner.labels(variable).map_err(map_error)?.to_vec())
            }

            /// The dense full keys `M ⊛ R ⊛ Φ(k)` of a variable, one row per code.
            fn keys<'py>(&self, py: Python<'py>, variable: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
                let keys = self.inner.keys(variable).map_err(map_error)?;
                Ok(dense_matrix(py, &self.inner, &keys))
            }

            /// The dense full keys `M ⊛ R ⊛ B^x` of a continuous variable for the given (scaled) values.
            fn continuous_keys<'py>(
                &self,
                py: Python<'py>,
                variable: &str,
                values: Vec<f64>,
            ) -> PyResult<Bound<'py, PyArray2<f32>>> {
                let keys = self.inner.continuous_keys(variable, &values).map_err(map_error)?;
                Ok(dense_matrix(py, &self.inner, &keys))
            }

            /// Evaluate an expression into its dense representation.
            /// Resolves codebook names, `<variable>:<label>` and `<variable>:=<scaled value>`.
            fn evaluate<'py>(&self, py: Python<'py>, expression: &str) -> PyResult<Bound<'py, PyArray1<f32>>> {
                let expression: Expression = expression.parse().map_err(map_error)?;
                let vector = self.inner.evaluate(&expression).map_err(map_error)?;
                Ok(self.inner.to_dense(vector.as_storage()).into_pyarray(py))
            }

            /// Evaluate an expression into a vector.
            fn evaluate_vector(&self, py: Python<'_>, expression: &str) -> PyResult<Py<$vector_py>> {
                let expression: Expression = expression.parse().map_err(map_error)?;
                let vector = self.inner.evaluate(&expression).map_err(map_error)?;
                $vector_py::from_inner(py, vector)
            }

            /// Get a named vector of the codebook.
            fn get(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<$vector_py>>> {
                match self.inner.codebook().get(&name).cloned() {
                    Some(inner) => Ok(Some($vector_py::from_inner(py, inner)?)),
                    None => Ok(None),
                }
            }

            /// Dense random vectors derived from `<name>#<index>`.
            fn random<'py>(&self, py: Python<'py>, name: &str, count: usize) -> Bound<'py, PyArray2<f32>> {
                let flat = self.inner.random_dense(name, count);
                Array2::from_shape_vec((count, self.inner.size()), flat)
                    .expect("consistent shape")
                    .into_pyarray(py)
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(size={}, variables={})",
                    $encoder_name,
                    self.inner.size(),
                    self.inner.spec().variables.len()
                )
            }
        }

        #[pyclass(name = $vector_name, extends = PyVector)]
        #[derive(Clone, Debug)]
        struct $vector_py {
            inner: $vector_inner,
        }

        impl $vector_py {
            fn from_inner(py: Python<'_>, inner: $vector_inner) -> PyResult<Py<Self>> {
                let size = inner.size.size();
                Py::new(
                    py,
                    (Self { inner }, PyVector::new($architecture_name, size)),
                )
            }
        }

        #[pymethods]
        impl $vector_py {
            fn similarity(&self, other: &$vector_py) -> f64 {
                self.inner.similarity(&other.inner)
            }

            fn permute(&self, py: Python<'_>, shifts: usize) -> PyResult<Py<$vector_py>> {
                $vector_py::from_inner(py, self.inner.clone().permute(shifts))
            }

            fn bind(&self, py: Python<'_>, other: &$vector_py) -> PyResult<Py<$vector_py>> {
                $vector_py::from_inner(py, &self.inner * &other.inner)
            }

            fn bundle(&self, py: Python<'_>, other: &$vector_py) -> PyResult<Py<$vector_py>> {
                $vector_py::from_inner(py, (&self.inner + &other.inner).normalize())
            }

            fn bundle_unnormalized(
                &self,
                py: Python<'_>,
                other: &$vector_py,
            ) -> PyResult<Py<$vector_unnormalized_py>> {
                $vector_unnormalized_py::from_inner(py, &self.inner + &other.inner)
            }

            fn equals(&self, other: &$vector_py) -> bool {
                self.inner == other.inner
            }

            $($inverse_methods)*

            fn __mul__(&self, py: Python<'_>, other: &$vector_py) -> PyResult<Py<$vector_py>> {
                self.bind(py, other)
            }

            fn __add__(&self, py: Python<'_>, other: &$vector_py) -> PyResult<Py<$vector_py>> {
                self.bundle(py, other)
            }

            fn __repr__(&self) -> String {
                format!("{}(size={})", $vector_name, self.inner.size.size())
            }

            fn to_list(&self) -> Vec<f64> {
                self.inner.serialize()
            }
        }

        #[pyclass(name = $vector_unnormalized_name, extends = PyVector)]
        #[derive(Clone, Debug)]
        struct $vector_unnormalized_py {
            inner: $vector_unnormalized_inner,
        }

        impl $vector_unnormalized_py {
            fn from_inner(py: Python<'_>, inner: $vector_unnormalized_inner) -> PyResult<Py<Self>> {
                let size = inner.size.size();
                Py::new(
                    py,
                    (Self { inner }, PyVector::new($architecture_name, size)),
                )
            }
        }

        #[pymethods]
        impl $vector_unnormalized_py {
            fn normalize(&self, py: Python<'_>) -> PyResult<Py<$vector_py>> {
                $vector_py::from_inner(py, self.inner.clone().normalize())
            }

            /// Similarity computed directly on the un-normalized (pre-thresholding) data.
            fn similarity(&self, other: &$vector_unnormalized_py) -> f64 {
                self.inner.similarity(&other.inner)
            }

            /// Find the key of the highest-similarity candidate.
            fn best_match(
                &self,
                py: Python<'_>,
                candidates: std::collections::HashMap<String, Py<$vector_unnormalized_py>>,
            ) -> Option<String> {
                let mut best: Option<String> = None;
                let mut best_sim = f64::NEG_INFINITY;
                for (key, candidate) in &candidates {
                    let candidate = candidate.borrow(py);
                    let sim = self.inner.similarity(&candidate.inner);
                    if sim > best_sim {
                        best_sim = sim;
                        best = Some(key.clone());
                    }
                }
                best
            }

            fn bind(
                &self,
                py: Python<'_>,
                other: &$vector_py,
            ) -> PyResult<Py<$vector_unnormalized_py>> {
                $vector_unnormalized_py::from_inner(
                    py,
                    self.inner.clone().bind(&other.inner),
                )
            }

            fn bundle(&self, py: Python<'_>, other: &$vector_py) -> PyResult<Py<$vector_unnormalized_py>> {
                $vector_unnormalized_py::from_inner(py, &self.inner + &other.inner)
            }

            fn equals(&self, other: &$vector_unnormalized_py) -> bool {
                self.inner == other.inner
            }

            fn __add__(&self, py: Python<'_>, other: &$vector_py) -> PyResult<Py<$vector_unnormalized_py>> {
                self.bundle(py, other)
            }

            fn __mul__(
                &self,
                py: Python<'_>,
                other: &$vector_py,
            ) -> PyResult<Py<$vector_unnormalized_py>> {
                self.bind(py, other)
            }

            fn __repr__(&self) -> String {
                format!("{}(size={})", $vector_unnormalized_name, self.inner.size.size())
            }

            fn to_list(&self) -> Vec<f64> {
                self.inner.serialize()
            }
        }

        #[pyclass(name = $storage_name, extends = PyStorage)]
        #[derive(Clone, Debug)]
        struct $storage_py {
            inner: $storage_inner,
            size: usize,
        }

        impl $storage_py {
            fn from_inner(
                py: Python<'_>,
                size: usize,
                inner: $storage_inner,
            ) -> PyResult<Py<Self>> {
                Py::new(
                    py,
                    (
                        Self { inner, size },
                        PyStorage::new($architecture_name, size),
                    ),
                )
            }
        }

        #[pymethods]
        impl $storage_py {
            #[new]
            fn new(architecture: &$architecture_py, size: usize) -> PyResult<(Self, PyStorage)> {
                let inner = <$storage_inner>::new(architecture.inner.clone(), to_dynamic_size(size)?)
                    .ok_or_else(|| invalid_size::<$architecture_inner>(size))?;
                Ok((
                    Self { inner, size },
                    PyStorage::new($architecture_name, size),
                ))
            }

            #[staticmethod]
            fn from_dataframe(
                py: Python<'_>,
                architecture: &$architecture_py,
                size: usize,
                dataframe: PyDataFrame,
            ) -> PyResult<Py<$storage_py>> {
                let dataframe: DataFrame = dataframe.into();
                let inner = <$storage_inner>::from_dataframe(
                    architecture.inner.clone(),
                    to_dynamic_size(size)?,
                    &dataframe,
                )
                .map_err(map_error)?;

                $storage_py::from_inner(py, size, inner)
            }

            fn push(&mut self, name: String) {
                self.inner.push(name);
            }

            fn extend(&mut self, names: Vec<String>) {
                self.inner.extend(names);
            }

            fn columns(&self) -> Vec<String> {
                self.inner
                    .columns()
                    .map(|value| value.to_string())
                    .collect()
            }

            fn values(&self) -> Vec<String> {
                self.inner.values().map(|value| value.to_string()).collect()
            }

            fn get(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<$vector_py>>> {
                match self.inner.get(&name).cloned() {
                    Some(inner) => Ok(Some($vector_py::from_inner(py, inner)?)),
                    None => Ok(None),
                }
            }

            fn get_column_value(
                &self,
                py: Python<'_>,
                column: &str,
                value: &str,
            ) -> PyResult<Option<Py<$vector_py>>> {
                match self
                    .inner
                    .get(&(
                        ::symbolar::Column::from_str(column),
                        ::symbolar::Value::from_str(value),
                    ))
                    .cloned()
                {
                    Some(inner) => Ok(Some($vector_py::from_inner(py, inner)?)),
                    None => Ok(None),
                }
            }

            fn execute(&self, py: Python<'_>, expression: &str) -> PyResult<Py<$vector_py>> {
                let expression: Expression = expression.parse().map_err(map_error)?;
                let result = self.inner.execute(&expression).map_err(map_error)?;
                $vector_py::from_inner(py, result.into_owned())
            }

            fn find(
                &self,
                py: Python<'_>,
                vector: &$vector_py,
            ) -> PyResult<Option<Py<$vector_py>>> {
                match self
                    .inner
                    .find(&vector.inner, &())
                    .and_then(|index| self.inner.get(&index))
                    .cloned()
                {
                    Some(inner) => Ok(Some($vector_py::from_inner(py, inner)?)),
                    None => Ok(None),
                }
            }

            fn subset(&self, dataframe: PyDataFrame) -> $subset_py {
                $subset_py {
                    storage: self.inner.clone(),
                    dataframe: dataframe.into(),
                }
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(size={}, columns={}, values={})",
                    $storage_name,
                    self.size,
                    self.inner.columns().count(),
                    self.inner.values().count()
                )
            }
        }

        #[pyclass(name = $subset_name)]
        #[derive(Clone, Debug)]
        struct $subset_py {
            storage: $storage_inner,
            dataframe: DataFrame,
        }

        impl $subset_py {
            fn build_subset(&self) -> Result<Subset<'_, Dynamic, $architecture_inner>, PyErr> {
                self.storage.subset(&self.dataframe).map_err(map_error)
            }
        }

        #[pymethods]
        impl $subset_py {
            fn bundle_rows(&self, py: Python<'_>) -> PyResult<Vec<Option<Py<$vector_py>>>> {
                let subset = self.build_subset()?;
                subset
                    .bundle_rows::<Normalized<$architecture_inner>, SamplesWithColumnVector>()
                    .map(|item| match item {
                        Some(inner) => Ok(Some($vector_py::from_inner(py, inner)?)),
                        None => Ok(None),
                    })
                    .collect()
            }

            fn bundle_rows_unnormalized(&self, py: Python<'_>) -> PyResult<Vec<Option<Py<$vector_unnormalized_py>>>> {
                let subset = self.build_subset()?;
                subset
                    .bundle_rows::<NotNormalized<$architecture_inner>, SamplesWithColumnVector>()
                    .map(|item| match item {
                        Some(inner) => Ok(Some($vector_unnormalized_py::from_inner(py, inner)?)),
                        None => Ok(None),
                    })
                    .collect()
            }

            fn bundle_dataset(&self, py: Python<'_>) -> PyResult<Option<Py<$vector_py>>> {
                let subset = self.build_subset()?;
                match subset.bundle_dataset::<
                    Normalized<$architecture_inner>,
                    Normalized<$architecture_inner>,
                    SamplesWithColumnVector
                >() {
                    Some(inner) => Ok(Some($vector_py::from_inner(py, inner)?)),
                    None => Ok(None),
                }
            }

            fn bundle_dataset_unnormalized(&self, py: Python<'_>) -> PyResult<Option<Py<$vector_unnormalized_py>>> {
                let subset = self.build_subset()?;
                match subset.bundle_dataset::<
                    NotNormalized<$architecture_inner>,
                    NotNormalized<$architecture_inner>,
                    SamplesWithColumnVector
                >() {
                    Some(inner) => Ok(Some($vector_unnormalized_py::from_inner(py, inner)?)),
                    None => Ok(None),
                }
            }

            fn bundle_dataset_combinatorial(&self, py: Python<'_>, ways: usize) -> PyResult<Option<Py<$vector_unnormalized_py>>> {
                let subset = self.build_subset()?;
                match subset.bundle_dataset_combinatorial(ways) {
                    Some(inner) => Ok(Some($vector_unnormalized_py::from_inner(py, inner)?)),
                    None => Ok(None),
                }
            }

            fn bundle_dataset_temporal(&self, py: Python<'_>) -> PyResult<Option<Py<$vector_py>>> {
                let subset = self.build_subset()?;
                match subset.bundle_dataset_with_binding::<Normalized<$architecture_inner>, SamplesWithPosition>() {
                    Some(inner) => Ok(Some($vector_py::from_inner(py, inner)?)),
                    None => Ok(None),
                }
            }

            fn bundle_dataset_temporal_unnormalized(&self, py: Python<'_>) -> PyResult<Option<Py<$vector_unnormalized_py>>> {
                let subset = self.build_subset()?;
                match subset.bundle_dataset_with_binding::<NotNormalized<$architecture_inner>, SamplesWithPosition>() {
                    Some(inner) => Ok(Some($vector_unnormalized_py::from_inner(py, inner)?)),
                    None => Ok(None),
                }
            }

            /// Bundle rows grouped by an externally supplied per-row key (e.g. a sample or class id)
            fn bundle_dataset_temporal_grouped(
                &self,
                py: Python<'_>,
                groups: Vec<i64>,
            ) -> PyResult<std::collections::HashMap<i64, Py<$vector_unnormalized_py>>> {
                let subset = self.build_subset()?;
                let grouped = subset
                    .bundle_groups_temporal::<i64, NotNormalized<$architecture_inner>>(&groups)
                    .map_err(map_error)?;
                grouped
                    .into_iter()
                    .map(|(k, v)| Ok((k, $vector_unnormalized_py::from_inner(py, v)?)))
                    .collect()
            }

            fn bind_dataset(&self, py: Python<'_>) -> PyResult<Option<Py<$vector_py>>> {
                let subset = self.build_subset()?;
                match subset.bind_dataset::<SamplesWithColumnVector>() {
                    Some(inner) => Ok(Some($vector_py::from_inner(py, inner)?)),
                    None => Ok(None),
                }
            }
        }
    };
}

type BscArchitecture = BinarySpatterCode<usize>;
type HrrArchitecture = HolographicReducedRepresentation<f64, rand::rngs::StdRng>;
type MapArchitecture = MultiplyAddPermute<usize>;
type VtbArchitecture = VectorDerivedTransformationBinding<f64, rand::rngs::StdRng>;
type BscVector = Vector<Dynamic, BscArchitecture, Normalized<BscArchitecture>>;
type HrrVector = Vector<Dynamic, HrrArchitecture, Normalized<HrrArchitecture>>;
type MapVector = Vector<Dynamic, MapArchitecture, Normalized<MapArchitecture>>;
type VtbVector = Vector<Dynamic, VtbArchitecture, Normalized<VtbArchitecture>>;
type BscUnnormalizedVector = Vector<Dynamic, BscArchitecture, NotNormalized<BscArchitecture>>;
type HrrUnnormalizedVector = Vector<Dynamic, HrrArchitecture, NotNormalized<HrrArchitecture>>;
type MapUnnormalizedVector = Vector<Dynamic, MapArchitecture, NotNormalized<MapArchitecture>>;
type VtbUnnormalizedVector = Vector<Dynamic, VtbArchitecture, NotNormalized<VtbArchitecture>>;
type BscStorage = Storage<Dynamic, BscArchitecture>;
type HrrStorage = Storage<Dynamic, HrrArchitecture>;
type MapStorage = Storage<Dynamic, MapArchitecture>;
type VtbStorage = Storage<Dynamic, VtbArchitecture>;

define_architecture_bindings!(
    architecture_py = PyBinarySpatterCode,
    architecture_name = "BinarySpatterCode",
    architecture_inner = BscArchitecture,
    vector_py = PyBscVector,
    vector_name = "BscVector",
    vector_inner = BscVector,
    vector_unnormalized_py = PyBscUnnormalizedVector,
    vector_unnormalized_name = "BscUnnormalizedVector",
    vector_unnormalized_inner = BscUnnormalizedVector,
    inverse_methods = {},
    storage_py = PyBscStorage,
    storage_name = "BscStorage",
    storage_inner = BscStorage,
    subset_py = PyBscSubset,
    subset_name = "BscSubset",
    encoder_py = PyBscEncoder,
    encoder_name = "BscEncoder",
);

define_architecture_bindings!(
    architecture_py = PyMultiplyAddPermute,
    architecture_name = "MultiplyAddPermute",
    architecture_inner = MapArchitecture,
    vector_py = PyMapVector,
    vector_name = "MapVector",
    vector_inner = MapVector,
    vector_unnormalized_py = PyMapUnnormalizedVector,
    vector_unnormalized_name = "MapUnnormalizedVector",
    vector_unnormalized_inner = MapUnnormalizedVector,
    inverse_methods = {},
    storage_py = PyMapStorage,
    storage_name = "MapStorage",
    storage_inner = MapStorage,
    subset_py = PyMapSubset,
    subset_name = "MapSubset",
    encoder_py = PyMapEncoder,
    encoder_name = "MapEncoder",
);

define_architecture_bindings!(
    architecture_py = PyHolographicReducedRepresentation,
    architecture_name = "HolographicReducedRepresentation",
    architecture_inner = HrrArchitecture,
    vector_py = PyHrrVector,
    vector_name = "HrrVector",
    vector_inner = HrrVector,
    vector_unnormalized_py = PyHrrUnnormalizedVector,
    vector_unnormalized_name = "HrrUnnormalizedVector",
    vector_unnormalized_inner = HrrUnnormalizedVector,
    inverse_methods = {
        fn inverse(&self, py: Python<'_>) -> PyResult<Py<PyHrrVector>> {
            PyHrrVector::from_inner(py, -&self.inner)
        }

        fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyHrrVector>> {
            self.inverse(py)
        }
    },
    storage_py = PyHrrStorage,
    storage_name = "HrrStorage",
    storage_inner = HrrStorage,
    subset_py = PyHrrSubset,
    subset_name = "HrrSubset",
    encoder_py = PyHrrEncoder,
    encoder_name = "HrrEncoder",
);

define_architecture_bindings!(
    architecture_py = PyVectorDerivedTransformationBinding,
    architecture_name = "VectorDerivedTransformationBinding",
    architecture_inner = VtbArchitecture,
    vector_py = PyVtbVector,
    vector_name = "VtbVector",
    vector_inner = VtbVector,
    vector_unnormalized_py = PyVtbUnnormalizedVector,
    vector_unnormalized_name = "VtbUnnormalizedVector",
    vector_unnormalized_inner = VtbUnnormalizedVector,
    inverse_methods = {
        fn inverse(&self, py: Python<'_>) -> PyResult<Py<PyVtbVector>> {
            PyVtbVector::from_inner(py, -&self.inner)
        }

        fn __neg__(&self, py: Python<'_>) -> PyResult<Py<PyVtbVector>> {
            self.inverse(py)
        }
    },
    storage_py = PyVtbStorage,
    storage_name = "VtbStorage",
    storage_inner = VtbStorage,
    subset_py = PyVtbSubset,
    subset_name = "VtbSubset",
    encoder_py = PyVtbEncoder,
    encoder_name = "VtbEncoder",
);

#[pymodule]
fn _native(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyVsa>()?;
    module.add_class::<PyVector>()?;
    module.add_class::<PyStorage>()?;

    module.add_class::<PyBinarySpatterCode>()?;
    module.add_class::<PyMultiplyAddPermute>()?;
    module.add_class::<PyHolographicReducedRepresentation>()?;
    module.add_class::<PyVectorDerivedTransformationBinding>()?;

    module.add_class::<PyBscVector>()?;
    module.add_class::<PyMapVector>()?;
    module.add_class::<PyHrrVector>()?;
    module.add_class::<PyVtbVector>()?;
    module.add_class::<PyBscUnnormalizedVector>()?;
    module.add_class::<PyMapUnnormalizedVector>()?;
    module.add_class::<PyHrrUnnormalizedVector>()?;
    module.add_class::<PyVtbUnnormalizedVector>()?;

    module.add_class::<PyBscStorage>()?;
    module.add_class::<PyMapStorage>()?;
    module.add_class::<PyHrrStorage>()?;
    module.add_class::<PyVtbStorage>()?;

    module.add_class::<PyBscSubset>()?;
    module.add_class::<PyBscEncoder>()?;
    module.add_class::<PyMapSubset>()?;
    module.add_class::<PyMapEncoder>()?;
    module.add_class::<PyHrrSubset>()?;
    module.add_class::<PyHrrEncoder>()?;
    module.add_class::<PyVtbSubset>()?;
    module.add_class::<PyVtbEncoder>()?;

    Ok(())
}
