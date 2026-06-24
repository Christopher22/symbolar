use polars::prelude::*;

use super::ColumnType;

use crate::{
    Normalized, NotNormalized, Size, Storage, Vector, VectorIndex, VectorType,
    architectures::VectorSymbolicArchitecture,
};

/// A subset of a dataframe used to derive vectors for specific rows or the entire dataset.
#[derive(Debug, Clone)]
pub struct Subset<'a, S: Size, V: VectorSymbolicArchitecture> {
    storage: &'a Storage<S, V>,
    vectors: Vec<Option<VectorIndex>>,
    columns: Vec<VectorIndex>,
    width: usize,
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Subset<'a, S, V> {
    pub(crate) fn new(storage: &'a Storage<S, V>, dataframe: &DataFrame) -> Result<Self, Error> {
        let width = dataframe.width();
        let height = dataframe.height();
        let mut vectors = Vec::with_capacity(width * height);
        let mut columns = Vec::with_capacity(width);
        for column in dataframe.columns() {
            let ref_column = storage
                .columns
                .0
                .get(column.name().as_str())
                .ok_or_else(|| Error::ColumnNotFound {
                    name: column.name().to_string(),
                })?;

            match (&ref_column.column_type, column.dtype()) {
                (ColumnType::String, DataType::String) => {
                    for value in column.str().expect("string column") {
                        vectors.push(match value.map(|v| storage.vectors.names.get(v).copied()) {
                            Some(Some(vector)) => Some(vector),
                            None => None, // Skip null values.
                            Some(None) => {
                                return Err(Error::ValueNotFound {
                                    column: column.name().to_string(),
                                    value: value.unwrap_or("").to_string(),
                                });
                            }
                        })
                    }
                }
                (
                    ColumnType::Enum {
                        categories: ref_categories,
                        values: ref_values,
                    },
                    DataType::Enum(categories, ..),
                ) if ensure_same_frozen_categories(ref_categories, categories).is_ok() => {
                    for row in 0..height {
                        vectors.push(match column.get(row)? {
                            AnyValue::Null => None,
                            AnyValue::Enum(index, _) | AnyValue::EnumOwned(index, _) => {
                                Some(ref_values[index as usize])
                            }
                            _ => {
                                return Err(Error::ColumnTypeMismatch {
                                    name: column.name().to_string(),
                                });
                            }
                        })
                    }
                }
                _ => {
                    return Err(Error::ColumnTypeMismatch {
                        name: column.name().to_string(),
                    });
                }
            }

            columns.push(ref_column.vector);
        }

        Ok(Self {
            storage,
            vectors,
            columns,
            width,
        })
    }

    /// Iterate through the rows of the subset, bundling the vectors for each row into a single vector.
    pub fn bundle_rows<T: VectorType<V>>(&self) -> impl Iterator<Item = Option<Vector<S, V, T>>> {
        RowIterator::from(self).map(|sample_iter| {
            sample_iter
                .filter_map(|x| x.map(V::denormalize))
                .fold(None, |acc, value| match acc {
                    Some(mut acc) => {
                        self.storage
                            .vectors
                            .vsa
                            .bundle_with_accumulator(&mut acc, &value);
                        Some(acc)
                    }
                    None => Some(value),
                })
                .map(|vector_unnormalized| {
                    let vsa = self.storage.vectors.vsa.clone();
                    Vector::new(vsa, self.storage.vectors.size, vector_unnormalized)
                })
        })
    }

    /// Iterate through the rows of the subset, binding the vectors for each row into a single vector.
    pub fn bind_rows(&self) -> impl Iterator<Item = Option<Vector<S, V, Normalized<V>>>> {
        RowIterator::from(self).map(|sample_iter| {
            sample_iter
                .flatten()
                .fold(None, |acc: Option<V::Storage>, value| match acc {
                    Some(mut acc) => {
                        V::bind(&mut acc, &value);
                        Some(acc)
                    }
                    None => Some(value),
                })
                .map(|vector| {
                    let vsa = self.storage.vectors.vsa.clone();
                    Vector::from_normalized(vsa, self.storage.vectors.size, vector)
                })
        })
    }

    /// Iterate through the rows of the subset, generating combinatorial bundles of the vectors for each row.
    pub fn bundle_rows_combinatorial(
        &self,
        ways: usize,
    ) -> impl Iterator<Item = Option<Vector<S, V, NotNormalized<V>>>> {
        RowIterator::from(self).map(move |sample_iter| {
            // Flatten cleanly drops the `None` values (missing data slots) yielded
            // by the corrected SampleIterator, natively implementing our missing-data theory.
            let features: Vec<V::Storage> = sample_iter.flatten().collect();

            if features.is_empty() {
                return None;
            }

            let mut row_accumulator: Option<V::Accumulator> = None;
            let n = features.len();

            // Local helper to recursively generate mathematical combinations of indices
            fn combine(
                start: usize,
                n: usize,
                k: usize,
                current: &mut Vec<usize>,
                result: &mut Vec<Vec<usize>>,
            ) {
                if current.len() == k {
                    result.push(current.clone());
                    return;
                }
                for i in start..n {
                    current.push(i);
                    combine(i + 1, n, k, current, result);
                    current.pop();
                }
            }

            // Construct all k-way combinations up to `ways`
            for k in 1..=ways {
                if k > n {
                    break; // Cannot generate combinations larger than available valid features
                }

                let mut combos = Vec::new();
                combine(0, n, k, &mut Vec::with_capacity(k), &mut combos);

                for combo in combos {
                    // Start the binding term with the first feature in the combination
                    let mut bound_term = features[combo[0]].clone();

                    // Bind the remaining features
                    for &idx in &combo[1..] {
                        V::bind(&mut bound_term, &features[idx]);
                    }

                    // Convert into un-normalized state so they can be summed without amplitude loss
                    let term_denorm = V::denormalize(bound_term);

                    // Add into the row's mathematical superposition
                    match &mut row_accumulator {
                        Some(acc) => {
                            self.storage
                                .vectors
                                .vsa
                                .bundle_with_accumulator(acc, &term_denorm);
                        }
                        None => {
                            row_accumulator = Some(term_denorm);
                        }
                    }
                }
            }

            row_accumulator.map(|acc| {
                let vsa = self.storage.vectors.vsa.clone();
                Vector::new(vsa, self.storage.vectors.size, acc)
            })
        })
    }

    /// Bundle the entire dataset into a single vector.
    pub fn bundle_dataset<T1: VectorType<V>, T2: VectorType<V>>(&self) -> Option<Vector<S, V, T2>> {
        self.bundle_vectors(self.bundle_rows::<T1>())
    }

    /// Bundle combinatorial interactions of the entire dataset into a single vector.
    pub fn bundle_dataset_combinatorial(
        &self,
        ways: usize,
    ) -> Option<Vector<S, V, NotNormalized<V>>> {
        self.bundle_vectors(self.bundle_rows_combinatorial(ways))
    }

    /// Bind the bundled rows of dataset into a single vector.
    pub fn bind_dataset(&self) -> Option<Vector<S, V, Normalized<V>>> {
        self.bind_vectors(self.bundle_rows())
    }

    /// Bind the entire dataset into a single vector.
    pub fn bind_entire_dataset(&self) -> Option<Vector<S, V, Normalized<V>>> {
        self.bind_vectors(self.bind_rows())
    }

    fn bundle_vectors<T1: VectorType<V>, T2: VectorType<V>>(
        &self,
        iterator: impl Iterator<Item = Option<Vector<S, V, T1>>>,
    ) -> Option<Vector<S, V, T2>> {
        iterator
            .flatten()
            .fold(
                None,
                |acc: Option<V::Accumulator>, row_vector: Vector<S, V, T1>| match acc {
                    Some(mut acc) => {
                        let row_vector_denormalized = row_vector.data.into().0;
                        self.storage
                            .vectors
                            .vsa
                            .bundle_with_accumulator(&mut acc, &row_vector_denormalized);
                        Some(acc)
                    }
                    None => Some(row_vector.data.into().0),
                },
            )
            .map(|vector_unnormalized| {
                let vsa = self.storage.vectors.vsa.clone();
                Vector::new(vsa, self.storage.vectors.size, vector_unnormalized)
            })
    }

    fn bind_vectors(
        &self,
        iterator: impl Iterator<Item = Option<Vector<S, V, Normalized<V>>>>,
    ) -> Option<Vector<S, V, Normalized<V>>> {
        iterator
            .flatten()
            .fold(
                None,
                |acc: Option<V::Storage>, row_vector: Vector<S, V, Normalized<V>>| match acc {
                    Some(mut acc) => {
                        V::bind(&mut acc, &row_vector.data.0);
                        Some(acc)
                    }
                    None => Some(row_vector.data.0),
                },
            )
            .map(|vector| {
                let vsa = self.storage.vectors.vsa.clone();
                Vector::from_normalized(vsa, self.storage.vectors.size, vector)
            })
    }

    fn height(&self) -> usize {
        self.vectors.len() / self.width
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture> Subset<'a, S, V> {
    fn calculate_index(&self, x: usize, y: usize) -> usize {
        y + x * self.height()
    }
}

/// An error that can occur when creating a subset.
#[derive(Debug, Clone)]
pub enum Error {
    /// A new column not previously seen in the storage was found in the dataframe.
    ColumnNotFound {
        /// The name of the column.
        name: String,
    },
    /// A column in the dataframe has an incompatible type with the storage.
    ColumnTypeMismatch {
        /// The name of the column.
        name: String,
    },
    /// A value in the dataframe was not found in the storage.
    ValueNotFound {
        /// The name of the column.
        column: String,
        /// The value that was not found.
        value: String,
    },
    /// The underlying Polars library returned an error.
    PolarsError(PolarsError),
}

impl From<PolarsError> for Error {
    fn from(err: PolarsError) -> Self {
        Error::PolarsError(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ColumnNotFound { name } => write!(f, "column '{}' not found", name),
            Error::ColumnTypeMismatch { name } => {
                write!(f, "column '{}' has incompatible type", name)
            }
            Error::ValueNotFound { column, value } => {
                write!(f, "value '{}' not found in column '{}'", value, column)
            }
            Error::PolarsError(err) => write!(f, "polars error: {}", err),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug, Clone)]
struct RowIterator<'a, 'b, S: Size, V: VectorSymbolicArchitecture> {
    subset: &'a Subset<'b, S, V>,
    y_iter: std::ops::Range<usize>,
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> From<&'a Subset<'b, S, V>>
    for RowIterator<'a, 'b, S, V>
{
    fn from(subset: &'a Subset<'b, S, V>) -> Self {
        Self {
            subset,
            y_iter: 0..subset.height(),
        }
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> Iterator for RowIterator<'a, 'b, S, V> {
    type Item = SampleIterator<'a, 'b, S, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let y = self.y_iter.next()?;
        Some(SampleIterator {
            subset: self.subset,
            y,
            x_iter: 0..self.subset.width,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.y_iter.size_hint()
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> ExactSizeIterator
    for RowIterator<'a, 'b, S, V>
{
    fn len(&self) -> usize {
        self.y_iter.len()
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> std::iter::FusedIterator
    for RowIterator<'a, 'b, S, V>
{
}

#[derive(Debug, Clone)]
struct SampleIterator<'a, 'b, S: Size, V: VectorSymbolicArchitecture> {
    subset: &'a Subset<'b, S, V>,
    y: usize,
    x_iter: std::ops::Range<usize>,
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> Iterator for SampleIterator<'a, 'b, S, V> {
    type Item = Option<V::Storage>;

    fn next(&mut self) -> Option<Self::Item> {
        let x = self.x_iter.next()?;
        let index = self.subset.calculate_index(x, self.y);
        let Some(vector_index) = self.subset.vectors[index] else {
            return Some(None);
        };

        let column_vector_index = self.subset.columns[x];
        let mut bound = self.subset.storage.vectors.vectors[column_vector_index.0]
            .data
            .0
            .clone();
        V::bind(
            &mut bound,
            &self.subset.storage.vectors.vectors[vector_index.0].data.0,
        );

        Some(Some(bound))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.x_iter.size_hint()
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> ExactSizeIterator
    for SampleIterator<'a, 'b, S, V>
{
    fn len(&self) -> usize {
        self.x_iter.len()
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture> std::iter::FusedIterator
    for SampleIterator<'a, 'b, S, V>
{
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dollar_dataframe() -> DataFrame {
        let name_dtype = DataType::from_frozen_categories(
            FrozenCategories::new(["USA", "MEX"].into_iter()).unwrap(),
        );
        let money_dtype = DataType::from_frozen_categories(
            FrozenCategories::new(["DOL", "PES"].into_iter()).unwrap(),
        );
        let capital_dtype = DataType::from_frozen_categories(
            FrozenCategories::new(["WDC", "MXC"].into_iter()).unwrap(),
        );

        DataFrame::new(
            2,
            vec![
                Column::new("NAM".into(), &["USA", "MEX"])
                    .cast(&name_dtype)
                    .unwrap(),
                Column::new("MON".into(), &["DOL", "PES"])
                    .cast(&money_dtype)
                    .unwrap(),
                Column::new("CAP".into(), &["WDC", "MXC"])
                    .cast(&capital_dtype)
                    .unwrap(),
            ],
        )
        .expect("valid dataframe")
    }

    #[test]
    fn test_dataset_bundle_vector() {
        let df = create_dollar_dataframe();
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<10000>, &df).expect("valid vector storage");
        let expression: crate::Expression =
            "((NAM * USA) + (CAP * WDC) + (MON * DOL)) + ((NAM * MEX) + (CAP * MXC) + (MON * PES))"
                .parse()
                .expect("valid expression");

        let subset = storage.subset(&df).expect("valid subset");
        assert_eq!(
            storage
                .execute(&expression)
                .expect("valid execution")
                .into_owned(),
            subset
                .bundle_dataset::<Normalized<crate::architectures::MultiplyAddPermute<u8>>, Normalized<crate::architectures::MultiplyAddPermute<u8>>>()
                .expect("valid dataset vector")
        );
    }

    #[test]
    fn test_dataset_bind_bundled_vector() {
        let df = create_dollar_dataframe();
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<10000>, &df).expect("valid vector storage");
        let expression: crate::Expression =
            "((NAM * USA) + (CAP * WDC) + (MON * DOL)) * ((NAM * MEX) + (CAP * MXC) + (MON * PES))"
                .parse()
                .expect("valid expression");

        let subset = storage.subset(&df).expect("valid subset");
        assert_eq!(
            storage
                .execute(&expression)
                .expect("valid execution")
                .into_owned(),
            subset.bind_dataset().expect("valid dataset vector")
        );
    }

    #[test]
    fn test_dataset_bind_vector_entire_dataset() {
        let df = create_dollar_dataframe();
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<10000>, &df).expect("valid vector storage");
        let expression: crate::Expression =
            "((NAM * USA) * (CAP * WDC) * (MON * DOL)) * ((NAM * MEX) * (CAP * MXC) * (MON * PES))"
                .parse()
                .expect("valid expression");

        let subset = storage.subset(&df).expect("valid subset");
        assert_eq!(
            storage
                .execute(&expression)
                .expect("valid execution")
                .into_owned(),
            subset.bind_entire_dataset().expect("valid dataset vector")
        );
    }
}
