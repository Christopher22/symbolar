use polars::prelude::*;

use super::ColumnType;

use crate::{
    Normalized, Size, Storage, Vector, VectorIndex, VectorType,
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

    /// Get the vectors corresponding to a specific rows without normalization.
    pub fn bundle_rows<T: VectorType<V>>(&self) -> Vec<Option<Vector<S, V, T>>> {
        (0..self.height())
            // Calculate each row
            .map(|y| {
                let mut bundled: Option<V::Accumulator> = None;

                for x in 0..self.width {
                    let index = self.calculate_index(x, y);
                    let Some(vector_index) = self.vectors[index] else {
                        continue;
                    };

                    let column_vector_index = self.columns[x];
                    let mut bound = self.storage.vectors.vectors[column_vector_index.0]
                        .data
                        .0
                        .clone();
                    V::bind(
                        &mut bound,
                        &self.storage.vectors.vectors[vector_index.0].data.0,
                    );
                    let value = V::denormalize(bound);

                    match bundled.as_mut() {
                        Some(acc) => self
                            .storage
                            .vectors
                            .vsa
                            .bundle_with_accumulator(acc, &value),
                        None => bundled = Some(value),
                    }
                }

                bundled.map(|vector_unnormalized| {
                    let vsa = self.storage.vectors.vsa.clone();
                    Vector::new(vsa, self.storage.vectors.size, vector_unnormalized)
                })
            })
            .collect()
    }

    /// Bundle the entire dataset into a single vector.
    pub fn bundle_dataset<T1: VectorType<V>, T2: VectorType<V>>(&self) -> Option<Vector<S, V, T2>> {
        let rows = self.bundle_rows::<T1>();
        let mut bundled: Option<V::Accumulator> = None;

        for row in rows.into_iter().flatten() {
            let value = row.data.into().0;
            match bundled.as_mut() {
                Some(acc) => self
                    .storage
                    .vectors
                    .vsa
                    .bundle_with_accumulator(acc, &value),
                None => bundled = Some(value),
            }
        }

        bundled.map(|vector_unnormalized| {
            let vsa = self.storage.vectors.vsa.clone();
            Vector::new(vsa, self.storage.vectors.size, vector_unnormalized)
        })
    }

    /// Bind the entire dataset into a single vector.
    pub fn bind_dataset(&self) -> Option<Vector<S, V, Normalized<V>>> {
        let rows = self.bundle_rows::<Normalized<V>>();
        rows.into_iter().fold(None, |acc, row| match (acc, row) {
            (Some(mut acc_vector), Some(row_vector)) => {
                V::bind(&mut acc_vector.data.0, &row_vector.data.0);
                Some(Vector::new(
                    acc_vector.vsa,
                    acc_vector.size,
                    V::denormalize(acc_vector.data.0),
                ))
            }
            (None, Some(row_vector)) => Some(row_vector),
            (acc, None) => acc,
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
    fn test_dataset_bind_vector() {
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
}
