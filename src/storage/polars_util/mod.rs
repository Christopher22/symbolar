mod subset;
use std::{borrow::Cow, collections::HashMap, sync::Arc};

use crate::{Queryable, Selector, Size, Storage, Value, architectures::VectorSymbolicArchitecture};

use super::VectorIndex;

pub use self::subset::{
    Error as SubsetError, IntoSampleIterator, SamplesWithColumnVector, SamplesWithPosition, Subset,
};

#[derive(Debug, Clone)]
pub(crate) struct ColumnData {
    vector: VectorIndex,
    column_type: ColumnType,
}

#[derive(Debug, Clone)]
pub(crate) enum ColumnType {
    String,
    Enum {
        categories: Arc<polars::datatypes::FrozenCategories>,
        values: Vec<VectorIndex>,
    },
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Columns(pub(super) HashMap<String, ColumnData>);

impl Columns {
    pub fn from_columns<
        'a,
        S: Size,
        V: VectorSymbolicArchitecture,
        T: IntoIterator<Item = &'a polars::frame::column::Column>,
    >(
        vectors: &mut super::NamedVectors<S, V>,
        iter: T,
    ) -> Result<Self, super::StorageError> {
        let mut columns = HashMap::new();

        for col in iter {
            let name = col.name();
            columns.insert(
                name.to_string(),
                match col.dtype() {
                    polars::datatypes::DataType::Enum(categories, _) => ColumnData {
                        vector: vectors.get_or_insert(name.as_str()),
                        column_type: ColumnType::Enum {
                            categories: categories.clone(),
                            values: categories
                                .categories()
                                .iter()
                                .map(|cat| {
                                    // TODO: Handly invalid str
                                    vectors.get_or_insert(cat.expect("valid str"))
                                })
                                .collect(),
                        },
                    },
                    polars::datatypes::DataType::String => {
                        // The slow way: we need to iterate over all values to create vectors for them.
                        for value in col.str().expect("string column").into_iter().flatten() {
                            vectors.get_or_insert(value);
                        }
                        ColumnData {
                            vector: vectors.get_or_insert(name.as_str()),
                            column_type: ColumnType::String,
                        }
                    }
                    dtype => {
                        return Err(super::StorageError::InvalidDataType {
                            column: col.name().to_string(),
                            dtype: dtype.clone(),
                        });
                    }
                },
            );
        }

        Ok(Self(columns))
    }
}

/// A column name query.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Column<'s>(Cow<'s, str>);

impl<'a> Column<'a> {
    /// Create a column query from a string.
    pub const fn from_str(name: &'a str) -> Self {
        Self(Cow::Borrowed(name))
    }
}

impl<'s> From<&'s str> for Column<'s> {
    fn from(value: &'s str) -> Self {
        Self(Cow::Borrowed(value))
    }
}

impl From<String> for Column<'static> {
    fn from(value: String) -> Self {
        Self(Cow::Owned(value))
    }
}

impl std::fmt::Display for Column<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.as_ref())
    }
}

impl<'s> Queryable for Column<'s> {
    fn query_index<S: Size, V: VectorSymbolicArchitecture>(
        &self,
        storage: &Storage<S, V>,
    ) -> Option<VectorIndex> {
        storage.columns.0.get(self.0.as_ref()).map(|col| col.vector)
    }
}

impl<'s1, 's2> Queryable for (Column<'s1>, Value<'s2>) {
    fn query_index<S: Size, V: VectorSymbolicArchitecture>(
        &self,
        storage: &Storage<S, V>,
    ) -> Option<VectorIndex> {
        let column = storage.columns.0.get(self.0.0.as_ref())?;
        match column.column_type {
            ColumnType::String => {
                // For string columns, we can directly query the value vector.
                storage.vectors.names.get(self.1.0.as_ref()).copied()
            }
            ColumnType::Enum {
                ref categories,
                ref values,
            } => {
                // For enum columns, we check the existance.
                let value_index = categories
                    .categories()
                    .iter()
                    .position(|cat| cat == Some(self.1.0.as_ref()))?;
                Some(values[value_index])
            }
        }
    }
}

impl<S: Size, V: VectorSymbolicArchitecture> Selector<S, V> for Column<'_> {
    type Indices<'a>
        = ColumnIter<'a>
    where
        Self: 'a,
        S: 'a,
        V: 'a;

    fn select<'a>(&'a self, storage: &'a Storage<S, V>) -> Self::Indices<'a> {
        storage
            .columns
            .0
            .get(self.0.as_ref())
            .map(|col| match col.column_type {
                ColumnType::String => ColumnIter::String(0..storage.vectors.vectors.len()),
                ColumnType::Enum { ref values, .. } => ColumnIter::Enum(values.iter()),
            })
            .unwrap_or(ColumnIter::Invalid)
    }
}

/// A iterator over the vectors in a column.
#[derive(Debug, Clone)]
pub enum ColumnIter<'a> {
    Invalid,
    String(std::ops::Range<usize>),
    Enum(std::slice::Iter<'a, VectorIndex>),
}

impl Iterator for ColumnIter<'_> {
    type Item = VectorIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Invalid => None,
            Self::String(iter) => iter.next().map(VectorIndex),
            Self::Enum(iter) => iter.next().copied(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Invalid => (0, Some(0)),
            Self::String(iter) => iter.size_hint(),
            Self::Enum(iter) => iter.size_hint(),
        }
    }
}

impl ExactSizeIterator for ColumnIter<'_> {
    fn len(&self) -> usize {
        match self {
            ColumnIter::Invalid => 0,
            ColumnIter::String(iter) => iter.len(),
            ColumnIter::Enum(iter) => iter.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_from_dataframe() {
        let enum_dtype = polars::datatypes::DataType::from_frozen_categories(
            polars::datatypes::FrozenCategories::new(vec!["circle", "square", "triangle"]).unwrap(),
        );

        let df = polars::frame::DataFrame::new(
            3,
            vec![
                // Create a string column
                polars::frame::column::Column::new("color".into(), &["red", "green", "blue"]),
                // Create a enum column
                polars::frame::column::Column::new("shape".into(), &["circle", "square", "circle"])
                    .cast(&enum_dtype)
                    .unwrap(),
            ],
        )
        .expect("valid dataframe");

        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage =
            Storage::from_dataframe(vsa, crate::Fixed::<128>, &df).expect("valid vector storage");

        assert_eq!(storage.values().count(), 2 * 3 + 2);
        assert_eq!(storage.columns().count(), 2);

        assert_eq!(storage["color"], storage[Column::from_str("color")]);
        assert_eq!(storage["shape"], storage[Column::from_str("shape")]);
        assert_ne!(storage["shape"], storage["color"]);

        assert_eq!(storage["red"], storage[Value::from_str("red")]);

        assert_eq!(
            storage[(Column::from_str("shape"), Value::from_str("circle"))],
            storage[Value::from_str("circle")]
        );

        // There is nothing special about a column vector ...
        assert_eq!(storage[Column::from_str("color")], storage["color"]);
        // ... but it must exist.
        assert!(storage.get(&Column::from_str("red")).is_none());

        // A string column can be everything
        assert_eq!(
            storage[(Column::from_str("color"), Value::from_str("circle"))],
            storage["circle"]
        );

        // Enum-specific behavior
        assert!(storage.get(&Value::from_str("triangle")).is_some()); // Triangle has a vector, even if it was not in the column
        assert!(
            storage
                .get(&(Column::from_str("shape"), Value::from_str("red")))
                .is_none()
        ); // Red is not a valid value for the shape column
    }
}
