use std::{borrow::Cow, collections::HashMap};

use polars::prelude::*;

use crate::{Size, Vector, architectures::VectorSymbolicArchitecture};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VectorIndex(usize);

#[derive(Debug, Clone)]
struct NamedVectors<S: Size, V: VectorSymbolicArchitecture> {
    vectors: Vec<Vector<S, V>>,
    names: Vec<String>,
}

impl<S: Size, V: VectorSymbolicArchitecture> NamedVectors<S, V> {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            names: Vec::new(),
        }
    }

    fn get(&self, name: &str) -> Option<&Vector<S, V>> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|index| &self.vectors[index])
    }

    fn push(&mut self, name: String, vector: Vector<S, V>) -> VectorIndex {
        let index = self.vectors.len();
        self.vectors.push(vector);
        self.names.push(name);
        VectorIndex(index)
    }

    pub fn get_or_insert(&mut self, vsa: &V, size: S, name: &str) -> VectorIndex {
        if let Some(index) = self.names.iter().position(|n| n == name) {
            return VectorIndex(index);
        }
        self.push(name.to_string(), Vector::random(vsa, size))
    }
}

#[derive(Debug, Clone)]
struct ColumnData<S: Size, V: VectorSymbolicArchitecture> {
    vector: Vector<S, V>,
    column_type: ColumnType,
}

#[derive(Debug, Clone)]
enum ColumnType {
    String,
    Enum {
        categories: Arc<FrozenCategories>,
        values: Vec<VectorIndex>,
    },
}

/// A storage of (named) vectors related to a DataFrame.
#[derive(Debug, Clone)]
pub struct VectorStorage<S: Size, V: VectorSymbolicArchitecture> {
    vectors: NamedVectors<S, V>,
    columns: HashMap<String, ColumnData<S, V>>,
}

impl<S: Size, V: VectorSymbolicArchitecture> VectorStorage<S, V> {
    /// Create a new vector storage from a DataFrame.
    pub fn new(vsa: V, size: S, dataframe: DataFrame) -> Result<Self, StorageError> {
        let mut vectors = NamedVectors::new();
        let mut columns = HashMap::new();

        for col in dataframe.columns().iter() {
            let name = col.name();
            columns.insert(
                name.to_string(),
                match col.dtype() {
                    DataType::Enum(categories, _) => ColumnData {
                        vector: Vector::random(&vsa, size),
                        column_type: ColumnType::Enum {
                            categories: categories.clone(),
                            values: categories
                                .categories()
                                .iter()
                                .map(|cat| {
                                    // TODO: Handly invalid str
                                    vectors.get_or_insert(&vsa, size, cat.expect("valid str"))
                                })
                                .collect(),
                        },
                    },
                    DataType::String => {
                        // The slow way: we need to iterate over all values to create vectors for them.
                        for value in col.str().expect("string column").into_iter().flatten() {
                            vectors.get_or_insert(&vsa, size, value);
                        }
                        ColumnData {
                            vector: Vector::random(&vsa, size),
                            column_type: ColumnType::String,
                        }
                    }
                    dtype => {
                        return Err(StorageError::InvalidDataType {
                            column: col.name().to_string(),
                            dtype: dtype.clone(),
                        });
                    }
                },
            );
        }

        Ok(Self { vectors, columns })
    }

    /// Retrieve a vector by a query.
    pub fn get<Q: Queryable>(&self, query: &Q) -> Option<&Vector<S, V>> {
        query.query(self)
    }
}

impl<S: Size, V: VectorSymbolicArchitecture, Q: Queryable> std::ops::Index<Q>
    for VectorStorage<S, V>
{
    type Output = Vector<S, V>;

    fn index(&self, query: Q) -> &Self::Output {
        query.query(self).expect("valid reference")
    }
}

/// A queryable item for a vector storage.
pub trait Queryable {
    /// Try to query a vector from the storage.
    fn query<'a, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        value: &'a VectorStorage<S, V>,
    ) -> Option<&'a Vector<S, V>>;
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
    fn query<'a, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        value: &'a VectorStorage<S, V>,
    ) -> Option<&'a Vector<S, V>> {
        value.columns.get(self.0.as_ref()).map(|col| &col.vector)
    }
}

/// A query for a specific value in a column.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Value<'s>(Cow<'s, str>);

impl<'a> Value<'a> {
    /// Create a value query from a string.
    pub const fn from_str(name: &'a str) -> Self {
        Self(Cow::Borrowed(name))
    }
}

impl<'s> From<&'s str> for Value<'s> {
    fn from(value: &'s str) -> Self {
        Self(Cow::Borrowed(value))
    }
}

impl From<String> for Value<'static> {
    fn from(value: String) -> Self {
        Self(Cow::Owned(value))
    }
}

impl std::fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0.as_ref())
    }
}

impl<'s> Queryable for Value<'s> {
    fn query<'a, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        value: &'a VectorStorage<S, V>,
    ) -> Option<&'a Vector<S, V>> {
        value.vectors.get(self.0.as_ref())
    }
}

impl<'s1, 's2> Queryable for (Column<'s1>, Value<'s2>) {
    fn query<'a, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        value: &'a VectorStorage<S, V>,
    ) -> Option<&'a Vector<S, V>> {
        let column = value.columns.get(self.0.0.as_ref())?;
        match column.column_type {
            ColumnType::String => {
                // For string columns, we can directly query the value vector.
                value.vectors.get(self.1.0.as_ref())
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
                let vector_index = &values[value_index];
                value.vectors.vectors.get(vector_index.0)
            }
        }
    }
}

impl Queryable for &str {
    fn query<'a, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        value: &'a VectorStorage<S, V>,
    ) -> Option<&'a Vector<S, V>> {
        Column::from_str(self)
            .query(value)
            .or_else(|| Value::from_str(self).query(value))
    }
}

/// An error related to vector storage construction or querying.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    /// An invalid data type was encountered in the DataFrame.
    InvalidDataType {
        /// The name of the column with the invalid data type.
        column: String,
        /// The invalid data type encountered.
        dtype: DataType,
    },
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::InvalidDataType { column, dtype } => {
                write!(f, "invalid data type for column '{column}': {dtype}")
            }
        }
    }
}

impl std::error::Error for StorageError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let enum_dtype = DataType::Enum(
            FrozenCategories::new(vec!["circle".into(), "square".into(), "triangle".into()])
                .unwrap(),
            Arc::new(CategoricalMapping::with_hasher(3, Default::default())),
        );

        let df = DataFrame::new(
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
            VectorStorage::new(vsa, crate::Fixed::<128>, df).expect("valid vector storage");

        assert_eq!(storage["color"], storage[Column::from_str("color")]);
        assert_eq!(storage["shape"], storage[Column::from_str("shape")]);
        assert_ne!(storage["shape"], storage["color"]);

        assert_eq!(storage["red"], storage[Value::from_str("red")]);

        assert_eq!(
            storage[(Column::from_str("shape"), Value::from_str("circle"))],
            storage[Value::from_str("circle")]
        );

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
