mod queryable;
mod selector;

use std::{borrow::Cow, collections::HashMap};

use polars::prelude::*;

use crate::{Expression, Size, UnknownValue, Vector, architectures::VectorSymbolicArchitecture};

pub use self::queryable::Queryable;
pub use self::selector::Selector;

/// A numerical ID for a vector in the storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VectorIndex(pub(crate) usize);

#[derive(Debug, Clone)]
struct NamedVectors<S: Size, V: VectorSymbolicArchitecture> {
    vectors: Vec<Vector<S, V>>,
    names: HashMap<String, VectorIndex>,
    vsa: V,
    size: S,
}

impl<S: Size, V: VectorSymbolicArchitecture> NamedVectors<S, V> {
    fn new(vsa: V, size: S) -> Self {
        Self {
            vectors: Vec::new(),
            names: HashMap::new(),
            vsa,
            size,
        }
    }

    fn get_by_index(&self, index: VectorIndex) -> Option<&Vector<S, V>> {
        self.vectors.get(index.0)
    }

    fn get_by_name(&self, name: &str) -> Option<&Vector<S, V>> {
        self.names.get(name).map(|&index| &self.vectors[index.0])
    }

    pub fn get_or_insert(&mut self, name: impl Into<String>) -> VectorIndex {
        *self.names.entry(name.into()).or_insert_with(|| {
            let index = VectorIndex(self.vectors.len());
            self.vectors.push(Vector::random(&self.vsa, self.size));
            index
        })
    }
}

#[derive(Debug, Clone)]
struct ColumnData {
    vector: VectorIndex,
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
pub struct Storage<S: Size, V: VectorSymbolicArchitecture> {
    vectors: NamedVectors<S, V>,
    columns: HashMap<String, ColumnData>,
}

impl<S: Size, V: VectorSymbolicArchitecture> Storage<S, V> {
    /// Create a new empty vector storage.
    pub fn new(vsa: V, size: S) -> Self {
        Self {
            vectors: NamedVectors::new(vsa, size),
            columns: HashMap::new(),
        }
    }

    /// Create a new vector storage from a DataFrame.
    pub fn from_dataframe(vsa: V, size: S, dataframe: DataFrame) -> Result<Self, StorageError> {
        let mut vectors = NamedVectors::new(vsa, size);
        let mut columns = HashMap::new();

        for col in dataframe.columns().iter() {
            let name = col.name();
            columns.insert(
                name.to_string(),
                match col.dtype() {
                    DataType::Enum(categories, _) => ColumnData {
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
                    DataType::String => {
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

    /// Add a new vector with the given name to the storage. If it already exists, return the existing vector.
    pub fn push(&mut self, name: impl Into<String>) -> &Vector<S, V> {
        let index = self.vectors.get_or_insert(name);
        &self.vectors.vectors[index.0]
    }

    /// Add multiple vectors with the given names to the storage.
    pub fn extend(&mut self, names: impl IntoIterator<Item = impl Into<String>>) {
        for name in names {
            self.push(name);
        }
    }

    /// Retrieve a vector by a query.
    pub fn get<Q: Queryable>(&self, query: &Q) -> Option<&Vector<S, V>> {
        query.query(self)
    }

    /// Retrieve multiple vectors from the storage using a selector.
    pub fn get_multiple<'a, I: Selector<S, V>>(
        &'a self,
        selector: &'a I,
    ) -> VectorIter<'a, S, V, I> {
        VectorIter::new(self, selector)
    }

    /// Query all known colums in the storage.
    pub fn columns(&self) -> impl Iterator<Item = Column<'_>> {
        self.columns.keys().map(|s| Column::from_str(s.as_str()))
    }

    /// Query all known values in the storage. This include the columns themselves.
    pub fn values(&self) -> impl Iterator<Item = Value<'_>> {
        self.vectors
            .names
            .keys()
            .map(|s| Value::from_str(s.as_str()))
    }

    /// Execute an expression on the storage, returning the resulting vector.
    pub fn execute<'x, 'y: 'x>(
        &'x self,
        expression: &'y Expression,
    ) -> Result<Cow<'x, Vector<S, V>>, UnknownValue> {
        expression.evaluate(|name| self.get(&name).map(Cow::Borrowed))
    }

    /// Compute the cosine similarity between a given vector and all vectors selected by a selector.
    pub fn cosine_similarities<'a, I: Selector<S, V> + 'a>(
        &'a self,
        vector: &'a Vector<S, V>,
        selector: &'a I,
    ) -> impl Iterator<Item = (VectorIndex, f64)> + 'a {
        selector
            .select(self)
            .map(|index| (index, vector.similarity(&self.vectors.vectors[index.0])))
    }

    /// Find the most similar vector to a given vector among those selected by a selector.
    pub fn find<I: Selector<S, V>>(
        &self,
        vector: &Vector<S, V>,
        selector: &I,
    ) -> Option<VectorIndex> {
        self.cosine_similarities(vector, selector)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(index, _)| index)
    }
}

impl<S: Size, V: VectorSymbolicArchitecture, Q: Queryable> std::ops::Index<Q> for Storage<S, V> {
    type Output = Vector<S, V>;

    fn index(&self, query: Q) -> &Self::Output {
        query.query(self).expect("valid reference")
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

/// An iterator over vectors in the storage, selected by a selector.
#[derive(Debug, Clone)]
pub struct VectorIter<'a, S: Size, V: VectorSymbolicArchitecture, I: 'a + Selector<S, V>> {
    storage: &'a Storage<S, V>,
    iterator: I::Indices<'a>,
}

impl<'a, S: Size, V: VectorSymbolicArchitecture, I: 'a + Selector<S, V>> VectorIter<'a, S, V, I> {
    fn new(storage: &'a Storage<S, V>, selector: &'a I) -> Self {
        Self {
            storage,
            iterator: selector.select(storage),
        }
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture, I: Selector<S, V>> Iterator
    for VectorIter<'a, S, V, I>
{
    type Item = &'a Vector<S, V>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator
            .next()
            .map(|index| &self.storage.vectors.vectors[index.0])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iterator.size_hint()
    }
}

impl<'a, S: Size, V: VectorSymbolicArchitecture, I: Selector<S, V>> ExactSizeIterator
    for VectorIter<'a, S, V, I>
where
    I::Indices<'a>: ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.iterator.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction_empty() {
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let storage = Storage::new(vsa, crate::Fixed::<128>);
        assert!(storage.columns().next().is_none());
        assert!(storage.values().next().is_none());
    }

    #[test]
    fn test_construction_from_dataframe() {
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
            Storage::from_dataframe(vsa, crate::Fixed::<128>, df).expect("valid vector storage");

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

    #[test]
    fn test_push() {
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let mut storage = Storage::new(vsa, crate::Fixed::<128>);
        storage.push("vec1");
        storage.push("vec2");
        storage.push("vec1");

        assert_ne!(storage["vec1"], storage["vec2"]);
        assert_eq!(storage["vec1"], storage["vec1"]);
        assert_eq!(storage.values().count(), 2);
        assert_eq!(storage.columns().count(), 0);
    }

    #[test]
    fn test_extend() {
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let mut storage = Storage::new(vsa, crate::Fixed::<128>);
        storage.extend(["vec1", "vec1", "vec2"]);

        assert_ne!(storage["vec1"], storage["vec2"]);
        assert_eq!(storage["vec1"], storage["vec1"]);
        assert_eq!(storage.values().count(), 2);
        assert_eq!(storage.columns().count(), 0);
    }

    #[test]
    fn test_execute() {
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let mut storage = Storage::new(vsa, crate::Fixed::<128>);
        storage.extend(["vec1", "vec2"]);

        let expr = Expression::new("vec1");
        let result = storage.execute(&expr).expect("expression should exist");
        assert_eq!(result.as_ref(), &storage["vec1"]);
    }

    #[test]
    fn test_select_mutliple() {
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let mut storage = Storage::new(vsa, crate::Fixed::<128>);
        storage.extend(["vec1", "vec2", "vec3"]);

        let selector = ["vec1", "vec3"].as_slice();
        let selected_1: Vec<_> = storage.get_multiple(&selector).collect();
        assert_eq!(selected_1.len(), 2);
        assert_eq!(selected_1[0], &storage["vec1"]);
        assert_eq!(selected_1[1], &storage["vec3"]);

        let selected_2: Vec<_> = storage.get_multiple(&()).collect();
        assert_eq!(selected_2.len(), 3);
        assert_eq!(selected_2[0], &storage["vec1"]);
        assert_eq!(selected_2[1], &storage["vec2"]);
        assert_eq!(selected_2[2], &storage["vec3"]);
    }
}
