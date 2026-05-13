#[cfg(feature = "polars")]
mod polars_util;
mod queryable;
mod selector;

use std::{borrow::Cow, collections::HashMap};

#[cfg(feature = "polars")]
use self::polars_util::Columns;
use crate::{Expression, Size, UnknownValue, Vector, architectures::VectorSymbolicArchitecture};

#[cfg(feature = "polars")]
pub use self::polars_util::{Column, Subset, SubsetError};
pub use self::queryable::Queryable;
pub use self::selector::Selector;

/// A numerical ID for a vector in the storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorIndex(pub(crate) usize);

#[derive(Debug, Clone)]
struct NamedVectors<S: Size, V: VectorSymbolicArchitecture> {
    vectors: Vec<Vector<S, V>>,
    names: HashMap<String, VectorIndex>,
    vsa: V,
    size: S,
}

impl<S: Size, V: VectorSymbolicArchitecture> NamedVectors<S, V> {
    fn new(vsa: V, size: S) -> Option<Self> {
        if !V::valid_size(size) {
            return None;
        }
        Some(Self {
            vectors: Vec::new(),
            names: HashMap::new(),
            vsa,
            size,
        })
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
            self.vectors
                .push(Vector::random(&self.vsa, self.size).expect("previously tested"));
            index
        })
    }
}

/// A storage of (named) vectors related to a DataFrame.
#[derive(Debug, Clone)]
pub struct Storage<S: Size, V: VectorSymbolicArchitecture> {
    vectors: NamedVectors<S, V>,
    #[cfg(feature = "polars")]
    columns: Columns,
}

impl<S: Size, V: VectorSymbolicArchitecture> Storage<S, V> {
    fn execute_expression<'x, 'y: 'x>(
        &'x self,
        expression: &'y Expression,
    ) -> Result<Cow<'x, Vector<S, V>>, UnknownValue> {
        expression.evaluate(|name| self.get(&name).map(Cow::Borrowed))
    }

    /// Create a new empty vector storage.
    pub fn new(vsa: V, size: S) -> Option<Self> {
        Some(Self {
            vectors: NamedVectors::new(vsa, size)?,
            #[cfg(feature = "polars")]
            columns: Columns::default(),
        })
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
        self.execute_expression(expression)
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

#[cfg(feature = "polars")]
impl<S: Size, V: VectorSymbolicArchitecture> Storage<S, V> {
    /// Create a new vector storage from a DataFrame.
    pub fn from_dataframe(
        vsa: V,
        size: S,
        dataframe: &polars::frame::DataFrame,
    ) -> Result<Self, StorageError> {
        let mut vectors = NamedVectors::new(vsa, size).ok_or(StorageError::InvalidSize)?;
        let columns = Columns::from_columns(&mut vectors, dataframe.columns().iter())?;
        Ok(Self { vectors, columns })
    }

    /// Create a subset of the storage corresponding to a specific DataFrame.
    pub fn subset<'a>(
        &'a self,
        dataframe: &polars::frame::DataFrame,
    ) -> Result<Subset<'a, S, V>, SubsetError> {
        Subset::new(self, dataframe)
    }

    /// Query all known colums in the storage.
    pub fn columns(&self) -> impl Iterator<Item = Column<'_>> {
        self.columns.0.keys().map(|s| Column::from_str(s.as_str()))
    }
}

impl<S: Size, V: VectorSymbolicArchitecture, Q: Queryable> std::ops::Index<Q> for Storage<S, V> {
    type Output = Vector<S, V>;

    fn index(&self, query: Q) -> &Self::Output {
        query.query(self).expect("valid reference")
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
#[allow(missing_copy_implementations)]
pub enum StorageError {
    /// An invalid data type was encountered in the DataFrame.
    #[cfg(feature = "polars")]
    InvalidDataType {
        /// The name of the column with the invalid data type.
        column: String,
        /// The invalid data type encountered.
        dtype: polars::datatypes::DataType,
    },
    /// The size of the vectors is invalid for the architecture.
    InvalidSize,
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "polars")]
            StorageError::InvalidDataType { column, dtype } => {
                write!(f, "invalid data type for column '{column}': {dtype}")
            }
            StorageError::InvalidSize => write!(f, "invalid vector size for the architecture"),
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
        let storage = Storage::new(vsa, crate::Fixed::<128>).expect("valid size");
        #[cfg(feature = "polars")]
        assert!(storage.columns().next().is_none());
        assert!(storage.values().next().is_none());
    }

    #[test]
    fn test_push() {
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let mut storage = Storage::new(vsa, crate::Fixed::<128>).expect("valid size");
        storage.push("vec1");
        storage.push("vec2");
        storage.push("vec1");

        assert_ne!(storage["vec1"], storage["vec2"]);
        assert_eq!(storage["vec1"], storage["vec1"]);
        assert_eq!(storage.values().count(), 2);
        #[cfg(feature = "polars")]
        assert_eq!(storage.columns().count(), 0);
    }

    #[test]
    fn test_extend() {
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let mut storage = Storage::new(vsa, crate::Fixed::<128>).expect("valid size");
        storage.extend(["vec1", "vec1", "vec2"]);

        assert_ne!(storage["vec1"], storage["vec2"]);
        assert_eq!(storage["vec1"], storage["vec1"]);
        assert_eq!(storage.values().count(), 2);
        #[cfg(feature = "polars")]
        assert_eq!(storage.columns().count(), 0);
    }

    #[test]
    fn test_execute() {
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let mut storage = Storage::new(vsa, crate::Fixed::<128>).expect("valid size");
        storage.extend(["vec1", "vec2"]);

        let expr = Expression::new("vec1");
        let result = storage.execute(&expr).expect("expression should exist");
        assert_eq!(result.as_ref(), &storage["vec1"]);
    }

    #[test]
    fn test_select_mutliple() {
        let vsa = crate::architectures::MultiplyAddPermute::<u8>::new(42);
        let mut storage = Storage::new(vsa, crate::Fixed::<128>).expect("valid size");
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

    #[test]
    fn test_invalid_size() {
        let vsa = crate::architectures::VectorDerivedTransformationBinding::<f64>::new(42);
        assert!(Storage::new(vsa, crate::Fixed::<128>).is_none());
    }
}
