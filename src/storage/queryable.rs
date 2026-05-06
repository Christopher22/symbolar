use super::ColumnType;
use crate::{
    Column, Size, Storage, Value, Vector, VectorIndex, architectures::VectorSymbolicArchitecture,
};

/// A queryable item for a vector storage.
pub trait Queryable {
    /// Try to query a vector index from the storage.
    fn query_index<S: Size, V: VectorSymbolicArchitecture>(
        &self,
        storage: &Storage<S, V>,
    ) -> Option<VectorIndex>;

    /// Try to query a vector from the storage.
    fn query<'a, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        storage: &'a Storage<S, V>,
    ) -> Option<&'a Vector<S, V>> {
        Self::query_index(self, storage).map(|index| &storage.vectors.vectors[index.0])
    }
}

impl<'s> Queryable for Value<'s> {
    fn query_index<S: Size, V: VectorSymbolicArchitecture>(
        &self,
        storage: &Storage<S, V>,
    ) -> Option<VectorIndex> {
        storage.vectors.names.get(self.0.as_ref()).copied()
    }
}

impl<'s> Queryable for Column<'s> {
    fn query_index<S: Size, V: VectorSymbolicArchitecture>(
        &self,
        storage: &Storage<S, V>,
    ) -> Option<VectorIndex> {
        storage.columns.get(self.0.as_ref()).map(|col| col.vector)
    }
}

impl<'s1, 's2> Queryable for (Column<'s1>, Value<'s2>) {
    fn query_index<S: Size, V: VectorSymbolicArchitecture>(
        &self,
        storage: &Storage<S, V>,
    ) -> Option<VectorIndex> {
        let column = storage.columns.get(self.0.0.as_ref())?;
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

impl Queryable for VectorIndex {
    fn query<'a, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        storage: &'a Storage<S, V>,
    ) -> Option<&'a Vector<S, V>> {
        storage.vectors.get_by_index(*self)
    }

    fn query_index<S: Size, V: VectorSymbolicArchitecture>(
        &self,
        _storage: &Storage<S, V>,
    ) -> Option<VectorIndex> {
        Some(*self)
    }
}

impl Queryable for &str {
    fn query_index<S: Size, V: VectorSymbolicArchitecture>(
        &self,
        storage: &Storage<S, V>,
    ) -> Option<VectorIndex> {
        storage.vectors.names.get(*self).copied()
    }

    fn query<'a, S: Size, V: VectorSymbolicArchitecture>(
        &self,
        storage: &'a Storage<S, V>,
    ) -> Option<&'a Vector<S, V>> {
        storage.vectors.get_by_name(self)
    }
}
