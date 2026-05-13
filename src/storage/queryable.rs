use crate::{Size, Storage, Value, Vector, VectorIndex, architectures::VectorSymbolicArchitecture};

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
