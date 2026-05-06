use super::ColumnType;
use crate::{
    Column, Queryable, Size, Storage, VectorIndex, architectures::VectorSymbolicArchitecture,
};

/// A selector for multiple vectors in the storage.
pub trait Selector<S: Size, V: VectorSymbolicArchitecture> {
    /// A iterator returing indices which should be included.
    type Indices<'a>: Iterator<Item = VectorIndex>
    where
        Self: 'a,
        S: 'a,
        V: 'a;

    /// Select the indices of the vectors in the storage.
    fn select<'a>(&'a self, storage: &'a Storage<S, V>) -> Self::Indices<'a>;
}

impl<S: Size, V: VectorSymbolicArchitecture> Selector<S, V> for () {
    type Indices<'a>
        = AllIter
    where
        S: 'a,
        V: 'a;

    fn select<'a>(&'a self, storage: &'a Storage<S, V>) -> Self::Indices<'a> {
        AllIter::new(storage)
    }
}

impl<'b, S: Size, V: VectorSymbolicArchitecture, T: Queryable> Selector<S, V> for &'b [T] {
    type Indices<'a>
        = SelectionIter<'a, 'b, S, V, T>
    where
        Self: 'a,
        S: 'a,
        V: 'a;

    fn select<'a>(&'a self, storage: &'a Storage<S, V>) -> Self::Indices<'a> {
        SelectionIter::new(storage, self)
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
            .get(self.0.as_ref())
            .map(|col| match col.column_type {
                ColumnType::String => ColumnIter::String(0..storage.vectors.vectors.len()),
                ColumnType::Enum { ref values, .. } => ColumnIter::Enum(values.iter()),
            })
            .unwrap_or(ColumnIter::Invalid)
    }
}

// --------------------- The different iterators ---------------------

#[derive(Debug, Clone)]
pub struct AllIter(std::ops::Range<usize>);

impl AllIter {
    fn new<S: Size, V: VectorSymbolicArchitecture>(storage: &Storage<S, V>) -> Self {
        Self(0..storage.vectors.vectors.len())
    }
}

impl Iterator for AllIter {
    type Item = VectorIndex;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(VectorIndex)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl ExactSizeIterator for AllIter {
    fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, Clone)]
pub struct SelectionIter<'a, 'b, S: Size, V: VectorSymbolicArchitecture, Q: Queryable> {
    storage: &'a Storage<S, V>,
    indices: std::slice::Iter<'b, Q>,
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture, Q: Queryable> SelectionIter<'a, 'b, S, V, Q> {
    fn new(storage: &'a Storage<S, V>, queries: &'b [Q]) -> Self {
        Self {
            storage,
            indices: queries.iter(),
        }
    }
}

impl<'a, 'b, S: Size, V: VectorSymbolicArchitecture, Q: Queryable> Iterator
    for SelectionIter<'a, 'b, S, V, Q>
{
    type Item = VectorIndex;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let query = self.indices.next()?;
            if let Some(index) = query.query_index(self.storage) {
                return Some(index);
            }
        }
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
