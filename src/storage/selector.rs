use crate::{Queryable, Size, Storage, VectorIndex, architectures::VectorSymbolicArchitecture};

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
