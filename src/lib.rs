//! Type-safe and efficient Vector Symbolic Architectures.

#![deny(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

mod expression;
mod storage;
mod vsa;

pub use self::expression::{EvaluateOps, Expression, ParseError, UnknownValue};
pub use self::storage::{
    Column, Queryable, Selector, Storage, StorageError, Subset, SubsetError, Value, VectorIndex,
    VectorIter,
};
pub use self::vsa::*;
