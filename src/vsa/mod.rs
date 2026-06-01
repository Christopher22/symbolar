pub mod architectures;
mod vector;

pub use self::vector::{
    Dynamic, Fixed, FixedSize, Normalized, NotNormalized, Size, Vector, VectorType,
};
