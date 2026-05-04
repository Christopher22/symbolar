use polars_vsa::{Vector, architectures::MultiplyAddPermute};

fn main() {
    let vsa = MultiplyAddPermute::<u8>::new(42);
    let vec1 = Vector::random_fixed::<32>(&vsa);
    let vec2 = Vector::random_fixed::<32>(&vsa);

    let _bundling_with_normalize = vec1.clone() + &vec2;
    let _binding = vec1.clone() * &vec2;
    let _permuted = vec1.permute(5);
}
