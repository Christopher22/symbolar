use polars_vsa::{
    Dynamic, Vector,
    architectures::{Storage, VectorSymbolicArchitecture},
};

trait Comparable<T>: Copy {
    fn is_equal(&self, v1: &T, v2: &T) -> bool;
}

#[derive(Debug, Clone, Copy)]
struct FloatCompare {
    tolerance: f64,
}

impl Comparable<f64> for FloatCompare {
    fn is_equal(&self, v1: &f64, v2: &f64) -> bool {
        (v1 - v2).abs() < self.tolerance
    }
}

#[derive(Debug, Clone, Copy)]
struct ExactCompare;

impl<T: Eq> Comparable<T> for ExactCompare {
    fn is_equal(&self, v1: &T, v2: &T) -> bool {
        v1 == v2
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
struct Fixure {
    v1: Vec<f64>,
    v2: Vec<f64>,
    similarity: f64,
    bundled: Vec<f64>,
    bound: Vec<f64>,
    inverse: Vec<f64>,
    permuted: Vec<f64>,
}

impl Fixure {
    fn test<V: VectorSymbolicArchitecture, C: Comparable<<V::Storage as Storage>::Primitive>>(
        &self,
        name: &str,
        vsa: &V,
        bundle_deterministic: bool,
        compare: C,
    ) where
        <V::Storage as Storage>::Primitive: PartialOrd + Copy + std::fmt::Display,
    {
        let v1 = Vector::<Dynamic, V>::parse(vsa.clone(), &self.v1).expect("valid vector");
        let v2 = Vector::<Dynamic, V>::parse(vsa.clone(), &self.v2).expect("valid vector");

        let bound_ref =
            Vector::<Dynamic, V>::parse(vsa.clone(), &self.bound).expect("valid vector");
        let permuted_ref =
            Vector::<Dynamic, V>::parse(vsa.clone(), &self.permuted).expect("valid vector");
        let similarity = v1.similarity(&v2);
        let bound = &v1 * &v2;
        let permuted = v1.clone().permute(3); // Arbitrary shift for testing

        assert_eq!(
            similarity - self.similarity < 1e-6,
            true,
            "{}: Similarity does not match reference",
            name
        );

        assert_close(
            name,
            (&bound).into_iter(),
            bound_ref.into_iter(),
            compare,
            "Bound vector does not match reference",
        );
        assert_close(
            name,
            permuted.into_iter(),
            permuted_ref.into_iter(),
            compare,
            "Permuted vector does not match reference",
        );

        if bundle_deterministic {
            let bundled_ref =
                Vector::<Dynamic, V>::parse(vsa.clone(), &self.bundled).expect("valid vector");
            let bundled = v1 + v2;
            assert_close(
                name,
                bundled.into_iter(),
                bundled_ref.into_iter(),
                compare,
                "Bundled vector does not match reference",
            );
        }
    }
}

fn assert_close<
    T: std::fmt::Display,
    C: Comparable<T>,
    I1: ExactSizeIterator<Item = T>,
    I2: ExactSizeIterator<Item = T>,
>(
    name: &str,
    actual: I1,
    expected: I2,
    compare: C,
    message: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", message);
    for (index, (left, right)) in actual.zip(expected).enumerate() {
        assert!(
            compare.is_equal(&left, &right),
            "{}: {}: mismatch at index {} (actual={}, expected={})",
            name,
            message,
            index,
            left,
            right
        );
    }
}

#[test]
fn test_compatibility_with_torchhd() {
    let fixture_data = include_str!("fixtures/reference.json");
    let fixtures: std::collections::HashMap<String, Fixure> =
        serde_json::from_str(fixture_data).expect("Failed to parse fixture data");

    for (arch_name, fixture) in fixtures {
        match arch_name.as_str() {
            "BSC" => {
                let bsc = polars_vsa::architectures::BinarySpatterCode::<u8>::new(42);
                fixture.test(arch_name.as_str(), &bsc, false, ExactCompare);

                let v1 =
                    Vector::<Dynamic, _>::parse(bsc.clone(), &fixture.v1).expect("valid vector");
                let inverse_ref = Vector::<Dynamic, _>::parse(bsc.clone(), &fixture.inverse)
                    .expect("valid inverse vector");
                assert_close(
                    arch_name.as_str(),
                    v1.into_iter(),
                    inverse_ref.into_iter(),
                    ExactCompare,
                    "Inverse vector does not match reference",
                );
            }
            "MAP" => {
                let map = polars_vsa::architectures::MultiplyAddPermute::<u8>::new(42);
                fixture.test(arch_name.as_str(), &map, true, ExactCompare);

                let v1 =
                    Vector::<Dynamic, _>::parse(map.clone(), &fixture.v1).expect("valid vector");
                let inverse_ref = Vector::<Dynamic, _>::parse(map.clone(), &fixture.inverse)
                    .expect("valid inverse vector");
                assert_close(
                    arch_name.as_str(),
                    v1.into_iter(),
                    inverse_ref.into_iter(),
                    ExactCompare,
                    "Inverse vector does not match reference",
                );
            }
            "HRR" => {
                let hrr = polars_vsa::architectures::HolographicReducedRepresentation::<
                    f64,
                    rand::rngs::StdRng,
                >::new(42);
                fixture.test(
                    arch_name.as_str(),
                    &hrr,
                    true,
                    FloatCompare { tolerance: 1e-6 },
                );

                let v1 =
                    Vector::<Dynamic, _>::parse(hrr.clone(), &fixture.v1).expect("valid vector");
                let inverse = -&v1;
                let inverse_ref = Vector::<Dynamic, _>::parse(hrr.clone(), &fixture.inverse)
                    .expect("valid inverse vector");
                assert_close(
                    arch_name.as_str(),
                    inverse.into_iter(),
                    inverse_ref.into_iter(),
                    FloatCompare { tolerance: 1e-6 },
                    "Inverse vector does not match reference",
                );
            }
            "VTB" => {
                let vtb = polars_vsa::architectures::VectorDerivedTransformationBinding::<
                    f64,
                    rand::rngs::StdRng,
                >::new(42);
                fixture.test(
                    arch_name.as_str(),
                    &vtb,
                    true,
                    FloatCompare { tolerance: 0.001 },
                );

                let v1 =
                    Vector::<Dynamic, _>::parse(vtb.clone(), &fixture.v1).expect("valid vector");
                let inverse = -&v1;
                let inverse_ref = Vector::<Dynamic, _>::parse(vtb.clone(), &fixture.inverse)
                    .expect("valid inverse vector");
                assert_close(
                    arch_name.as_str(),
                    inverse.into_iter(),
                    inverse_ref.into_iter(),
                    FloatCompare { tolerance: 0.001 },
                    "Inverse vector does not match reference",
                );
            }
            _ => panic!("Unknown architecture: {}", arch_name),
        }
    }
}
