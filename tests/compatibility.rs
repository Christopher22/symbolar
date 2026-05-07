use polars_vsa::architectures::{PrimaryStorage, VectorSymbolicArchitecture};

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
struct Fixure {
    v1: Vec<f64>,
    v2: Vec<f64>,
    similarity: f64,
    bundled: Vec<f64>,
    bound: Vec<f64>,
    permuted: Vec<f64>,
}

impl Fixure {
    fn test<V: VectorSymbolicArchitecture>(&self, vsa: &V, bundle_deterministic: bool) {
        let v1 = V::Storage::parse(&self.v1);
        let v2 = V::Storage::parse(&self.v2);

        let bound_ref = V::Storage::parse(&self.bound);
        let permuted_ref = V::Storage::parse(&self.permuted);
        let similarity = V::similarity(&v1, &v2);
        let bound = V::bind(&v1, &v2);
        let permuted = V::permute(&v1, 3); // Arbitrary shift for testing

        assert_eq!(
            similarity - self.similarity < 1e-6,
            true,
            "Similarity does not match reference"
        );
        assert_eq!(bound, bound_ref, "Bound vector does not match reference");
        assert_eq!(
            permuted, permuted_ref,
            "Permuted vector does not match reference"
        );

        if bundle_deterministic {
            let bundled_ref = V::Storage::parse(&self.bundled);
            let bundled = vsa.bundle(&v1, &v2);
            assert_eq!(
                bundled, bundled_ref,
                "Bundled vector does not match reference"
            );
        }
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
                fixture.test(&bsc, false);
            }
            "MAP" => {
                let map = polars_vsa::architectures::MultiplyAddPermute::<u8>::new(42);
                fixture.test(&map, true);
            }
            _ => panic!("Unknown architecture: {}", arch_name),
        }
    }
}
