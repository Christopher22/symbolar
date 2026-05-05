use bitvec::prelude::{BitVec, Lsb0};
use polars_vsa::architectures::{
    BinarySpatterCode, MultiplyAddPermute, VectorSymbolicArchitecture,
};
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Clone)]
struct ArchitectureReference {
    bundle_input_a: Vec<f64>,
    bundle_input_b: Vec<f64>,
    bind_input_a: Vec<f64>,
    bind_input_b: Vec<f64>,
    bundle: Vec<f64>,
    bind: Vec<f64>,
    permute_by: usize,
    permute: Vec<f64>,
    inverse: Vec<f64>,
    cosine_similarity: f64,
}

#[derive(Debug, Clone)]
struct Fixture {
    torchhd_version: String,
    architectures: HashMap<String, ArchitectureReference>,
}

fn parse_fixture() -> Fixture {
    let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("torchhd_reference.json");
    let content = fs::read_to_string(&fixture_path)
        .unwrap_or_else(|err| panic!("failed to read fixture {}: {err}", fixture_path.display()));

    let root: serde_json::Value = serde_json::from_str(&content)
        .unwrap_or_else(|err| panic!("failed to parse fixture {}: {err}", fixture_path.display()));

    let torchhd_version = root
        .get("torchhd_version")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();

    let mut architectures = HashMap::new();
    let arch_obj = root
        .get("architectures")
        .and_then(serde_json::Value::as_object)
        .expect("missing architectures object");

    for (name, value) in arch_obj {
        let get_number_list = |key: &str| {
            value
                .get(key)
                .and_then(serde_json::Value::as_array)
                .unwrap_or_else(|| panic!("missing array field: {key}"))
                .iter()
                .map(|v| {
                    v.as_f64()
                        .unwrap_or_else(|| panic!("array field {key} contains non-number value"))
                })
                .collect::<Vec<f64>>()
        };
        let get_usize = |key: &str| {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(|| panic!("missing numeric field: {key}")) as usize
        };
        let get_f64 = |key: &str| {
            value
                .get(key)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or_else(|| panic!("missing float field: {key}"))
        };

        let item = ArchitectureReference {
            bundle_input_a: get_number_list("bundle_input_a"),
            bundle_input_b: get_number_list("bundle_input_b"),
            bind_input_a: get_number_list("bind_input_a"),
            bind_input_b: get_number_list("bind_input_b"),
            bundle: get_number_list("bundle"),
            bind: get_number_list("bind"),
            permute_by: get_usize("permute_by"),
            permute: get_number_list("permute"),
            inverse: get_number_list("inverse"),
            cosine_similarity: get_f64("cosine_similarity"),
        };
        architectures.insert(name.clone(), item);
    }

    Fixture {
        torchhd_version,
        architectures,
    }
}

fn bitvec_from_numeric(values: &[f64]) -> BitVec<u8, Lsb0> {
    values.iter().map(|v| *v > 0.0).collect()
}

fn bsc_numeric_from_bitvec(values: &BitVec<u8, Lsb0>) -> Vec<f64> {
    values.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect()
}

fn map_numeric_from_bitvec(values: &BitVec<u8, Lsb0>) -> Vec<f64> {
    values.iter().map(|v| if *v { 1.0 } else { -1.0 }).collect()
}

fn assert_numeric_vectors_equal(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "vector lengths differ");
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-12, "mismatch at index {idx}: {a} != {e}");
    }
}

fn assert_architecture_matches_reference<V>(
    fixture: &Fixture,
    architecture_key: &str,
    vsa: &V,
    storage_to_numeric: fn(&BitVec<u8, Lsb0>) -> Vec<f64>,
) where
    V: VectorSymbolicArchitecture<Storage = BitVec<u8, Lsb0>>,
{
    let reference = fixture
        .architectures
        .get(architecture_key)
        .unwrap_or_else(|| panic!("missing {architecture_key} architecture fixture"));

    let bundle_a = bitvec_from_numeric(&reference.bundle_input_a);
    let bundle_b = bitvec_from_numeric(&reference.bundle_input_b);
    let bind_a = bitvec_from_numeric(&reference.bind_input_a);
    let bind_b = bitvec_from_numeric(&reference.bind_input_b);

    let bundled = vsa.bundle(&bundle_a, &bundle_b);
    let bound = V::bind(&bind_a, &bind_b);
    let permuted = V::permute(&bind_a, reference.permute_by);
    let inverse = V::inverse(&bind_a);
    let cosine = V::cosine_similarity(&bind_a, &bind_b);

    assert_numeric_vectors_equal(&storage_to_numeric(&bundled), &reference.bundle);
    assert_numeric_vectors_equal(&storage_to_numeric(&bound), &reference.bind);
    assert_numeric_vectors_equal(&storage_to_numeric(&permuted), &reference.permute);
    assert_numeric_vectors_equal(&storage_to_numeric(&inverse), &reference.inverse);
    assert!(
        (cosine - reference.cosine_similarity).abs() < 1e-12,
        "torchhd version: {}",
        fixture.torchhd_version
    );
}

#[test]
fn bsc_matches_torchhd_reference() {
    let fixture = parse_fixture();
    let bsc = BinarySpatterCode::<u8>::new(42);
    assert_architecture_matches_reference(&fixture, "bsc", &bsc, bsc_numeric_from_bitvec);
}

#[test]
fn map_matches_torchhd_reference() {
    let fixture = parse_fixture();
    let map = MultiplyAddPermute::<u8>::new(42);
    assert_architecture_matches_reference(&fixture, "map", &map, map_numeric_from_bitvec);
}
