use super::*;
use crate::architectures::{
    BinarySpatterCode, HolographicReducedRepresentation, MultiplyAddPermute,
    VectorDerivedTransformationBinding,
};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        norm_a += f64::from(*x) * f64::from(*x);
        norm_b += f64::from(*y) * f64::from(*y);
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn nominal(name: &str, module: &str, categories: usize) -> VariableSpec {
    VariableSpec {
        name: name.to_string(),
        module: Some(module.to_string()),
        keys: (0..categories)
            .map(|index| KeySpec {
                label: format!("c{index}"),
                filler: Filler::Symbol(format!("{name}=c{index}")),
            })
            .collect(),
        continuous: None,
    }
}

fn ordinal(name: &str, module: &str, levels: usize) -> VariableSpec {
    VariableSpec {
        name: name.to_string(),
        module: Some(module.to_string()),
        keys: (0..levels)
            .map(|index| KeySpec {
                label: format!("#{index}"),
                filler: Filler::Level {
                    pool: name.to_string(),
                    index,
                    count: levels,
                },
            })
            .collect(),
        continuous: None,
    }
}

/// A specification with `num_variables` variables in 4 modules, alternating nominal and ordinal ones.
fn spec(size: usize, structure: Structure, num_variables: usize) -> EncoderSpec {
    EncoderSpec {
        size,
        structure,
        normalize_modules: true,
        variables: (0..num_variables)
            .map(|index| {
                let module = format!("m{}", index % 4);
                if index % 2 == 0 {
                    nominal(&format!("v{index}"), &module, 5)
                } else {
                    ordinal(&format!("v{index}"), &module, 8)
                }
            })
            .collect(),
    }
}

/// Deterministic pseudo-random codes for `rows` records.
fn codes(spec: &EncoderSpec, rows: usize) -> Vec<Vec<u32>> {
    spec.variables
        .iter()
        .enumerate()
        .map(|(column, variable)| {
            (0..rows)
                .map(|row| ((row * 7 + column * 13 + row * column) % variable.keys.len()) as u32)
                .collect()
        })
        .collect()
}

/// The share of cells decoded correctly by the nearest full key.
fn decoding_accuracy<V>(vsa: V, spec: EncoderSpec, rows: usize) -> f64
where
    V: VectorSymbolicArchitecture + Send + Sync,
    V::Storage: Send + Sync,
{
    let encoder = Encoder::new(vsa, spec.clone()).expect("valid spec");
    let codes = codes(&spec, rows);
    let columns: Vec<Column<'_>> = codes.iter().map(|c| Column::Codes(c)).collect();
    let mut out = vec![0.0f32; rows * spec.size];
    encoder
        .encode_into(&columns, &mut out, true)
        .expect("valid columns");

    let mut correct = 0;
    for (column, variable) in spec.variables.iter().enumerate() {
        let keys: Vec<Vec<f32>> = encoder
            .keys(&variable.name)
            .unwrap()
            .iter()
            .map(|key| encoder.to_dense(key))
            .collect();
        for (row, record) in out.chunks(spec.size).enumerate() {
            let best = keys
                .iter()
                .enumerate()
                .max_by(|a, b| cosine(record, a.1).total_cmp(&cosine(record, b.1)))
                .map(|(index, _)| index as u32);
            if best == Some(codes[column][row]) {
                correct += 1;
            }
        }
    }
    correct as f64 / (rows * spec.variables.len()) as f64
}

#[test]
fn levels_decrease_linearly() {
    let vsa = MultiplyAddPermute::<usize>::new(42);
    let first = level(&vsa, 10_000, "p", 0, 11);
    let first = to_dense::<MultiplyAddPermute<usize>>(&first, 10_000);
    let mut previous = 1.1;
    for index in 0..11 {
        let current =
            to_dense::<MultiplyAddPermute<usize>>(&level(&vsa, 10_000, "p", index, 11), 10_000);
        let similarity = cosine(&first, &current);
        let expected = 1.0 - index as f64 / 10.0;
        assert!(
            (similarity - expected).abs() < 0.05,
            "level {index}: {similarity} vs {expected}"
        );
        assert!(similarity < previous);
        previous = similarity;
    }
}

#[test]
fn levels_work_for_real_valued_architectures() {
    let vsa = HolographicReducedRepresentation::<f64>::new(42);
    let dense = |index| {
        to_dense::<HolographicReducedRepresentation<f64>>(&level(&vsa, 4096, "p", index, 5), 4096)
    };
    let (first, middle, last) = (dense(0), dense(2), dense(4));
    assert!((cosine(&first, &middle) - 0.5).abs() < 0.1);
    assert!(cosine(&first, &last).abs() < 0.1);
}

#[test]
fn fractional_power_follows_kernel() {
    let fft = FftPlanner::<f64>::new().plan_fft_inverse(10_000);
    let sinc = FractionalPower::new(42, "b", Kernel::Sinc, fft.clone());
    let gaussian = FractionalPower::new(42, "b", Kernel::Gaussian, fft);

    let norm: f64 = sinc.dense(0.3).iter().map(|v| f64::from(*v).powi(2)).sum();
    assert!((norm - 1.0).abs() < 1e-4, "not unit norm: {norm}");

    for delta in [0.0f64, 0.25, 0.5, 1.0, 2.0] {
        let expected_sinc = if delta == 0.0 {
            1.0
        } else {
            (std::f64::consts::PI * delta).sin() / (std::f64::consts::PI * delta)
        };
        let actual_sinc = cosine(&sinc.dense(1.0), &sinc.dense(1.0 + delta));
        assert!(
            (actual_sinc - expected_sinc).abs() < 0.05,
            "sinc at {delta}: {actual_sinc} vs {expected_sinc}"
        );

        let expected_gaussian = (-delta * delta / 2.0).exp();
        let actual_gaussian = cosine(&gaussian.dense(-3.0), &gaussian.dense(-3.0 + delta));
        assert!(
            (actual_gaussian - expected_gaussian).abs() < 0.05,
            "gaussian at {delta}: {actual_gaussian} vs {expected_gaussian}"
        );
    }
}

#[test]
fn decoding_recovers_values_for_all_architectures() {
    for structure in [Structure::Flat, Structure::Hierarchical] {
        let accuracy = decoding_accuracy(
            MultiplyAddPermute::<usize>::new(1),
            spec(10_000, structure, 24),
            20,
        );
        assert!(accuracy > 0.95, "MAP {structure:?}: {accuracy}");
        let accuracy = decoding_accuracy(
            BinarySpatterCode::<usize>::new(1),
            spec(10_000, structure, 24),
            20,
        );
        assert!(accuracy > 0.95, "BSC {structure:?}: {accuracy}");
        let accuracy = decoding_accuracy(
            HolographicReducedRepresentation::<f64>::new(1),
            spec(4096, structure, 24),
            10,
        );
        assert!(accuracy > 0.95, "HRR {structure:?}: {accuracy}");
        let accuracy = decoding_accuracy(
            VectorDerivedTransformationBinding::<f64>::new(1),
            spec(4096, structure, 24),
            10,
        );
        assert!(accuracy > 0.95, "VTB {structure:?}: {accuracy}");
    }
}

#[test]
fn capacity_is_limited() {
    // With many bindings per record, decoding degrades for both structures. The module-wise normalization of the
    // hierarchical structure adds quantization noise, so it does not decode better than the flat structure.
    let flat = decoding_accuracy(
        MultiplyAddPermute::<usize>::new(3),
        spec(1_000, Structure::Flat, 200),
        20,
    );
    let hierarchical = decoding_accuracy(
        MultiplyAddPermute::<usize>::new(3),
        spec(1_000, Structure::Hierarchical, 200),
        20,
    );
    assert!(flat < 0.9, "flat should be over capacity: {flat}");
    assert!(
        hierarchical < 0.9,
        "hierarchical should be over capacity: {hierarchical}"
    );
    assert!(hierarchical > 0.2 && flat > 0.2, "far above chance");
}

#[test]
fn omitted_cells_do_not_contribute() {
    let full = spec(2_048, Structure::Hierarchical, 6);
    let mut reduced = full.clone();
    reduced.variables.remove(3);

    let codes_full: Vec<Vec<u32>> =
        vec![vec![1], vec![2], vec![3], vec![OMITTED], vec![4], vec![0]];
    let mut codes_reduced = codes_full.clone();
    codes_reduced.remove(3);

    let encode = |spec: &EncoderSpec, codes: &[Vec<u32>]| {
        let encoder = Encoder::new(MultiplyAddPermute::<usize>::new(5), spec.clone()).unwrap();
        let columns: Vec<Column<'_>> = codes.iter().map(|c| Column::Codes(c)).collect();
        let mut out = vec![0i8; spec.size];
        encoder.encode_into(&columns, &mut out, true).unwrap();
        out
    };
    assert_eq!(encode(&full, &codes_full), encode(&reduced, &codes_reduced));

    let all_omitted = vec![vec![OMITTED]; 6];
    let encoder = Encoder::new(MultiplyAddPermute::<usize>::new(5), full.clone()).unwrap();
    let columns: Vec<Column<'_>> = all_omitted.iter().map(|c| Column::Codes(c)).collect();
    let mut out = vec![1.0f32; full.size];
    encoder.encode_into(&columns, &mut out, true).unwrap();
    assert!(out.iter().all(|v| *v == 0.0));
    assert!(encoder.encode_record(&columns, 0).unwrap().is_none());
}

#[test]
fn hierarchical_without_module_normalization_is_linear() {
    // Without module normalization, the record is the plain sum of the full keys.
    let mut spec = spec(1_024, Structure::Hierarchical, 6);
    spec.normalize_modules = false;
    let encoder = Encoder::new(
        HolographicReducedRepresentation::<f64>::new(5),
        spec.clone(),
    )
    .unwrap();
    let codes: Vec<Vec<u32>> = (0..6).map(|i| vec![i as u32 % 5]).collect();
    let columns: Vec<Column<'_>> = codes.iter().map(|c| Column::Codes(c)).collect();
    let mut out = vec![0.0f32; spec.size];
    encoder.encode_into(&columns, &mut out, false).unwrap();

    let mut expected = vec![0.0f32; spec.size];
    for (variable, code) in spec.variables.iter().zip(&codes) {
        let key = &encoder.keys(&variable.name).unwrap()[code[0] as usize];
        HolographicReducedRepresentation::<f64>::accumulate_dense(key, &mut expected, 1.0);
    }
    for (a, b) in out.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-4);
    }
}

#[test]
fn continuous_values_are_recoverable() {
    let spec = EncoderSpec {
        size: 4_096,
        structure: Structure::Hierarchical,
        normalize_modules: true,
        variables: vec![
            nominal("a", "m", 3),
            VariableSpec {
                name: "x".into(),
                module: Some("m".into()),
                keys: vec![KeySpec {
                    label: "MISSING=refused".into(),
                    filler: Filler::Symbol("MISSING=refused".into()),
                }],
                continuous: Some(ContinuousSpec {
                    base: "x".into(),
                    kernel: Kernel::Gaussian,
                }),
            },
        ],
    };
    let encoder = Encoder::new(
        HolographicReducedRepresentation::<f64>::new(9),
        spec.clone(),
    )
    .unwrap();
    let codes_a = vec![0, 1, 2];
    let codes_x = vec![OMITTED, OMITTED, 0];
    let values = vec![-1.0, 2.0, f64::NAN];
    let columns = [
        Column::Codes(&codes_a),
        Column::Continuous {
            codes: &codes_x,
            values: &values,
        },
    ];
    let mut out = vec![0.0f32; 3 * spec.size];
    encoder.encode_into(&columns, &mut out, true).unwrap();

    let grid: Vec<f64> = (-40..=40).map(|i| f64::from(i) / 10.0).collect();
    let keys: Vec<Vec<f32>> = encoder
        .continuous_keys("x", &grid)
        .unwrap()
        .iter()
        .map(|key| encoder.to_dense(key))
        .collect();
    for (row, expected) in [(0usize, -1.0f64), (1, 2.0)] {
        let record = &out[row * spec.size..(row + 1) * spec.size];
        let (best, _) = grid
            .iter()
            .zip(&keys)
            .max_by(|a, b| cosine(record, a.1).total_cmp(&cosine(record, b.1)))
            .unwrap();
        assert!(
            (best - expected).abs() <= 0.2,
            "row {row}: {best} vs {expected}"
        );
    }

    let missing = encoder.to_dense(&encoder.key("x", "MISSING=refused").unwrap());
    let record = &out[2 * spec.size..];
    assert!(cosine(record, &missing) > 0.3);

    // Fractional power encoding requires real-valued vectors.
    assert!(matches!(
        Encoder::new(MultiplyAddPermute::<usize>::new(9), spec),
        Err(EncodingError::ContinuousUnsupported { .. })
    ));
}

#[test]
fn expressions_resolve_keys_and_symbols() {
    let spec = spec(2_048, Structure::Hierarchical, 4);
    let encoder = Encoder::new(MultiplyAddPermute::<usize>::new(5), spec).unwrap();

    let by_label = encoder.evaluate(&"v0:c2".parse().unwrap()).unwrap();
    let by_symbols = encoder
        .evaluate(&"@m0 * v0 * v0=c2".parse().unwrap())
        .unwrap();
    assert_eq!(by_label.data.0, encoder.key("v0", "c2").unwrap());
    assert_eq!(by_label, by_symbols);

    let level = encoder.evaluate(&"v1#3".parse().unwrap()).unwrap();
    assert_eq!(
        (encoder.codebook()["@m1"].clone() * &encoder.codebook()["v1"]) * &level,
        encoder.evaluate(&"v1:#3".parse().unwrap()).unwrap()
    );

    assert!(matches!(
        encoder.evaluate(&"v0:unknown".parse().unwrap()),
        Err(EncodingError::UnknownValue(_))
    ));
}

#[test]
fn invalid_input_is_rejected() {
    let spec = spec(1_024, Structure::Flat, 2);
    let encoder = Encoder::new(MultiplyAddPermute::<usize>::new(5), spec.clone()).unwrap();
    let valid = vec![0u32];
    let invalid = vec![50u32];
    let mut out = vec![0.0f32; spec.size];
    assert!(matches!(
        encoder.encode_into(&[Column::Codes(&valid)], &mut out, true),
        Err(EncodingError::ColumnCount { .. })
    ));
    assert!(matches!(
        encoder.encode_into(
            &[Column::Codes(&valid), Column::Codes(&invalid)],
            &mut out,
            true
        ),
        Err(EncodingError::CodeOutOfRange { .. })
    ));
    let mut out_i8 = vec![0i8; spec.size];
    assert!(matches!(
        encoder.encode_into(
            &[Column::Codes(&valid), Column::Codes(&valid)],
            &mut out_i8,
            false
        ),
        Err(EncodingError::InvalidOutput(_))
    ));

    let mut duplicate = spec.clone();
    duplicate.variables.push(duplicate.variables[0].clone());
    assert!(matches!(
        Encoder::new(MultiplyAddPermute::<usize>::new(5), duplicate),
        Err(EncodingError::DuplicateVariable(_))
    ));

    let mut mismatch = spec;
    mismatch.variables[1].keys[0].filler = Filler::Level {
        pool: "v1".into(),
        index: 0,
        count: 3,
    };
    mismatch.variables[1].keys[1].filler = Filler::Level {
        pool: "v1".into(),
        index: 1,
        count: 4,
    };
    assert!(matches!(
        Encoder::new(MultiplyAddPermute::<usize>::new(5), mismatch),
        Err(EncodingError::InvalidLevel { .. })
    ));
}

#[test]
fn codebook_is_reproducible() {
    let a = Encoder::new(
        MultiplyAddPermute::<usize>::new(5),
        spec(1_024, Structure::Hierarchical, 8),
    )
    .unwrap();
    let mut reversed = spec(1_024, Structure::Hierarchical, 8);
    reversed.variables.reverse();
    let b = Encoder::new(MultiplyAddPermute::<usize>::new(5), reversed).unwrap();
    assert_eq!(a.codebook().digest(), b.codebook().digest());
    assert_eq!(a.key("v3", "#2").unwrap(), b.key("v3", "#2").unwrap());
}
