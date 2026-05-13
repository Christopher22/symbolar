use polars_vsa::{
    Expression, Fixed, Size, Storage,
    architectures::{
        BinarySpatterCode, NonSelfInverseVectorSymbolicArchitecture,
        SelfInverseVectorSymbolicArchitecture, VectorSymbolicArchitecture,
    },
};

fn create_storage<S: Size, V: VectorSymbolicArchitecture>(v: V, size: S) -> Storage<S, V> {
    let mut storage = Storage::new(v, size).expect("valid size");
    storage.extend([
        "NAM", "MON", "CAP", // Features
        "USA", "DOL", "WDC", // USA
        "MEX", "PES", "MXC", // Mexico
    ]);

    storage
}

fn test_dollar_mexico_self_inverse<V: SelfInverseVectorSymbolicArchitecture>(v: V) {
    let storage = create_storage(v, Fixed::<100000>);

    let expression: Expression =
        "((NAM * USA) + (CAP * WDC) + (MON * DOL)) * ((NAM * MEX) + (CAP * MXC) + (MON * PES))"
            .parse()
            .expect("valid expression");
    let knowledge_base = storage
        .execute(&expression)
        .expect("valid execution")
        .into_owned();

    let query = &storage["DOL"];
    let solution = storage
        .find(&(query * &knowledge_base), &())
        .expect("at least one vector");

    assert_eq!(&storage[solution], &storage["PES"]);
}

fn test_dollar_mexico_non_self_inverse<
    const N: usize,
    V: NonSelfInverseVectorSymbolicArchitecture,
>(
    v: V,
) {
    let storage = create_storage(v, Fixed::<N>);

    // Match TorchHD reference:
    // us = hash_table(keys, [usa, wdc, usd])
    // mx = hash_table(keys, [mex, mxc, mxn])
    // mx_us = bind(inverse(us), mx)
    // usd_of_mex = bind(mx_us, usd)
    let us: Expression = "(NAM * USA) + (CAP * WDC) + (MON * DOL)"
        .parse()
        .expect("valid us expression");
    let mx: Expression = "(NAM * MEX) + (CAP * MXC) + (MON * PES)"
        .parse()
        .expect("valid mx expression");

    let us = storage.execute(&us).expect("valid execution").into_owned();
    let mx = storage.execute(&mx).expect("valid execution").into_owned();
    let mx_us = -&us * &mx;

    let query = &storage["DOL"];
    let decoded = &mx_us * query;

    // TorchHD reference evaluates cosine similarities against memory vectors.
    let pes_similarity = decoded.similarity(&storage["PES"]);
    let dol_similarity = decoded.similarity(&storage["DOL"]);

    assert!(
        pes_similarity > dol_similarity,
        "expected PES similarity ({pes_similarity}) to exceed DOL similarity ({dol_similarity})"
    );
}

#[test]
fn test_dollar_mexico_bsc() {
    test_dollar_mexico_self_inverse(BinarySpatterCode::<usize>::new(42));
}

#[test]
fn test_dollar_mexico_map() {
    test_dollar_mexico_self_inverse(polars_vsa::architectures::MultiplyAddPermute::<usize>::new(
        42,
    ));
}

#[test]
fn test_dollar_mexico_hrr() {
    test_dollar_mexico_non_self_inverse::<2048, _>(
        // HRR binding is O(n^2), so keep dimensions practical for tests.
        // This still provides stable retrieval while avoiding very long runtime.
        polars_vsa::architectures::HolographicReducedRepresentation::<f64>::new(42),
    );
}

#[test]
fn test_dollar_mexico_vtb() {
    test_dollar_mexico_non_self_inverse::<10000, _>(
        polars_vsa::architectures::VectorDerivedTransformationBinding::<f64>::new(42),
    );
}
