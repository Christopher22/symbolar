use polars_vsa::{
    Expression, Fixed, Storage,
    architectures::{BinarySpatterCode, SelfInverseVectorSymbolicArchitecture},
};

fn test_dollar_mexico<V: SelfInverseVectorSymbolicArchitecture>(v: V) {
    let mut storage = Storage::new(v, Fixed::<10000>);
    storage.extend([
        "NAM", "MON", "CAP", // Features
        "USA", "DOL", "WDC", // USA
        "MEX", "PES", "MXC", // Mexico
    ]);

    let expression: Expression =
        "((NAM * USA) + (CAP * WDC) + (MON * DOL)) * ((NAM * MEX) + (CAP * MXC) + (MON * PES))"
            .parse()
            .expect("valid expression");
    let knowledge_base_2 = storage.execute(&expression).expect("valid execution");

    let query = &storage["DOL"];
    let solution = storage
        .find(&(query * knowledge_base_2.as_ref()), &())
        .expect("at least one vector");

    assert_eq!(&storage[solution], &storage["PES"]);
}

#[test]
fn test_dollar_mexico_bsc() {
    test_dollar_mexico(BinarySpatterCode::<usize>::new(42));
}

#[test]
fn test_dollar_mexico_map() {
    test_dollar_mexico(polars_vsa::architectures::MultiplyAddPermute::<usize>::new(
        42,
    ));
}
