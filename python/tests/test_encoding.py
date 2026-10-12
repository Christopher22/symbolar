import json

import numpy as np
import polars as pl
import pytest

import symbolar as sb

SMOKING = pl.Enum(["never", "former", "current"])
MISSING = sb.MissingCodes({8888: "refused", 9999: "dont_know", 7775: "structural"}, {"refused": "encode"})


def synthetic(n: int, seed: int = 0) -> pl.DataFrame:
    """A small table with every kind of variable, including missing-value codes."""
    rng = np.random.default_rng(seed)
    mi = rng.choice([0, 1, 8888], n, p=[0.8, 0.15, 0.05])
    return pl.DataFrame(
        {
            "sex": rng.choice(["f", "m"], n),
            "smoking": pl.Series(rng.choice(["never", "former", "current"], n)).cast(SMOKING),
            "isced": rng.integers(0, 7, n),
            "bmi": rng.normal(27, 4, n),
            "diabetes": rng.random(n) < 0.15,
            "pm25": rng.normal(12, 3, n),
            "income": np.where(rng.random(n) < 0.1, 8888, rng.integers(1, 6, n)),
            "mi": mi,
            "mi_dt": np.where(mi == 1, rng.integers(0, 30, n), np.nan),
            "icd": rng.choice(["I21.0", "I21.4", "I25.1", "C50.9", "E11.9"], n),
        },
    ).with_columns(pl.col("mi_dt").fill_nan(None))


MODULES = {
    "sex": "basis",
    "isced": "basis",
    "income": "basis",
    "smoking": "tabak",
    "bmi": "anthro",
    "diabetes": "hke",
    "icd": "hke",
    "herzinfarkt": "hke",
    "pm25": "geo",
}


def schema() -> sb.Schema:
    return sb.Schema.infer(
        synthetic(10),
        overrides={
            "isced": sb.Ordinal("isced"),
            "income": sb.Ordinal("income", levels=[1, 2, 3, 4, 5]),
            "herzinfarkt": sb.TimedEvent("herzinfarkt", indicator="mi", delta="mi_dt"),
            "icd": sb.HierarchicalCode("icd", hierarchy=sb.icd10_hierarchy, pool="icd10"),
        },
        modules=MODULES,
        missing=MISSING,
    )


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return synthetic(1500)


@pytest.fixture(scope="module")
def encoder(data: pl.DataFrame) -> sb.Encoder:
    return sb.Encoder(sb.MultiplyAddPermute(42), 10_000, schema()).fit(data[:1000])


# ── Schema ──────────────────────────────────────────────────────────────────


def test_inference_follows_dtypes():
    df = pl.DataFrame(
        {
            "e": pl.Series(["a", "b"]).cast(pl.Enum(["a", "b", "c"])),
            "c": pl.Series(["a", "b"]).cast(pl.Categorical),
            "s": ["x", "y"],
            "b": [True, False],
            "i": [1, 2],
            "f": [1.0, 2.0],
        }
    )
    inferred = sb.Schema.infer(df)
    assert inferred["e"] == sb.Nominal("e", categories=["a", "b", "c"])
    assert isinstance(inferred["c"], sb.Nominal) and isinstance(inferred["s"], sb.Nominal)
    assert isinstance(inferred["b"], sb.Binary)
    assert inferred["i"] == sb.Numeric("i") and inferred["f"] == sb.Numeric("f")
    assert set(inferred.describe()["source"]) == {"inferred"}

    with pytest.raises(TypeError, match="d"):
        sb.Schema.infer(pl.DataFrame({"d": [pl.date(2020, 1, 1)]}).select(pl.col("d")))
    assert len(sb.Schema.infer(pl.DataFrame({"d": [pl.datetime(2020, 1, 1)]}).select(pl.col("d")), ignore=["d"])) == 0


def test_overrides_take_precedence():
    df = pl.DataFrame({"x": [1, 2, 3], "y": [1, 1, 2], "q": [0.1, 0.2, 0.3], "drop": [1, 2, 3]})
    inferred = sb.Schema.infer(
        df,
        overrides={
            "x": sb.Ordinal("x"),
            "q": sb.Levels(8),
            "y": {"module": "m", "pool": "ja_nein"},
            "drop": None,
        },
        modules=lambda name: "default",
        integer_as_nominal=2,
    )
    assert isinstance(inferred["x"], sb.Ordinal)
    assert inferred["q"] == sb.Numeric("q", sb.Levels(8), module="default")
    assert inferred["y"] == sb.Nominal("y", module="m", pool="ja_nein")
    assert "drop" not in inferred
    sources = dict(inferred.describe().select("variable", "source").iter_rows())
    assert sources == {"x": "override", "y": "override", "q": "override"}


def test_variables_consume_their_columns():
    inferred = schema()
    assert "herzinfarkt" in inferred and "mi" not in inferred and "mi_dt" not in inferred
    assert inferred["herzinfarkt"].module == "hke"


def test_explicit_schema_without_data():
    explicit = sb.Schema(
        [
            sb.Nominal("smoking", ["never", "former", "current"], module="tabak"),
            sb.Numeric("bmi", sb.Levels(32, low=15, high=45)),
        ],
        missing=MISSING,
    )
    description = explicit.describe()
    assert description["type"].to_list() == ["nominal", "numeric"]
    assert description["missing_encoded"].to_list() == ["refused", "refused"]
    with pytest.raises(ValueError, match="twice"):
        sb.Schema([sb.Nominal("a"), sb.Numeric("a")])


# ── Fitting and encoding ────────────────────────────────────────────────────


def test_fit_only_uses_training_data(data: pl.DataFrame):
    base = sb.Encoder(sb.MultiplyAddPermute(1), 1024, sb.Schema([sb.Numeric("bmi", sb.Levels(16))]))
    train = data[:500]
    fitted = base.fit(train)
    outliers = pl.DataFrame({"bmi": [100.0] * 50})
    refitted = base.fit(pl.concat([data[:500].select("bmi"), outliers]))

    # Transforming other data never changes the fitted levels ...
    before = fitted.transform(train)
    fitted.transform(outliers)
    assert np.array_equal(before, fitted.transform(train))
    # ... while fitting on other data does.
    assert fitted._index["bmi"].levels != refitted._index["bmi"].levels
    assert not base.is_fitted


def test_transform_shapes_and_types(encoder: sb.Encoder, data: pl.DataFrame):
    X = encoder.transform(data[:10])
    assert X.shape == (10, 10_000) and X.dtype == np.int8
    assert set(np.unique(X)) <= {-1, 1}
    raw = encoder.transform(data[:10], normalize=False)
    assert raw.dtype == np.float32
    assert np.array_equal(np.where(raw >= 0, 1, -1), X)
    with pytest.raises(ValueError, match="integer output"):
        encoder.transform(data[:10], normalize=False, dtype="int8")

    batches = list(encoder.transform_batches(data[:25], 10))
    assert [batch.shape[0] for batch in batches] == [10, 10, 5]
    assert np.array_equal(np.vstack(batches), encoder.transform(data[:25]))


def test_empty_records_are_zero():
    encoder = sb.Encoder(sb.MultiplyAddPermute(1), 1024, sb.Schema([sb.Nominal("a", ["x", "y"])])).fit(
        pl.DataFrame({"a": ["x", "y"]})
    )
    X = encoder.transform(pl.DataFrame({"a": [None, "x", "unknown"]}, schema={"a": pl.String}))
    assert not X[0].any() and X[1].any() and not X[2].any()


def test_decoding_recovers_values(encoder: sb.Encoder, data: pl.DataFrame):
    test = data[1000:1200]
    decoded = encoder.decode(encoder.transform(test))
    truth = encoder.codes(test)
    for variable in ["sex", "smoking", "isced", "diabetes", "icd", "herzinfarkt", "income"]:
        labels = np.array(encoder.labels(variable), dtype=object)
        codes = truth[variable].to_numpy()
        observed = ~np.isnan(codes.astype(float))
        expected = labels[codes[observed].astype(int)]
        actual = decoded.filter(pl.col("variable") == variable).sort("row")["label"].to_numpy()[observed]
        assert np.mean(actual == expected) > 0.9, variable

    bmi = decoded.filter(pl.col("variable") == "bmi").sort("row")["value"].to_numpy()
    low, high, count = encoder._index["bmi"].levels
    assert np.mean(np.abs(bmi - test["bmi"].clip(low, high).to_numpy()) <= (high - low) / (count - 1)) > 0.9


def test_missing_policy(data: pl.DataFrame):
    test = pl.DataFrame({"income": [8888, 3, 7775]})
    variables = [sb.Ordinal("income", levels=[1, 2, 3, 4, 5]), sb.Nominal("other", ["a"])]
    frame = test.with_columns(other=pl.lit("a"))

    encoded = sb.Encoder(sb.MultiplyAddPermute(1), 10_000, sb.Schema(variables, MISSING)).fit(frame)
    assert encoded.labels("income")[-1] == "MISSING=refused"
    decoded = encoded.decode(encoded.transform(frame), ["income"])["label"].to_list()
    assert decoded[:2] == ["MISSING=refused", "3"]

    omitted = sb.Encoder(sb.MultiplyAddPermute(1), 10_000, sb.Schema(variables, MISSING.with_policy("omit"))).fit(frame)
    X = omitted.transform(frame)
    only_other = omitted.transform(frame.with_columns(income=pl.lit(None, pl.Int64)))
    assert np.array_equal(X[0], only_other[0]) and np.array_equal(X[2], only_other[2])
    assert not np.array_equal(X[1], only_other[1])


def test_continuous_requires_real_valued_architecture(data: pl.DataFrame):
    variables = sb.Schema([sb.Numeric("pm25", sb.FractionalPower(0.5))])
    with pytest.raises(ValueError, match="FractionalPower"):
        sb.Encoder(sb.MultiplyAddPermute(1), 1024, variables).fit(data)

    encoder = sb.Encoder(sb.HolographicReducedRepresentation(1), 4096, variables).fit(data)
    X = encoder.transform(data[:100])
    assert X.dtype == np.float32
    decoded = encoder.decode(X, grid_size=201).sort("row")["value"].to_numpy()
    assert np.mean(np.abs(decoded - data["pm25"][:100].to_numpy()) < 1.0) > 0.9


def test_vtb_suggests_valid_sizes():
    with pytest.raises(ValueError, match="perfect square, e.g. 4900 or 5041"):
        sb.Encoder(sb.VectorDerivedTransformationBinding(1), 5000, sb.Schema([sb.Nominal("a")])).fit(
            pl.DataFrame({"a": ["x"]})
        )


# ── Expressions, queries and explanations ───────────────────────────────────


def test_expressions(encoder: sb.Encoder):
    by_label = encoder.vector("smoking:current")
    by_symbols = encoder.vector("@tabak * smoking * smoking=current")
    assert np.array_equal(by_label, by_symbols)
    assert encoder.rewrite("isced>=5") == "(`isced:5` + `isced:6`)"
    assert encoder.rewrite("herzinfarkt<1 * x") == "`herzinfarkt:yes#0` * x"
    assert encoder.rewrite("herzinfarkt<=1") == "(`herzinfarkt:yes#0` + `herzinfarkt:yes#1`)"
    assert encoder.rewrite("`bmi>=3`") == "`bmi>=3`"
    with pytest.raises(ValueError, match="ranges require"):
        encoder.rewrite("sex>=1")


def test_queries_match_exact_filters(encoder: sb.Encoder, data: pl.DataFrame):
    X = encoder.transform(data)
    threshold = encoder.null_threshold(X[:1000])
    result = encoder.query(X, ["smoking:current", "diabetes:true", "sex:f"], threshold)
    exact = ((data["smoking"] == "current") & data["diabetes"] & (data["sex"] == "f")).to_numpy()
    true_positives = np.sum(result.mask & exact)
    assert true_positives / max(result.mask.sum(), 1) >= 0.9
    assert true_positives / exact.sum() >= 0.9

    exact_range = ((data["bmi"] >= 30) & (data["smoking"] == "current")).to_numpy()
    decoded = encoder.query(X, ["bmi>=30", "smoking:current"], threshold, ranges="decode")
    true_positives = np.sum(decoded.mask & exact_range)
    assert true_positives / max(decoded.mask.sum(), 1) >= 0.9
    assert true_positives / exact_range.sum() >= 0.9


def test_explanations_recover_linear_effects(data: pl.DataFrame):
    larger = synthetic(3000, seed=1)
    encoder = sb.Encoder(sb.MultiplyAddPermute(7), 4096, schema()).fit(larger)
    X = encoder.transform(larger).astype(np.float32)
    y = (
        2.0 * (larger["smoking"] == "current").to_numpy()
        - 1.5 * larger["diabetes"].to_numpy()
        + 0.1 * np.random.default_rng(0).normal(size=larger.height)
    )
    # Ridge regression in its dual form, as d > n.
    gram = X @ X.T
    weights = X.T @ np.linalg.solve(gram + 100.0 * np.eye(len(gram)), y - y.mean())

    contributions = encoder.explain(weights)
    smoking = dict(contributions.filter(pl.col("variable") == "smoking").select("label", "contribution").iter_rows())
    diabetes = dict(contributions.filter(pl.col("variable") == "diabetes").select("label", "contribution").iter_rows())
    assert smoking["current"] > smoking["former"] and smoking["current"] > smoking["never"]
    assert diabetes["true"] < diabetes["false"]

    rows = encoder.explain_rows(weights, larger[:3], X[:3])
    assert set(rows["row"]) == {0, 1, 2}
    assert rows.filter(pl.col("variable") == "<residual>").height == 3
    total = rows.group_by("row").agg(pl.col("contribution").sum()).sort("row")["contribution"].to_numpy()
    assert np.allclose(total, X[:3] @ weights, atol=1e-3)


# ── Reproducibility ─────────────────────────────────────────────────────────


def test_round_trip_and_digest(encoder: sb.Encoder, data: pl.DataFrame):
    restored = sb.Encoder.from_json(encoder.to_json())
    assert restored.digest() == encoder.digest()
    assert np.array_equal(restored.transform(data[:20]), encoder.transform(data[:20]))

    state = json.loads(encoder.to_json())
    state["seed"] = 43
    with pytest.raises(ValueError, match="digest"):
        sb.Encoder.from_json(json.dumps(state))

    reordered = sb.Schema(list(reversed(schema().variables)), MISSING)
    again = sb.Encoder(sb.MultiplyAddPermute(42), 10_000, reordered).fit(data[:1000])
    assert again.digest() == encoder.digest()
    assert np.array_equal(again.keys("bmi"), encoder.keys("bmi"))


def test_icd10_hierarchy():
    assert sb.icd10_hierarchy("I21.4") == ["chapter:IX", "I21"]
    assert sb.icd10_hierarchy("i21") == ["chapter:IX"]
    assert sb.icd10_hierarchy("S72.0") == ["chapter:XIX", "S72"]
    assert sb.icd10_hierarchy("unknown") == []
    assert sb.icd10_chapter("C50.9") == "II"


def test_codes_share_ancestors(encoder: sb.Encoder):
    keys = encoder.keys("icd")
    labels = encoder.labels("icd")
    similarity = lambda a, b: float(keys[labels.index(a)] @ keys[labels.index(b)]) / keys.shape[1]
    assert similarity("I21.0", "I21.4") > similarity("I21.0", "I25.1") > similarity("I21.0", "C50.9")
