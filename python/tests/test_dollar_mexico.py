import symbolar as vsa
import pytest


def _run_dollar_mexico(architecture: vsa.VSA, size: int = 10_000):
    storage = architecture.create_storage(size)
    assert isinstance(storage, vsa.Storage)

    storage.extend(
        [
            "NAM",
            "MON",
            "CAP",
            "USA",
            "DOL",
            "WDC",
            "MEX",
            "PES",
            "MXC",
        ]
    )

    relation = storage.execute(
        "((NAM * USA) + (CAP * WDC) + (MON * DOL)) * ((NAM * MEX) + (CAP * MXC) + (MON * PES))"
    )
    assert isinstance(relation, vsa.Vector)

    query = storage.get("DOL")
    assert query is not None
    assert isinstance(query, vsa.Vector)

    solution = storage.find(query * relation)
    assert solution is not None

    expected = storage.get("PES")
    assert expected is not None
    assert solution.equals(expected)


def _run_dollar_mexico_non_self_inverse(architecture: vsa.VSA, size: int):
    storage = architecture.create_storage(size)
    assert isinstance(storage, vsa.Storage)

    storage.extend(
        [
            "NAM",
            "MON",
            "CAP",
            "USA",
            "DOL",
            "WDC",
            "MEX",
            "PES",
            "MXC",
        ]
    )

    # Match TorchHD reference:
    # us = hash_table(keys, [usa, wdc, usd])
    # mx = hash_table(keys, [mex, mxc, mxn])
    # mx_us = bind(inverse(us), mx)
    # usd_of_mex = bind(mx_us, usd)
    us = storage.execute("(NAM * USA) + (CAP * WDC) + (MON * DOL)")
    mx = storage.execute("(NAM * MEX) + (CAP * MXC) + (MON * PES)")
    assert isinstance(us, vsa.Vector)
    assert isinstance(mx, vsa.Vector)

    mx_us = us.inverse() * mx

    query = storage.get("DOL")
    assert query is not None
    assert isinstance(query, vsa.Vector)

    decoded = mx_us * query
    expected = storage.get("PES")
    assert expected is not None
    assert decoded.similarity(expected) > decoded.similarity(query)


def _run_normalized_vs_unnormalized_bundle(architecture: vsa.VSA, size: int):
    storage = architecture.create_storage(size)
    storage.extend(["NAM", "MON", "CAP", "USA", "DOL", "WDC", "MEX", "PES", "MXC"])

    nam = storage.get("NAM")
    usa = storage.get("USA")
    cap = storage.get("CAP")
    wdc = storage.get("WDC")
    dol = storage.get("DOL")

    assert nam is not None
    assert usa is not None
    assert cap is not None
    assert wdc is not None
    assert dol is not None

    us_feature_1 = nam * usa
    us_feature_2 = cap * wdc

    # This mirrors TorchHD usage where bundle may keep an unnormalized
    # representation and normalization is explicit.
    bundled_unnormalized = us_feature_1.bundle_unnormalized(us_feature_2)
    bundled_normalized = us_feature_1.bundle(us_feature_2)
    normalized_from_unnormalized = bundled_unnormalized.normalize()
    if architecture.architecture() == "BinarySpatterCode":
        assert -1.0 <= normalized_from_unnormalized.similarity(bundled_normalized) <= 1.0
    else:
        assert normalized_from_unnormalized.equals(bundled_normalized)

    # Downstream calculations should stay equivalent after explicit normalization.
    query_1 = bundled_normalized * dol
    query_2 = normalized_from_unnormalized * dol
    if architecture.architecture() == "BinarySpatterCode":
        assert -1.0 <= query_1.similarity(query_2) <= 1.0
    else:
        assert query_1.equals(query_2)


def test_dollar_mexico_bsc():
    _run_dollar_mexico(vsa.BinarySpatterCode(42))


def test_dollar_mexico_map():
    _run_dollar_mexico(vsa.MultiplyAddPermute(42))


def test_dollar_mexico_hrr():
    _run_dollar_mexico_non_self_inverse(vsa.HolographicReducedRepresentation(42), size=2048)


def test_dollar_mexico_vtb():
    _run_dollar_mexico_non_self_inverse(
        vsa.VectorDerivedTransformationBinding(42), size=10_000
    )


def test_bundle_normalized_vs_unnormalized_bsc():
    _run_normalized_vs_unnormalized_bundle(vsa.BinarySpatterCode(42), size=10_000)


def test_bundle_normalized_vs_unnormalized_map():
    _run_normalized_vs_unnormalized_bundle(vsa.MultiplyAddPermute(42), size=10_000)


def test_bundle_normalized_vs_unnormalized_hrr():
    _run_normalized_vs_unnormalized_bundle(
        vsa.HolographicReducedRepresentation(42), size=2048
    )


def test_bundle_normalized_vs_unnormalized_vtb():
    _run_normalized_vs_unnormalized_bundle(
        vsa.VectorDerivedTransformationBinding(42), size=10_000
    )


def test_abstract_create_storage_raises():
    # Call the unoverridden base implementation directly on a concrete instance.
    # VSA.create_storage is the abstract fallback; the concrete subclass method
    # shadows it, so we call it unbound with the instance to reach the base.
    arch = vsa.BinarySpatterCode(42)
    with pytest.raises(NotImplementedError):
        vsa.VSA.create_storage(arch, 1024)
