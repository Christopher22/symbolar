import polars_vsa as vsa
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


def test_abstract_create_storage_raises():
    # Call the unoverridden base implementation directly on a concrete instance.
    # VSA.create_storage is the abstract fallback; the concrete subclass method
    # shadows it, so we call it unbound with the instance to reach the base.
    arch = vsa.BinarySpatterCode(42)
    with pytest.raises(NotImplementedError):
        vsa.VSA.create_storage(arch, 1024)
