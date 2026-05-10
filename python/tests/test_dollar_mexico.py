import polars_vsa as vsa
import pytest


def _run_dollar_mexico(architecture: vsa.VSA):
    storage = architecture.create_storage(10_000)
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


def test_dollar_mexico_bsc():
    _run_dollar_mexico(vsa.BinarySpatterCode(42))


def test_dollar_mexico_map():
    _run_dollar_mexico(vsa.MultiplyAddPermute(42))


def test_abstract_create_storage_raises():
    # Call the unoverridden base implementation directly on a concrete instance.
    # VSA.create_storage is the abstract fallback; the concrete subclass method
    # shadows it, so we call it unbound with the instance to reach the base.
    arch = vsa.BinarySpatterCode(42)
    with pytest.raises(NotImplementedError):
        vsa.VSA.create_storage(arch, 1024)
