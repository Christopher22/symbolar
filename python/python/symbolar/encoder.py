"""Encoding of tables into hypervectors, and reading them out again.

The :class:`Encoder` combines a :class:`~symbolar.Schema` with an architecture and a dimensionality. Fitting it on
the training data resolves all data-dependent decisions (categories, level boundaries, scaling); the codebook itself
only depends on the names and the seed of the architecture. Encoding is done in Rust; Polars expressions translate
the table into the codes of each variable beforehand.

Records are encoded as ``P = Σ_m M_m ⊛ norm(Σ_{v ∈ m} R_v ⊛ Φ_v(x_v))`` (``structure="hierarchical"``) or
``P = Σ_v R_v ⊛ Φ_v(x_v)`` (``structure="flat"``). Since binding distributes over bundling and the (approximate)
inverses of all architectures are the adjoints of their binding, the similarity of ``P`` to a full key
``M_m ⊛ R_v ⊛ Φ_v(k)`` equals the similarity of the unbound ``P ⊘ M_m ⊘ R_v`` to ``Φ_v(k)`` up to a factor that is
the same for all ``k``. Decoding, querying and explaining therefore use the full keys returned by :meth:`Encoder.keys`.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

from . import _native
from .schema import (
    Binary,
    FractionalPower,
    HierarchicalCode,
    Levels,
    MissingCodes,
    Nominal,
    Numeric,
    Ordinal,
    Schema,
    TimedEvent,
    Variable,
)

__all__ = ["Encoder", "QueryResult"]

_ARCHITECTURES = {
    "BinarySpatterCode": _native.BinarySpatterCode,
    "MultiplyAddPermute": _native.MultiplyAddPermute,
    "HolographicReducedRepresentation": _native.HolographicReducedRepresentation,
    "VectorDerivedTransformationBinding": _native.VectorDerivedTransformationBinding,
}
_REAL_VALUED = {"HolographicReducedRepresentation", "VectorDerivedTransformationBinding"}
_FORMAT_VERSION = 1

Structure = Literal["hierarchical", "flat"]


# ─────────────────────────────────────────────────────────────────────────────
# Fitted variables
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Fitted:
    """A variable with all data-dependent decisions resolved. Plain data, so that it can be stored as JSON."""

    name: str
    kind: Literal["categorical", "levels", "continuous", "event", "code"]
    module: str | None
    pool: str
    columns: list[str]
    #: The values a cell is compared with, one per value key (categorical, code).
    categories: list[Any] = field(default_factory=list)
    #: The labels of the value keys, in the order of their codes.
    labels: list[str] = field(default_factory=list)
    #: The fillers of the value keys, in the format of the native specification.
    fillers: list[dict[str, Any]] = field(default_factory=list)
    #: The number represented by each value key, if any.
    values: list[float | None] = field(default_factory=list)
    #: Missing-value codes as ``[code, category]`` pairs and the categories bound to a filler.
    missing_codes: list[list[Any]] = field(default_factory=list)
    missing_encoded: list[str] = field(default_factory=list)
    #: Levels: ``[low, high, count]`` (also the time since an event).
    levels: list[float] | None = None
    #: Continuous: center, bandwidth and kernel of the fractional power encoding, and the range of the training data.
    center: float = 0.0
    bandwidth: float = 1.0
    kernel: str = "sinc"
    observed_range: list[float] | None = None
    #: Events: the values of the indicator meaning that the event happened.
    positive: list[Any] = field(default_factory=list)

    @property
    def missing(self) -> MissingCodes:
        return MissingCodes(
            {code: category for code, category in self.missing_codes},
            {category: "encode" for category in self.missing_encoded},
        )

    def all_labels(self) -> list[str]:
        return self.labels + [f"MISSING={category}" for category in self.missing_encoded]

    def spec(self) -> dict[str, Any]:
        keys = [{"label": label, "filler": filler} for label, filler in zip(self.labels, self.fillers)]
        keys += [
            {"label": f"MISSING={category}", "filler": {"symbol": f"MISSING={category}"}}
            for category in self.missing_encoded
        ]
        continuous = None
        if self.kind == "continuous":
            continuous = {"base": f"{self.pool}~", "kernel": self.kernel}
        return {"name": self.name, "module": self.module, "keys": keys, "continuous": continuous}

    # ── Levels ────────────────────────────────────────────────────────────

    def level_expr(self, expr: pl.Expr) -> pl.Expr:
        assert self.levels is not None
        low, high, count = self.levels
        if high <= low or count < 2:
            return pl.when(expr.is_not_null()).then(pl.lit(0, pl.UInt32))
        position = ((expr.cast(pl.Float64) - low) / (high - low) * (count - 1)).round()
        return position.clip(0, count - 1).cast(pl.UInt32)

    def scale(self, value: float) -> float:
        return (value - self.center) / self.bandwidth

    def unscale(self, value: float) -> float:
        return value * self.bandwidth + self.center


def _label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _plain(value: Any) -> Any:
    """Convert NumPy and other scalars into JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _missing_category(expr: pl.Expr, dtype: pl.DataType, missing: MissingCodes) -> pl.Expr | None:
    """The missing-value category of each cell, or null."""
    if not missing.codes or dtype == pl.Boolean:
        return None
    if dtype.is_numeric():
        pairs = [(float(code), category) for code, category in missing.codes.items() if _is_number(code)]
        if not pairs:
            return None
        return expr.cast(pl.Float64).replace_strict(
            [code for code, _ in pairs], [category for _, category in pairs], default=None, return_dtype=pl.String
        )
    pairs = [(_label(code), category) for code, category in missing.codes.items()]
    return expr.cast(pl.String).replace_strict(
        [code for code, _ in pairs], [category for _, category in pairs], default=None, return_dtype=pl.String
    )


def _match(expr: pl.Expr, dtype: pl.DataType, categories: Sequence[Any]) -> pl.Expr:
    """The index of each cell within ``categories``, or null."""
    indices = list(range(len(categories)))
    if dtype == pl.Boolean and all(isinstance(value, bool) for value in categories):
        return expr.replace_strict(list(categories), indices, default=None, return_dtype=pl.UInt32)
    if dtype.is_numeric() and all(_is_number(value) for value in categories):
        return expr.cast(pl.Float64).replace_strict(
            [float(value) for value in categories], indices, default=None, return_dtype=pl.UInt32
        )
    return expr.cast(pl.String).replace_strict(
        [_label(value) for value in categories], indices, default=None, return_dtype=pl.UInt32
    )


class _Cells:
    """A source column split into its missing-value category and its cleaned values."""

    def __init__(self, column: str, dtype: pl.DataType, missing: MissingCodes) -> None:
        self.dtype = dtype
        raw = pl.col(column)
        self.category = _missing_category(raw, dtype, missing)
        self.values = raw if self.category is None else pl.when(self.category.is_null()).then(raw)

    def code(self, fitted: _Fitted, value_code: pl.Expr) -> pl.Expr:
        """Combine the code of the values with the codes of the encoded missing-value categories."""
        if self.category is None:
            return value_code
        encoded = fitted.missing_encoded
        offset = len(fitted.labels)
        missing_code = self.category.replace_strict(
            encoded, list(range(offset, offset + len(encoded))), default=None, return_dtype=pl.UInt32
        )
        return pl.when(self.category.is_null()).then(value_code).otherwise(missing_code)


def _column_dtype(data: pl.DataFrame, column: str) -> pl.DataType:
    if column not in data.columns:
        raise KeyError(f"column {column!r} not found")
    return data.schema[column]


def _fit_variable(variable: Variable, schema: Schema, data: pl.DataFrame) -> _Fitted:
    missing = schema.missing_for(variable)
    columns = variable.columns()
    fitted = _Fitted(
        name=variable.name,
        kind="categorical",
        module=variable.module,
        pool=variable.filler_pool,
        columns=columns,
        missing_codes=[[_plain(code), category] for code, category in missing.codes.items()],
        missing_encoded=missing.encoded_categories(),
    )
    dtype = _column_dtype(data, columns[0])
    cells = _Cells(columns[0], dtype, missing)

    def observed() -> pl.Series:
        return data.select(cells.values.alias("value")).to_series().drop_nulls()

    def symbols(categories: Sequence[Any]) -> None:
        fitted.categories = [_plain(value) for value in categories]
        fitted.labels = [_label(value) for value in fitted.categories]
        fitted.fillers = [{"symbol": f"{fitted.pool}={label}"} for label in fitted.labels]
        fitted.values = [float(value) if _is_number(value) else None for value in fitted.categories]

    def levels(levels: Levels, series: pl.Series) -> list[float]:
        low, high = levels.low, levels.high
        if low is None or high is None:
            values = series.cast(pl.Float64).drop_nans()
            if values.is_empty():
                raise ValueError(f"{variable.name!r}: no observed values to derive the level range from")
            low = float(values.quantile(levels.quantiles[0], "linear")) if low is None else low
            high = float(values.quantile(levels.quantiles[1], "linear")) if high is None else high
        if high < low:
            raise ValueError(f"{variable.name!r}: the level range [{low}, {high}] is empty")
        return [float(low), float(high), float(levels.count)]

    def sorted_unique(series: pl.Series) -> list[Any]:
        if isinstance(series.dtype, (pl.Categorical, pl.Enum)):
            series = series.cast(pl.String)
        return series.unique().sort().to_list()

    if isinstance(variable, Nominal):
        if variable.categories is not None:
            categories = list(variable.categories)
        elif isinstance(dtype, pl.Enum):
            categories = list(dtype.categories)
        else:
            categories = sorted_unique(observed())
        symbols(categories)
    elif isinstance(variable, Binary):
        if variable.labels is not None:
            categories = list(variable.labels)
        elif dtype == pl.Boolean:
            categories = [False, True]
        else:
            categories = sorted_unique(observed())
            if len(categories) != 2:
                raise ValueError(
                    f"{variable.name!r}: a binary variable needs exactly two values, found {categories[:5]}"
                )
        symbols(categories)
    elif isinstance(variable, Ordinal):
        if variable.levels is not None:
            categories = list(variable.levels)
        elif isinstance(dtype, pl.Enum):
            categories = list(dtype.categories)
        elif dtype.is_numeric():
            categories = sorted_unique(observed())
        else:
            raise ValueError(f"{variable.name!r}: an ordinal variable needs `levels`, an Enum or a numeric column")
        symbols(categories)
        fitted.fillers = [
            {"level": {"pool": fitted.pool, "index": index, "count": len(categories)}}
            for index in range(len(categories))
        ]
    elif isinstance(variable, Numeric):
        if not dtype.is_numeric():
            raise TypeError(f"{variable.name!r}: a numeric variable needs a numeric column, found {dtype}")
        encoding = variable.encoding
        values = observed().cast(pl.Float64).drop_nans()
        if isinstance(encoding, Levels):
            fitted.kind = "levels"
            fitted.levels = levels(encoding, values)
            low, high, count = fitted.levels
            count = int(count)
            fitted.labels = [f"#{index}" for index in range(count)]
            fitted.fillers = [
                {"level": {"pool": fitted.pool, "index": index, "count": count}} for index in range(count)
            ]
            step = (high - low) / (count - 1) if count > 1 else 0.0
            fitted.values = [low + index * step for index in range(count)]
        else:
            fitted.kind = "continuous"
            if values.is_empty():
                raise ValueError(f"{variable.name!r}: no observed values to derive the scaling from")
            scale = 1.0
            if encoding.unit == "sd":
                std = values.std()
                scale = float(std) if std is not None and std > 0 else 1.0
            fitted.center = float(values.mean())  # type: ignore[arg-type]
            fitted.bandwidth = encoding.bandwidth * scale
            fitted.kernel = encoding.kernel
            fitted.observed_range = [
                float(values.quantile(0.005, "linear")),  # type: ignore[arg-type]
                float(values.quantile(0.995, "linear")),  # type: ignore[arg-type]
            ]
    elif isinstance(variable, TimedEvent):
        fitted.kind = "event"
        fitted.positive = [_plain(value) for value in variable.positive_values()]
        delta_dtype = _column_dtype(data, variable.delta)
        delta_cells = _Cells(variable.delta, delta_dtype, missing)
        deltas = data.select(delta_cells.values.alias("value")).to_series().drop_nulls()
        fitted.levels = levels(variable.delta_levels, deltas)
        low, high, count = fitted.levels
        count = int(count)
        step = (high - low) / (count - 1) if count > 1 else 0.0
        yes = {"symbol": f"{fitted.pool}=yes"}
        fitted.labels = ["no", "yes"] + [f"yes#{index}" for index in range(count)]
        fitted.fillers = [{"symbol": f"{fitted.pool}=no"}, yes] + [
            {"bind": [yes, {"level": {"pool": f"{fitted.pool}_dt", "index": index, "count": count}}]}
            for index in range(count)
        ]
        fitted.values = [None, None] + [low + index * step for index in range(count)]
    elif isinstance(variable, HierarchicalCode):
        fitted.kind = "code"
        codes = list(variable.codes) if variable.codes is not None else sorted_unique(observed().cast(pl.String))
        fitted.categories = [str(code) for code in codes]
        fitted.labels = list(fitted.categories)
        fitted.fillers = [
            {
                "bundle": [
                    {"symbol": f"{fitted.pool}={ancestor}"} for ancestor in [*variable.hierarchy(code), code]
                ]
            }
            for code in fitted.categories
        ]
        fitted.values = [None] * len(codes)
    else:
        raise TypeError(f"unsupported variable type {type(variable).__name__}")

    return fitted


def _code_exprs(fitted: _Fitted, data: pl.DataFrame) -> list[pl.Expr]:
    """The expressions producing the codes (and the scaled values) of a variable."""
    missing = fitted.missing
    dtype = _column_dtype(data, fitted.columns[0])
    cells = _Cells(fitted.columns[0], dtype, missing)

    if fitted.kind in ("categorical", "code"):
        value_code = _match(cells.values, dtype, fitted.categories)
        return [cells.code(fitted, value_code).alias(fitted.name)]
    if fitted.kind == "levels":
        value_code = fitted.level_expr(cells.values)
        return [cells.code(fitted, value_code).alias(fitted.name)]
    if fitted.kind == "continuous":
        value_code = pl.lit(None, pl.UInt32)
        scaled = ((cells.values.cast(pl.Float64) - fitted.center) / fitted.bandwidth).alias(f"{fitted.name}#value")
        return [cells.code(fitted, value_code).alias(fitted.name), scaled]
    if fitted.kind == "event":
        delta_dtype = _column_dtype(data, fitted.columns[1])
        delta = _Cells(fitted.columns[1], delta_dtype, missing).values
        positive = [value for value in fitted.positive if _compatible(value, dtype)]
        is_positive = (
            _match(cells.values, dtype, positive).is_not_null() if positive else pl.lit(False)
        )
        value_code = (
            pl.when(cells.values.is_null())
            .then(pl.lit(None, pl.UInt32))
            .when(~is_positive)
            .then(pl.lit(0, pl.UInt32))
            .when(delta.is_null())
            .then(pl.lit(1, pl.UInt32))
            .otherwise(fitted.level_expr(delta) + 2)
        )
        return [cells.code(fitted, value_code).cast(pl.UInt32).alias(fitted.name)]
    raise AssertionError(fitted.kind)


def _compatible(value: Any, dtype: pl.DataType) -> bool:
    if dtype == pl.Boolean:
        return isinstance(value, bool)
    if dtype.is_numeric():
        return _is_number(value)
    return isinstance(value, str)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QueryResult:
    """The result of :meth:`Encoder.query`: the matching records and the similarity to each term."""

    mask: np.ndarray
    similarities: np.ndarray
    threshold: float

    @property
    def indices(self) -> np.ndarray:
        return np.flatnonzero(self.mask)


class Encoder:
    """Encodes the rows of a table into hypervectors.

    >>> encoder = Encoder(MultiplyAddPermute(42), 10_000, schema).fit(train)
    >>> X_train, X_test = encoder.transform(train), encoder.transform(test)

    ``structure="hierarchical"`` bundles the variables of each module first (normalized if ``normalize_modules``)
    and binds the bundle to the module vector ``@<module>``; variables without a module are added directly.
    ``structure="flat"`` bundles all role-filler bindings at once and ignores modules.

    :meth:`fit` returns a new, fitted encoder and leaves this one untouched, so that one unfitted encoder can be
    fitted on each training fold.
    """

    def __init__(
        self,
        vsa: _native.VSA,
        size: int,
        schema: Schema,
        *,
        structure: Structure = "hierarchical",
        normalize_modules: bool = True,
    ) -> None:
        if structure not in ("hierarchical", "flat"):
            raise ValueError("structure must be 'hierarchical' or 'flat'")
        self.vsa = vsa
        self.size = size
        self.schema = schema
        self.structure: Structure = structure
        self.normalize_modules = normalize_modules
        self._fitted: list[_Fitted] | None = None
        self._native: Any = None
        self._index: dict[str, _Fitted] = {}

    # ── Fitting ───────────────────────────────────────────────────────────

    def fit(self, data: pl.DataFrame) -> Encoder:
        """Resolve all data-dependent decisions on ``data`` (e.g. the training fold) and build the codebook."""
        fitted = [_fit_variable(variable, self.schema, data) for variable in self.schema]
        encoder = Encoder(
            self.vsa, self.size, self.schema, structure=self.structure, normalize_modules=self.normalize_modules
        )
        encoder._set_fitted(fitted)
        return encoder

    def fit_transform(self, data: pl.DataFrame, **kwargs: Any) -> tuple[Encoder, np.ndarray]:
        """Fit on ``data`` and encode it. Returns the fitted encoder and the encoded matrix."""
        encoder = self.fit(data)
        return encoder, encoder.transform(data, **kwargs)

    def _set_fitted(self, fitted: list[_Fitted]) -> None:
        architecture = self.vsa.architecture()
        for variable in fitted:
            if variable.kind == "continuous" and architecture not in _REAL_VALUED:
                raise ValueError(
                    f"{variable.name!r}: FractionalPower requires a real-valued architecture "
                    "(HolographicReducedRepresentation, VectorDerivedTransformationBinding); use Levels instead"
                )
        self._fitted = fitted
        self._index = {variable.name: variable for variable in fitted}
        self._native = self.vsa.create_encoder(json.dumps(self.spec()))

    @property
    def is_fitted(self) -> bool:
        return self._fitted is not None

    def _require_fitted(self) -> list[_Fitted]:
        if self._fitted is None:
            raise RuntimeError("the encoder is not fitted yet; call `fit` first")
        return self._fitted

    def _variable(self, name: str) -> _Fitted:
        self._require_fitted()
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"unknown variable {name!r}") from None

    @property
    def native(self) -> Any:
        """The native encoder (e.g. ``MapEncoder``)."""
        self._require_fitted()
        return self._native

    @property
    def variables(self) -> list[str]:
        return [variable.name for variable in self._require_fitted()]

    def spec(self) -> dict[str, Any]:
        """The specification of the native encoder, which fully determines the codebook together with the seed."""
        return {
            "size": self.size,
            "structure": self.structure,
            "normalize_modules": self.normalize_modules,
            "variables": [variable.spec() for variable in self._require_fitted()],
        }

    def digest(self) -> str:
        """A stable fingerprint of the codebook. Equal digests indicate byte-identical codebooks."""
        return self.native.digest()

    # ── Encoding ──────────────────────────────────────────────────────────

    def codes(self, data: pl.DataFrame) -> pl.DataFrame:
        """The codes of all variables (and the scaled values of continuous ones), as passed to the native encoder."""
        fitted = self._require_fitted()
        expressions = [expression for variable in fitted for expression in _code_exprs(variable, data)]
        return data.select(expressions)

    def transform(
        self,
        data: pl.DataFrame,
        *,
        normalize: bool = True,
        dtype: Literal["float32", "int8"] | None = None,
    ) -> np.ndarray:
        """Encode the rows of ``data`` into a matrix with one row per record.

        Rows are normalized like vectors of the architecture if ``normalize`` (sign for BSC and MAP, L2 for HRR and
        VTB); otherwise they hold the raw sums. ``dtype`` defaults to ``"int8"`` for normalized BSC and MAP vectors
        (a quarter of the memory of ``"float32"``) and to ``"float32"`` otherwise. Records without any observed cell
        are rows of zeros.
        """
        if dtype is None:
            bipolar = self.vsa.architecture() not in _REAL_VALUED
            dtype = "int8" if bipolar and normalize else "float32"
        return self.native.encode(self.codes(data), normalize, dtype)

    def transform_batches(self, data: pl.DataFrame, batch_size: int, **kwargs: Any) -> Iterator[np.ndarray]:
        """Encode ``data`` in batches of ``batch_size`` rows, e.g. to write them to disk one after another."""
        for offset in range(0, data.height, batch_size):
            yield self.transform(data.slice(offset, batch_size), **kwargs)

    # ── Keys and expressions ──────────────────────────────────────────────

    def labels(self, variable: str) -> list[str]:
        """The labels of the keys of a variable: values first, then ``MISSING=<category>``."""
        return self._variable(variable).all_labels()

    def keys(self, variable: str) -> np.ndarray:
        """The dense full keys ``M ⊛ R ⊛ Φ(k)`` of a variable, one row per label."""
        self._variable(variable)
        return self.native.keys(variable)

    def continuous_keys(self, variable: str, values: Sequence[float]) -> np.ndarray:
        """The dense full keys ``M ⊛ R ⊛ B^x`` of a continuous variable for values in the original unit."""
        fitted = self._variable(variable)
        if fitted.kind != "continuous":
            raise ValueError(f"{variable!r} is not encoded by a fractional power")
        return self.native.continuous_keys(variable, [fitted.scale(float(value)) for value in values])

    def vector(self, expression: str) -> np.ndarray:
        """Evaluate an expression into a dense vector.

        Names are resolved in this order: names of the codebook (roles ``<variable>``, modules ``@<module>``,
        fillers ``<pool>=<value>``, levels ``<pool>#<index>``, ``MISSING=<category>``), full keys ``<variable>:<label>``
        and ranges, which are rewritten into bundles of full keys: ``<variable>>=<value>`` (also ``>``, ``<=``, ``<``)
        for levels, ordinal variables and the time since an event, and ``<variable>~<value>`` for continuous variables.
        Backticks quote names with whitespace or operators, e.g. ``smoking:`former smoker```.
        """
        return self.native.evaluate(self.rewrite(expression))

    def rewrite(self, expression: str) -> str:
        """Rewrite the ranges of an expression into bundles of full keys (see :meth:`vector`)."""
        self._require_fitted()
        segments = expression.split("`")
        for index in range(0, len(segments), 2):
            segments[index] = _RANGE.sub(self._rewrite_range, segments[index])
        return "`".join(segments)

    def _rewrite_range(self, match: re.Match[str]) -> str:
        name, operator, value = match.group("variable"), match.group("operator"), match.group("value")
        if name not in self._index:
            return match.group(0)
        fitted = self._index[name]

        if operator == "~":
            if fitted.kind != "continuous":
                raise ValueError(f"{name!r}: `~` requires a continuous variable")
            return f"{name}:={fitted.scale(float(value))!r}"
        _, selected = self._range_labels(fitted, operator, value)
        return self._join(name, selected)

    def _range_labels(self, fitted: _Fitted, operator: str, value: str) -> tuple[list[str], list[str]]:
        """All labels a range is chosen from, and the labels within the range."""
        name = fitted.name
        if fitted.kind == "continuous":
            raise ValueError(f"{name!r}: ranges are not supported for continuous variables; use `{name}~<value>`")

        if fitted.kind == "levels":
            labels, points = fitted.labels, fitted.values
        elif fitted.kind == "event":
            labels, points = fitted.labels[2:], fitted.values[2:]
        elif fitted.kind == "categorical" and fitted.fillers and "level" in fitted.fillers[0]:
            labels = fitted.labels
            if value in labels:
                # Compare by the order of the levels.
                position = labels.index(value)
                return labels, [label for index, label in enumerate(labels) if _compare(index, operator, position)]
            if not all(number is not None for number in fitted.values):
                raise ValueError(f"{name!r}: unknown level {value!r}")
            points = fitted.values
        else:
            raise ValueError(f"{name!r}: ranges require levels, an ordinal variable or an event")

        # Compare the points of the levels; values are encoded by the level of the nearest point.
        selected = [label for label, point in zip(labels, points) if _compare(point, operator, float(value))]  # type: ignore[arg-type]
        return labels, selected

    @staticmethod
    def _join(name: str, labels: list[str]) -> str:
        if not labels:
            raise ValueError(f"{name!r}: the range does not contain any level")
        terms = [f"`{name}:{label}`" for label in labels]
        return terms[0] if len(terms) == 1 else "(" + " + ".join(terms) + ")"

    # ── Reading out ───────────────────────────────────────────────────────

    def similarities(self, X: np.ndarray, vectors: np.ndarray, *, block_size: int = 8192) -> np.ndarray:
        """Cosine similarities between the records ``X`` (n × d) and the dense ``vectors`` (k × d or d)."""
        return _cosine(X, vectors, block_size)

    def null_threshold(
        self,
        X: np.ndarray,
        *,
        quantile: float = 0.999,
        samples: int = 1000,
        max_rows: int = 10_000,
        seed: str = "null",
    ) -> float:
        """The ``quantile`` of the similarities between records and random vectors (e.g. of the training fold).

        Similarities above this threshold are unlikely to arise by chance. ``max_rows`` evenly spaced records are used.
        """
        rows = X if X.shape[0] <= max_rows else X[np.linspace(0, X.shape[0] - 1, max_rows).astype(np.int64)]
        random = self.native.random(seed, samples)
        return float(np.quantile(_cosine(rows, random, 8192), quantile))

    def query(
        self,
        X: np.ndarray,
        terms: Sequence[str] | str,
        threshold: float,
        *,
        ranges: Literal["threshold", "contrast", "decode"] = "threshold",
    ) -> QueryResult:
        """Select the records matching *every* term (a conjunction, evaluated term by term).

        Each term is an expression as accepted by :meth:`vector`, e.g. ``["smoking:current", "bmi>=30"]``. A record
        matches a term if its cosine similarity to the term exceeds ``threshold`` (see :meth:`null_threshold`).

        Terms consisting of a single range (e.g. ``"bmi>=30"``) can be evaluated differently, since the similarity
        of neighbouring levels makes the threshold permissive: ``ranges="contrast"`` requires the record to be more
        similar to the levels within than to those outside of the range, and ``ranges="decode"`` requires the most
        similar level (cleanup) to lie within the range. The reported similarity is always the one to the term.
        """
        if ranges not in ("threshold", "contrast", "decode"):
            raise ValueError("ranges must be 'threshold', 'contrast' or 'decode'")
        terms = [terms] if isinstance(terms, str) else list(terms)
        vectors = np.stack([self.vector(term) for term in terms])
        similarities = _cosine(X, vectors, 8192)
        matches = similarities > threshold

        if ranges != "threshold":
            for column, term in enumerate(terms):
                match = _RANGE.fullmatch(term.strip())
                if match is None or match.group("operator") == "~" or match.group("variable") not in self._index:
                    continue
                fitted = self._index[match.group("variable")]
                labels, selected = self._range_labels(fitted, match.group("operator"), match.group("value"))
                outside = [label for label in labels if label not in selected]
                if ranges == "contrast":
                    if not outside:
                        matches[:, column] = True
                        continue
                    contrast = _cosine(X, self.vector(self._join(fitted.name, outside)), 8192)[:, 0]
                    matches[:, column] = similarities[:, column] > contrast
                else:
                    all_labels = fitted.all_labels()
                    indices = [all_labels.index(label) for label in labels]
                    best = np.argmax(_cosine(X, self.keys(fitted.name)[indices], 8192), axis=1)
                    chosen = np.isin(np.asarray(labels, dtype=object)[best], np.asarray(selected, dtype=object))
                    matches[:, column] = chosen & (np.abs(X).sum(axis=1) > 0)

        return QueryResult(np.all(matches, axis=1), similarities, threshold)

    def decode(
        self,
        X: np.ndarray,
        variables: Iterable[str] | None = None,
        *,
        top_k: int = 1,
        grid_size: int = 101,
    ) -> pl.DataFrame:
        """Decode the value of each variable for each record by the most similar full keys.

        Returns one row per record, variable and rank with the ``label`` of the key, its ``value`` (for levels, the
        point of the level; for continuous variables, the value on a grid of ``grid_size`` points over the central
        99 % of the training data) and the cosine ``similarity``.
        """
        names = self.variables if variables is None else list(variables)
        frames = []
        for name in names:
            fitted = self._variable(name)
            labels = fitted.all_labels()
            values = fitted.values + [None] * len(fitted.missing_encoded)
            keys = self.keys(name)
            if fitted.kind == "continuous":
                assert fitted.observed_range is not None
                grid = np.linspace(*fitted.observed_range, grid_size)
                keys = np.vstack([self.continuous_keys(name, grid.tolist()), keys]) if len(keys) else (
                    self.continuous_keys(name, grid.tolist())
                )
                labels = [f"~{value:.6g}" for value in grid] + labels
                values = [float(value) for value in grid] + values
            similarities = _cosine(X, keys, 8192)
            k = min(top_k, keys.shape[0])
            best = np.argsort(-similarities, axis=1)[:, :k]
            rows = np.repeat(np.arange(X.shape[0]), k)
            flat = best.reshape(-1)
            frames.append(
                pl.DataFrame(
                    {
                        "row": rows,
                        "variable": [name] * len(rows),
                        "rank": np.tile(np.arange(1, k + 1), X.shape[0]),
                        "label": [labels[index] for index in flat],
                        "value": [values[index] for index in flat],
                        "similarity": similarities[rows, flat],
                    },
                    schema={
                        "row": pl.Int64,
                        "variable": pl.String,
                        "rank": pl.Int64,
                        "label": pl.String,
                        "value": pl.Float64,
                        "similarity": pl.Float64,
                    },
                )
            )
        return pl.concat(frames) if frames else pl.DataFrame()

    def explain(self, weights: np.ndarray, variables: Iterable[str] | None = None, *, grid_size: int = 101) -> pl.DataFrame:
        """The contribution ``⟨w, M ⊛ R ⊛ Φ(k)⟩`` of every label of every variable to a linear readout ``w · P``.

        ``weights`` must refer to the encoded vectors themselves; for a readout on standardized features, pass the
        weights divided by the standard deviations. With normalization, the decomposition is approximate.
        """
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        names = self.variables if variables is None else list(variables)
        frames = []
        for name in names:
            fitted = self._variable(name)
            labels = fitted.all_labels()
            values = fitted.values + [None] * len(fitted.missing_encoded)
            keys = self.keys(name)
            if fitted.kind == "continuous":
                assert fitted.observed_range is not None
                grid = np.linspace(*fitted.observed_range, grid_size)
                grid_keys = self.continuous_keys(name, grid.tolist())
                keys = np.vstack([grid_keys, keys]) if len(keys) else grid_keys
                labels = [f"~{value:.6g}" for value in grid] + labels
                values = [float(value) for value in grid] + values
            frames.append(
                pl.DataFrame(
                    {
                        "variable": [name] * len(labels),
                        "label": labels,
                        "value": values,
                        "contribution": keys @ weights if len(keys) else np.zeros(0, np.float32),
                    },
                    schema={"variable": pl.String, "label": pl.String, "value": pl.Float64, "contribution": pl.Float64},
                )
            )
        return pl.concat(frames) if frames else pl.DataFrame()

    def explain_rows(self, weights: np.ndarray, data: pl.DataFrame, X: np.ndarray | None = None) -> pl.DataFrame:
        """The contribution of each observed cell of each record to a linear readout ``w · P``.

        Each record gets an additional row with the variable ``<residual>``: the difference between ``w · P`` and the
        sum of the contributions, caused by crosstalk and normalization. ``X`` defaults to ``transform(data)``.
        """
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if X is None:
            X = self.transform(data)
        codes = self.codes(data)
        total = np.zeros(data.height, dtype=np.float64)
        frames = []
        for fitted in self._require_fitted():
            column = codes.get_column(fitted.name).to_numpy(allow_copy=True)
            labels = np.array(fitted.all_labels() or [""], dtype=object)
            contributions = (self.keys(fitted.name) @ weights) if fitted.all_labels() else np.zeros(1, np.float32)
            observed = ~np.isnan(column.astype(np.float64))
            rows = np.flatnonzero(observed)
            indices = column[observed].astype(np.int64)
            values = contributions[indices].astype(np.float64)
            row_labels = labels[indices].tolist()

            if fitted.kind == "continuous":
                scaled = codes.get_column(f"{fitted.name}#value").to_numpy()
                continuous = ~observed & ~np.isnan(scaled)
                continuous_rows = np.flatnonzero(continuous)
                if len(continuous_rows):
                    keys = self.native.continuous_keys(fitted.name, scaled[continuous_rows].tolist())
                    rows = np.concatenate([rows, continuous_rows])
                    values = np.concatenate([values, (keys @ weights).astype(np.float64)])
                    row_labels += [f"~{fitted.unscale(value):.6g}" for value in scaled[continuous_rows]]

            np.add.at(total, rows, values)
            frames.append(
                pl.DataFrame(
                    {"row": rows, "variable": [fitted.name] * len(rows), "label": row_labels, "contribution": values},
                    schema={"row": pl.Int64, "variable": pl.String, "label": pl.String, "contribution": pl.Float64},
                )
            )

        readout = np.asarray(X, dtype=np.float32) @ weights
        frames.append(
            pl.DataFrame(
                {
                    "row": np.arange(data.height),
                    "variable": ["<residual>"] * data.height,
                    "label": [""] * data.height,
                    "contribution": readout.astype(np.float64) - total,
                },
                schema={"row": pl.Int64, "variable": pl.String, "label": pl.String, "contribution": pl.Float64},
            )
        )
        return pl.concat(frames).sort("row", maintain_order=True)

    # ── Persistence ───────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize the fitted encoder: architecture, seed, size, structure and all fitted variables."""
        return json.dumps(
            {
                "format": _FORMAT_VERSION,
                "architecture": self.vsa.architecture(),
                "seed": self.vsa.seed(),
                "size": self.size,
                "structure": self.structure,
                "normalize_modules": self.normalize_modules,
                "variables": [dataclasses.asdict(variable) for variable in self._require_fitted()],
                "digest": self.digest(),
            },
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> Encoder:
        """Restore a fitted encoder from :meth:`to_json`. The codebook is regenerated and checked against its digest.

        The schema of the restored encoder only describes the fitted variables; it cannot be fitted again.
        """
        state = json.loads(text)
        if state.get("format") != _FORMAT_VERSION:
            raise ValueError(f"unsupported format {state.get('format')!r}")
        vsa = _ARCHITECTURES[state["architecture"]](state["seed"])
        fitted = [_Fitted(**variable) for variable in state["variables"]]
        encoder = cls(
            vsa,
            state["size"],
            Schema([]),
            structure=state["structure"],
            normalize_modules=state["normalize_modules"],
        )
        encoder._set_fitted(fitted)
        if encoder.digest() != state["digest"]:
            raise ValueError("the regenerated codebook does not match the stored digest")
        return encoder

    def __repr__(self) -> str:
        state = f"{len(self._fitted)} variables" if self._fitted is not None else "not fitted"
        return f"Encoder({self.vsa.architecture()}, size={self.size}, structure={self.structure!r}, {state})"


_RANGE = re.compile(r"(?P<variable>[A-Za-z_][\w.]*)(?P<operator>>=|<=|>|<|~)(?P<value>[^\s()*+^`]+)")


def _compare(left: float, operator: str, right: float) -> bool:
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    raise ValueError(f"unknown operator {operator!r}")


def _cosine(X: np.ndarray, vectors: np.ndarray, block_size: int) -> np.ndarray:
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    vector_norms = np.linalg.norm(vectors, axis=1)
    vector_norms[vector_norms == 0] = 1.0
    result = np.empty((X.shape[0], vectors.shape[0]), dtype=np.float64)
    for start in range(0, X.shape[0], block_size):
        block = np.asarray(X[start : start + block_size], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1)
        norms[norms == 0] = 1.0
        result[start : start + block_size] = (block @ vectors.T) / norms[:, None] / vector_norms[None, :]
    return result

