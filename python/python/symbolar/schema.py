"""Declarative description of how the variables of a table are encoded.

A :class:`Schema` lists one :class:`Variable` per encoded variable. It can be written explicitly, without any data,
or inferred from the dtypes of a Polars DataFrame with :meth:`Schema.infer`, where every inferred decision can be
overridden per column.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import polars as pl

__all__ = [
    "Binary",
    "FractionalPower",
    "HierarchicalCode",
    "Levels",
    "MissingCodes",
    "Nominal",
    "Numeric",
    "Ordinal",
    "Schema",
    "TimedEvent",
    "Variable",
]

MissingPolicy = Literal["omit", "encode"]


# ─────────────────────────────────────────────────────────────────────────────
# Encodings of numbers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Levels:
    """Level vectors: ``count`` equally spaced points on ``[low, high]``.

    A value is encoded by the level of the nearest point; values outside the range are clipped. Neighbouring levels
    are similar, and the similarity decreases linearly until the first and last level are unrelated.

    If ``low`` or ``high`` is not given, it is set to the corresponding quantile of the training data during
    :meth:`Encoder.fit <symbolar.Encoder.fit>`.
    """

    count: int = 64
    low: float | None = None
    high: float | None = None
    quantiles: tuple[float, float] = (0.005, 0.995)

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("Levels need at least one level")
        lower, upper = self.quantiles
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower <= upper <= 1")


@dataclass(frozen=True)
class FractionalPower:
    """Fractional power encoding ``B^(x / β)`` for real-valued architectures (HRR, VTB).

    The similarity of two encoded values follows ``kernel((x - y) / β)``. The bandwidth ``β`` is ``bandwidth``
    times the standard deviation of the training data (``unit="sd"``) or ``bandwidth`` itself (``unit="absolute"``).
    """

    bandwidth: float = 1.0
    unit: Literal["sd", "absolute"] = "sd"
    kernel: Literal["sinc", "gaussian"] = "sinc"

    def __post_init__(self) -> None:
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")


NumericEncoding = Union[Levels, FractionalPower]


# ─────────────────────────────────────────────────────────────────────────────
# Missing values
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MissingCodes:
    """Codes marking missing values, grouped into named categories.

    ``codes`` maps a value (e.g. ``7775`` or ``"weiß nicht"``) to a category (e.g. ``"structural"``). For each
    category, ``policy`` decides whether the cell is left out of the record (``"omit"``, the default) or bound to the
    shared filler ``MISSING=<category>`` (``"encode"``). Nulls are always left out.
    """

    codes: Mapping[Any, str] = field(default_factory=dict)
    policy: Mapping[str, MissingPolicy] = field(default_factory=dict)
    default: MissingPolicy = "omit"

    def __post_init__(self) -> None:
        for category, policy in {**self.policy, "<default>": self.default}.items():
            if policy not in ("omit", "encode"):
                raise ValueError(f"invalid policy {policy!r} for {category!r}: use 'omit' or 'encode'")

    def categories(self) -> list[str]:
        """All categories in the order of their first appearance."""
        return list(dict.fromkeys(self.codes.values()))

    def encoded_categories(self) -> list[str]:
        """The categories bound to a filler, in the order of their first appearance."""
        return [category for category in self.categories() if self.policy.get(category, self.default) == "encode"]

    def with_policy(self, policy: Mapping[str, MissingPolicy] | MissingPolicy) -> MissingCodes:
        """The same codes with another policy, e.g. ``codes.with_policy("encode")``."""
        if isinstance(policy, str):
            return dataclasses.replace(self, policy={}, default=policy)
        return dataclasses.replace(self, policy=dict(policy))


# ─────────────────────────────────────────────────────────────────────────────
# Variables
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Variable:
    """Base class of all variables.

    ``name`` names the role vector and is used to refer to the variable in expressions (``name:label``). The source
    column defaults to ``name``. Variables sharing a ``pool`` share their filler vectors (e.g. all ``ja_nein``
    questions); the binding with the role keeps them distinguishable. ``missing`` overrides the codes of the schema.
    """

    name: str
    module: str | None = field(default=None, kw_only=True)
    pool: str | None = field(default=None, kw_only=True)
    column: str | None = field(default=None, kw_only=True)
    missing: MissingCodes | None = field(default=None, kw_only=True)

    @property
    def kind(self) -> str:
        return type(self).__name__.lower()

    @property
    def filler_pool(self) -> str:
        return self.pool if self.pool is not None else self.name

    def columns(self) -> list[str]:
        """The source columns of the variable."""
        return [self.column if self.column is not None else self.name]


@dataclass(frozen=True)
class Nominal(Variable):
    """Unordered categories, each bound to its own random filler ``<pool>=<category>``.

    Without ``categories``, they are taken from an ``Enum`` dtype or from the training data.
    """

    categories: Sequence[Any] | None = None


@dataclass(frozen=True)
class Ordinal(Variable):
    """Ordered categories, encoded by level vectors ``<pool>#<index>`` such that neighbouring levels are similar.

    Without ``levels``, they are taken from an ``Enum`` dtype (in its order) or, for numeric columns, from the sorted
    distinct values of the training data.
    """

    levels: Sequence[Any] | None = None


@dataclass(frozen=True)
class Binary(Variable):
    """Two categories ``(negative, positive)``, both encoded explicitly (the negative one is not left out).

    Without ``labels``, a ``Boolean`` column uses ``(False, True)`` and other columns the two sorted distinct values of
    the training data.
    """

    labels: tuple[Any, Any] | None = None


@dataclass(frozen=True)
class Numeric(Variable):
    """A number, encoded by :class:`Levels` or by :class:`FractionalPower` (only for HRR and VTB)."""

    encoding: NumericEncoding = field(default_factory=Levels)


@dataclass(frozen=True)
class TimedEvent(Variable):
    """An event with the time since it happened: ``R ⊛ <pool>=yes ⊛ T(Δt)``, or ``R ⊛ <pool>=no``.

    ``indicator`` is the column stating whether the event happened (it happened if the value is in ``positive``).
    ``delta`` is the column with the time since the event (e.g. age at baseline minus age at diagnosis), encoded by
    the level vectors ``<pool>_dt#<index>``. If the event happened but the time is unknown, ``<pool>=yes`` is bound alone.
    """

    indicator: str = ""
    delta: str = ""
    positive: Any = (True, 1, "1", "ja", "yes")
    delta_levels: Levels = field(default_factory=lambda: Levels(81, 0.0, 80.0))

    def __post_init__(self) -> None:
        if not self.indicator or not self.delta:
            raise ValueError(f"TimedEvent {self.name!r} requires an indicator and a delta column")

    def columns(self) -> list[str]:
        return [self.indicator, self.delta]

    def positive_values(self) -> list[Any]:
        if isinstance(self.positive, (list, tuple, set, frozenset)):
            return list(self.positive)
        return [self.positive]


@dataclass(frozen=True)
class HierarchicalCode(Variable):
    """A code within a hierarchy (e.g. ICD-10), encoded as ``norm(<pool>=<ancestor_1> + … + <pool>=<code>)``.

    Codes sharing ancestors are similar. ``hierarchy`` maps a code to its ancestors (from the most general one), e.g.
    :func:`symbolar.icd10_hierarchy`. Without ``codes``, they are taken from the training data; unknown codes are
    left out during encoding.
    """

    hierarchy: Callable[[str], Sequence[str]] = field(default=lambda code: [])
    codes: Sequence[str] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

Override = Union[Variable, Levels, FractionalPower, Mapping[str, Any], None]
ModuleAssignment = Union[Mapping[str, str], Callable[[str], Union[str, None]], None]

_INTEGER_TYPES = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)


class Schema:
    """The variables to encode, together with the default missing-value codes.

    >>> schema = Schema([Nominal("smoking", module="lifestyle"), Numeric("bmi", Levels(32))])
    """

    def __init__(
        self,
        variables: Iterable[Variable],
        missing: MissingCodes | None = None,
        *,
        sources: Mapping[str, str] | None = None,
    ) -> None:
        self._variables: list[Variable] = list(variables)
        self.missing = missing
        self._sources = dict(sources or {})

        names = [variable.name for variable in self._variables]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"variables defined twice: {duplicates}")

    # ── Inference ──────────────────────────────────────────────────────────

    @classmethod
    def infer(
        cls,
        data: pl.DataFrame | pl.LazyFrame | pl.Schema | Mapping[str, pl.DataType],
        *,
        overrides: Mapping[str, Override] | None = None,
        modules: ModuleAssignment = None,
        missing: MissingCodes | None = None,
        ignore: Iterable[str] = (),
        integer_as_nominal: int | None = None,
    ) -> Schema:
        """Infer the variables from the dtypes of a DataFrame (or a schema).

        =========================  ==========================================================================
        dtype                      inferred variable
        =========================  ==========================================================================
        ``Enum``                   :class:`Nominal` with the categories of the dtype
        ``Categorical``, ``String``  :class:`Nominal` with the categories of the training data
        ``Boolean``                :class:`Binary`
        integers, floats           :class:`Numeric` with :class:`Levels` (see ``integer_as_nominal``)
        anything else              error; use ``ignore`` or ``overrides``
        =========================  ==========================================================================

        ``overrides`` maps a column to a complete :class:`Variable` (e.g. ``Ordinal("x")``), to a numeric encoding
        (e.g. ``Levels(16)``), to a mapping of fields replacing those of the inferred variable (e.g.
        ``{"pool": "ja_nein"}``), or to ``None`` to drop the column. Overrides for names that are not columns add
        variables (e.g. a :class:`TimedEvent`); columns consumed by such variables are not inferred separately.

        ``modules`` assigns modules, as mapping or as function of the column name. ``integer_as_nominal=k`` infers
        integer columns with at most ``k`` distinct values as :class:`Nominal` (requires a DataFrame).
        """
        if isinstance(data, pl.LazyFrame):
            dtypes = data.collect_schema()
            frame = None
        elif isinstance(data, pl.DataFrame):
            dtypes = data.schema
            frame = data
        else:
            dtypes = pl.Schema(data)
            frame = None
        if integer_as_nominal is not None and frame is None:
            raise ValueError("integer_as_nominal requires a DataFrame to count distinct values")

        overrides = dict(overrides or {})
        ignored = set(ignore)
        unknown_ignored = sorted(ignored - set(dtypes))
        if unknown_ignored:
            raise ValueError(f"ignored columns not found: {unknown_ignored}")

        # Columns consumed by explicitly given variables are not inferred on their own.
        consumed: set[str] = set()
        for name, override in overrides.items():
            if isinstance(override, Variable):
                consumed.update(column for column in override.columns() if column != name)

        variables: list[Variable] = []
        sources: dict[str, str] = {}
        unsupported: list[str] = []
        for name, dtype in dtypes.items():
            if name in ignored or (name in consumed and name not in overrides):
                continue

            inferred = _infer_variable(name, dtype, frame, integer_as_nominal)
            if name in overrides:
                override = overrides.pop(name)
                if override is None:
                    continue
                variable = _apply_override(name, inferred, override)
                sources[variable.name] = "override"
            elif inferred is None:
                unsupported.append(f"{name} ({dtype})")
                continue
            else:
                variable = inferred
                sources[variable.name] = "inferred"
            variables.append(variable)

        if unsupported:
            raise TypeError(
                "cannot infer an encoding for the columns "
                + ", ".join(unsupported)
                + "; pass them to `ignore` or `overrides`"
            )

        for name, override in overrides.items():
            if not isinstance(override, Variable):
                raise ValueError(f"override for {name!r} is not a column; only complete variables can be added")
            variables.append(override)
            sources[override.name] = "override"

        variables = [_assign_module(variable, modules) for variable in variables]
        return cls(variables, missing, sources=sources)

    # ── Access ─────────────────────────────────────────────────────────────

    @property
    def variables(self) -> list[Variable]:
        return list(self._variables)

    def __iter__(self) -> Iterator[Variable]:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    def __getitem__(self, name: str) -> Variable:
        for variable in self._variables:
            if variable.name == name:
                return variable
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        return any(variable.name == name for variable in self._variables)

    def __repr__(self) -> str:
        return f"Schema({len(self)} variables)"

    def missing_for(self, variable: Variable) -> MissingCodes:
        """The missing-value codes applying to a variable."""
        if variable.missing is not None:
            return variable.missing
        return self.missing if self.missing is not None else MissingCodes()

    def replace(self, *variables: Variable, missing: MissingCodes | None = None) -> Schema:
        """A copy with variables replaced (by name) or added, and optionally other missing-value codes."""
        replaced = {variable.name: variable for variable in variables}
        updated = [replaced.pop(variable.name, variable) for variable in self._variables]
        sources = dict(self._sources)
        sources.update({variable.name: "override" for variable in variables})
        return Schema(
            [*updated, *replaced.values()],
            missing if missing is not None else self.missing,
            sources=sources,
        )

    def describe(self) -> pl.DataFrame:
        """One row per variable: name, type, module, pool, source columns, missing policy and origin of the decision."""
        rows = []
        for variable in self._variables:
            missing = self.missing_for(variable)
            encoding = ""
            if isinstance(variable, Numeric):
                encoding = type(variable.encoding).__name__
            rows.append(
                {
                    "variable": variable.name,
                    "type": variable.kind,
                    "encoding": encoding,
                    "module": variable.module,
                    "pool": variable.filler_pool,
                    "columns": ", ".join(variable.columns()),
                    "missing_encoded": ", ".join(missing.encoded_categories()),
                    "source": self._sources.get(variable.name, "explicit"),
                }
            )
        return pl.DataFrame(
            rows,
            schema={
                "variable": pl.String,
                "type": pl.String,
                "encoding": pl.String,
                "module": pl.String,
                "pool": pl.String,
                "columns": pl.String,
                "missing_encoded": pl.String,
                "source": pl.String,
            },
        )


def _infer_variable(
    name: str, dtype: pl.DataType, frame: pl.DataFrame | None, integer_as_nominal: int | None
) -> Variable | None:
    if isinstance(dtype, pl.Enum):
        return Nominal(name, categories=list(dtype.categories))
    if isinstance(dtype, (pl.Categorical, pl.String)) or dtype == pl.String:
        return Nominal(name)
    if dtype == pl.Boolean:
        return Binary(name)
    if dtype.is_integer() and integer_as_nominal is not None and frame is not None:
        if frame.get_column(name).drop_nulls().n_unique() <= integer_as_nominal:
            return Nominal(name)
    if dtype.is_numeric():
        return Numeric(name)
    return None


def _apply_override(name: str, inferred: Variable | None, override: Override) -> Variable:
    if isinstance(override, Variable):
        return override
    if isinstance(override, (Levels, FractionalPower)):
        base = inferred if isinstance(inferred, Numeric) else Numeric(name)
        return dataclasses.replace(base, encoding=override)
    if isinstance(override, Mapping):
        if inferred is None:
            raise TypeError(f"cannot infer an encoding for {name!r} to apply the override to; pass a variable")
        return dataclasses.replace(inferred, **override)
    raise TypeError(f"invalid override for {name!r}: {override!r}")


def _assign_module(variable: Variable, modules: ModuleAssignment) -> Variable:
    if variable.module is not None or modules is None:
        return variable
    module = modules(variable.name) if callable(modules) else modules.get(variable.name)
    return dataclasses.replace(variable, module=module) if module is not None else variable
