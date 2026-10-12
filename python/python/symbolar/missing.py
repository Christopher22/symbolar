"""Helpers for missing-value codes."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from .encoder import _missing_category
from .schema import MissingCodes

__all__ = ["split_missing"]


def split_missing(
    data: pl.DataFrame,
    missing: MissingCodes,
    columns: Iterable[str] | None = None,
    *,
    suffix: str = "__missing",
) -> pl.DataFrame:
    """Split each column into its values (missing-value codes replaced by null) and a column ``<column><suffix>``
    holding the missing-value category of each cell (or null).

    Only columns containing at least one code get the additional column. ``columns`` defaults to all columns.
    """
    selected = data.columns if columns is None else list(columns)
    categories = {}
    for column in selected:
        category = _missing_category(pl.col(column), data.schema[column], missing)
        if category is not None:
            categories[column] = category

    flagged = data.with_columns(category.alias(column + suffix) for column, category in categories.items())
    present = [column for column in categories if flagged.get_column(column + suffix).null_count() < flagged.height]
    return flagged.with_columns(
        pl.when(pl.col(column + suffix).is_null()).then(pl.col(column)).alias(column) for column in present
    ).drop([column + suffix for column in categories if column not in present])
