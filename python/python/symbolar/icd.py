"""Hierarchies of medical codes for :class:`~symbolar.HierarchicalCode`."""

from __future__ import annotations

import re

__all__ = ["icd10_chapter", "icd10_hierarchy"]

#: The chapters of the ICD-10 (WHO) as ``(chapter, first code, last code)``.
ICD10_CHAPTERS: tuple[tuple[str, str, str], ...] = (
    ("I", "A00", "B99"),
    ("II", "C00", "D48"),
    ("III", "D50", "D89"),
    ("IV", "E00", "E90"),
    ("V", "F00", "F99"),
    ("VI", "G00", "G99"),
    ("VII", "H00", "H59"),
    ("VIII", "H60", "H95"),
    ("IX", "I00", "I99"),
    ("X", "J00", "J99"),
    ("XI", "K00", "K93"),
    ("XII", "L00", "L99"),
    ("XIII", "M00", "M99"),
    ("XIV", "N00", "N99"),
    ("XV", "O00", "O99"),
    ("XVI", "P00", "P96"),
    ("XVII", "Q00", "Q99"),
    ("XVIII", "R00", "R99"),
    ("XIX", "S00", "T98"),
    ("XX", "V01", "Y98"),
    ("XXI", "Z00", "Z99"),
    ("XXII", "U00", "U85"),
)

_CATEGORY = re.compile(r"^([A-Z])(\d{2})")


def icd10_chapter(code: str) -> str | None:
    """The chapter (e.g. ``"IX"``) of an ICD-10 code, or ``None`` if the code is not recognized."""
    category = _CATEGORY.match(code.strip().upper())
    if category is None:
        return None
    three = category.group(1) + category.group(2)
    for chapter, first, last in ICD10_CHAPTERS:
        if first <= three <= last:
            return chapter
    return None


def icd10_hierarchy(code: str) -> list[str]:
    """The ancestors of an ICD-10 code, from the most general one: ``"I21.4"`` → ``["chapter:IX", "I21"]``.

    The code itself is added by :class:`~symbolar.HierarchicalCode`. Unrecognized codes have no ancestors.
    """
    normalized = code.strip().upper()
    category = _CATEGORY.match(normalized)
    if category is None:
        return []
    ancestors = []
    chapter = icd10_chapter(normalized)
    if chapter is not None:
        ancestors.append(f"chapter:{chapter}")
    three = category.group(1) + category.group(2)
    if normalized.replace(".", "") != three:
        ancestors.append(three)
    return ancestors
