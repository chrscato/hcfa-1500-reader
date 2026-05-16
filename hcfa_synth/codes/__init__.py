"""Reference codeset loaders. CSVs are bundled as package data."""

from __future__ import annotations

import csv
from functools import lru_cache
from importlib.resources import files
from typing import List, Tuple


def _load_csv(name: str) -> List[Tuple[str, str]]:
    path = files(__package__) / f"{name}.csv"
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(row["code"], row["description"]) for row in reader]


@lru_cache(maxsize=None)
def load_pos_codes() -> List[Tuple[str, str]]:
    """Place-of-service codes (code, description)."""
    return _load_csv("pos_codes")


@lru_cache(maxsize=None)
def load_icd10_codes() -> List[Tuple[str, str]]:
    """Sampled ICD-10-CM codes (code, description)."""
    return _load_csv("icd10_sample")


@lru_cache(maxsize=None)
def load_hcpcs_codes() -> List[Tuple[str, str]]:
    """Sampled HCPCS Level II codes (code, description)."""
    return _load_csv("hcpcs_sample")
