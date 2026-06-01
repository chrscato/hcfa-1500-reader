"""Field-level value normalization for scoring.

Two passes:
  - `exact`      — string equality, no normalization (the hard bar)
  - `normalized` — apply the rules below, then string-equal

Normalization rules (applied to BOTH pred and gt before comparison):
  * strip leading/trailing whitespace; collapse internal runs to a single space
  * upper-case
  * treat None / "" / "NONE" / "N/A" / "NA" as the same blank token
  * field-type-specific rules dispatched on key suffix:
      - dates  ("*_date", "*_birth", "physician_date", etc) → MM/DD/YYYY
      - phones ("phone")                                    → digits only
      - money  ("charges", "total_charge", "amount_paid")   → "%.2f"
      - npis, ids                                           → digits/alnum only
      - state codes                                         → 2-letter upper
      - sex                                                 → first letter upper

Type detection is by key name (substring match), not by value sniffing — the
GT schema is stable, so this is safer than guessing.
"""

from __future__ import annotations

import re
from typing import Tuple

_BLANK_TOKENS = {"", "NONE", "N/A", "NA", "NULL"}
_WS_RE = re.compile(r"\s+")
_DIGITS_RE = re.compile(r"\D+")
_DATE_PATTERNS = [
    re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"),
    re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2})$"),
    re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"),
]


def is_date_field(key: str) -> bool:
    k = key.lower()
    return (
        k.endswith("_date")
        or k.endswith("_from")
        or k.endswith("_to")
        or "_birth" in k
        or "current_illness_date" in k
        or "physician_signature_date" in k
    )


def is_phone_field(key: str) -> bool:
    return "phone" in key.lower()


def is_money_field(key: str) -> bool:
    k = key.lower()
    return k.endswith("charges") or k.endswith("total_charge") or k.endswith("amount_paid")


def is_state_field(key: str) -> bool:
    return key.lower().endswith(".state") or key.lower().endswith("_state") and "accident_state" not in key.lower()


def is_sex_field(key: str) -> bool:
    k = key.lower()
    return k.endswith("_sex") or k == "sex"


def is_id_field(key: str) -> bool:
    k = key.lower()
    return "npi" in k or k.endswith(".other_id") or "insured_id" in k or "policy_group" in k


def _generic_normalize(value: str) -> str:
    if value is None:
        return ""
    s = _WS_RE.sub(" ", str(value).strip()).upper()
    if s in _BLANK_TOKENS:
        return ""
    return s


def _normalize_date(value: str) -> str:
    s = _generic_normalize(value)
    if not s:
        return ""
    for pat in _DATE_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        g = m.groups()
        if pat.pattern.startswith("^(\\d{4})"):
            yyyy, mm, dd = g
        else:
            mm, dd, yy_or_yyyy = g
            yyyy = yy_or_yyyy if len(yy_or_yyyy) == 4 else (
                f"19{yy_or_yyyy}" if int(yy_or_yyyy) >= 50 else f"20{yy_or_yyyy}"
            )
        return f"{int(mm):02d}/{int(dd):02d}/{int(yyyy):04d}"
    return s


def _normalize_phone(value: str) -> str:
    s = _generic_normalize(value)
    if not s:
        return ""
    digits = _DIGITS_RE.sub("", s)
    return digits


def _normalize_money(value: str) -> str:
    s = _generic_normalize(value).replace("$", "").replace(",", "").strip()
    if not s:
        return ""
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return s


def _normalize_id(value: str) -> str:
    s = _generic_normalize(value)
    if not s:
        return ""
    return re.sub(r"[^A-Z0-9]", "", s)


def _normalize_state(value: str) -> str:
    s = _generic_normalize(value)
    return s[:2] if len(s) >= 2 else s


def _normalize_sex(value: str) -> str:
    s = _generic_normalize(value)
    if not s:
        return ""
    return s[0]


def normalize(key: str, value: str) -> str:
    """Return the normalized form of `value` for field `key`."""
    if is_date_field(key):
        return _normalize_date(value)
    if is_phone_field(key):
        return _normalize_phone(value)
    if is_money_field(key):
        return _normalize_money(value)
    if is_id_field(key):
        return _normalize_id(value)
    if is_sex_field(key):
        return _normalize_sex(value)
    if is_state_field(key):
        return _normalize_state(value)
    return _generic_normalize(value)


def compare(key: str, pred: str, gt: str) -> Tuple[bool, bool]:
    """Return (exact_match, normalized_match)."""
    p = "" if pred is None else str(pred)
    g = "" if gt is None else str(gt)
    exact = p == g
    norm = normalize(key, p) == normalize(key, g)
    return exact, norm
