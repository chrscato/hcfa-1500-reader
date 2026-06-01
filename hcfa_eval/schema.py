"""Flatten / unflatten the `logical` ground-truth view to a flat dict.

The model emits a flat JSON object — flat is easier for a 3B VLM to produce
correctly than deeply nested JSON, easier to constrain at decode time, and
trivially scoreable per-key.

Nested objects become dot-paths:
    box_5_patient_address.street

Arrays of objects (service_lines) become indexed paths:
    box_24_service_lines[0].procedure_code

Arrays of scalars (diagnoses, modifiers) are joined with ` | ` since they're
short and order-significant:
    box_21_diagnoses → "M15.9 | F90.9"

Empty strings, None, empty arrays, and empty objects all flatten to "".
That makes the model's job simpler: one universal "this field is blank"
output.
"""

from __future__ import annotations

import re
from typing import Any, Dict


SCALAR_LIST_JOIN = " | "


def flatten(logical: Dict[str, Any]) -> Dict[str, str]:
    """Flatten the `logical` GT view to a flat str→str dict.

    All values are coerced to strings. Empty/missing values become "".
    Key order follows the source dict (which mirrors form reading order).
    """
    out: Dict[str, str] = {}
    _walk(logical, prefix="", out=out)
    return out


def _walk(value: Any, prefix: str, out: Dict[str, str]) -> None:
    if value is None:
        out[prefix] = ""
        return
    if isinstance(value, dict):
        if not value:
            out[prefix] = ""
            return
        for k, v in value.items():
            child = f"{prefix}.{k}" if prefix else k
            _walk(v, child, out)
        return
    if isinstance(value, list):
        if not value:
            out[prefix] = ""
            return
        if all(not isinstance(x, (dict, list)) for x in value):
            out[prefix] = SCALAR_LIST_JOIN.join("" if x is None else str(x) for x in value)
            return
        for i, item in enumerate(value):
            _walk(item, f"{prefix}[{i}]", out)
        return
    # Scalar
    if isinstance(value, bool):
        out[prefix] = "YES" if value else "NO"
        return
    out[prefix] = str(value)


_INDEX_RE = re.compile(r"^(.*)\[(\d+)\]$")


def unflatten(flat: Dict[str, str]) -> Dict[str, Any]:
    """Inverse of flatten — useful when reconstructing nested GT for inspection.

    Note: this round-trips empty values back to "" rather than None/[]/{} since
    flatten() is lossy on that distinction. Good enough for diff/debug; do not
    use it to regenerate canonical GT.
    """
    root: Dict[str, Any] = {}
    for path, value in flat.items():
        _set_path(root, path, value)
    return root


def _set_path(root: Dict[str, Any], path: str, value: str) -> None:
    parts = path.split(".")
    cur: Any = root
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        m = _INDEX_RE.match(part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if key not in cur or not isinstance(cur[key], list):
                cur[key] = []
            while len(cur[key]) <= idx:
                cur[key].append({})
            if is_last:
                cur[key][idx] = value
            else:
                if not isinstance(cur[key][idx], dict):
                    cur[key][idx] = {}
                cur = cur[key][idx]
        else:
            if is_last:
                cur[part] = value
            else:
                if part not in cur or not isinstance(cur[part], dict):
                    cur[part] = {}
                cur = cur[part]
