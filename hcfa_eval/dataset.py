"""Qwen2.5-VL chat-format dataset builder.

Produces a list of rows (or an iterator) that any HF-style trainer can
consume. We do NOT tokenize here — the trainer's processor handles that.
Keeping this transformer-agnostic means the same loader works for Unsloth,
LLaMA-Factory, ms-swift, and bare `transformers`.

Each row:
    {
      "sample_id": str,
      "tier": str,
      "split": str,
      "image_path": absolute str (always; trainers don't like relative),
      "messages": [
          {"role": "system",   "content": [{"type": "text", "text": SYSTEM}]},
          {"role": "user",     "content": [
              {"type": "image", "image": <abs path>},
              {"type": "text",  "text": USER_PROMPT},
          ]},
          {"role": "assistant","content": [{"type": "text", "text": <gt json>}]},
      ],
      "target_json": str  # convenience copy of the assistant text
    }

Two prompt styles:
  - "minimal" : terse — assumes the model has learned the schema in FT
  - "schema"  : includes the full flat key list as a JSON template (use for
                zero-shot eval where the model has no prior knowledge of the
                target schema)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

from hcfa_eval.schema import flatten

PromptStyle = Literal["minimal", "schema"]

SYSTEM_PROMPT = (
    "You are a precise medical-claim extractor. You read CMS-1500 (HCFA) "
    "forms and emit a flat JSON object of field values. Use empty string "
    "for blank fields. Do not invent values. Do not include commentary."
)

USER_PROMPT_MINIMAL = (
    "Extract every field from this CMS-1500 form as a single flat JSON "
    "object. Return only the JSON."
)


def _user_prompt_schema(field_keys: List[str]) -> str:
    template = {k: "" for k in field_keys}
    return (
        "Extract every field from this CMS-1500 form. Return only a flat "
        "JSON object with EXACTLY these keys (use empty string for blank "
        "fields, preserve key order):\n"
        + json.dumps(template, indent=2)
    )


def _load_gt(json_path: Path) -> Dict[str, str]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return flatten(raw["logical"])


def build_row(
    manifest_row: dict,
    batch_dir: Path,
    *,
    prompt_style: PromptStyle = "minimal",
    schema_keys: Optional[List[str]] = None,
) -> dict:
    """Build one chat-format row from a manifest line + the batch dir it lives in."""
    image_path = (batch_dir / manifest_row["image"]).resolve()
    json_path = (batch_dir / manifest_row["json"]).resolve()
    gt_flat = _load_gt(json_path)
    target_json = json.dumps(gt_flat, ensure_ascii=False)

    if prompt_style == "schema":
        keys = schema_keys if schema_keys is not None else list(gt_flat.keys())
        user_text = _user_prompt_schema(keys)
    else:
        user_text = USER_PROMPT_MINIMAL

    return {
        "sample_id": manifest_row["sample_id"],
        "tier": manifest_row["tier"],
        "split": manifest_row.get("split", ""),
        "image_path": str(image_path),
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_json}],
            },
        ],
        "target_json": target_json,
    }


def load_split(
    split_path: Path,
    batch_dir: Path,
    *,
    prompt_style: PromptStyle = "minimal",
    schema_keys: Optional[List[str]] = None,
) -> List[dict]:
    """Eagerly load all rows from a split JSONL. Fine for <10k samples."""
    rows = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(
            build_row(
                json.loads(line),
                batch_dir,
                prompt_style=prompt_style,
                schema_keys=schema_keys,
            )
        )
    return rows


def iter_split(
    split_path: Path,
    batch_dir: Path,
    *,
    prompt_style: PromptStyle = "minimal",
    schema_keys: Optional[List[str]] = None,
) -> Iterable[dict]:
    """Lazy iterator — use this in HF `IterableDataset` wrappers."""
    with split_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield build_row(
                json.loads(line),
                batch_dir,
                prompt_style=prompt_style,
                schema_keys=schema_keys,
            )


def derive_schema_keys(batch_dir: Path, *, sample_count: int = 50) -> List[str]:
    """Derive the canonical flat key list by scanning the first N GT files.

    Service lines vary in length per sample; the canonical schema fixes the
    array width to the max seen. The model can always emit "" for trailing
    lines that don't apply.
    """
    manifest_path = batch_dir / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[:sample_count]
    all_keys: List[str] = []
    seen = set()
    for r in rows:
        gt = flatten(json.loads((batch_dir / r["json"]).read_text(encoding="utf-8"))["logical"])
        for k in gt:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    return all_keys
