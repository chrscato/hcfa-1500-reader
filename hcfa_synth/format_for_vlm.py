"""Format the synthetic CMS-1500 splits into a vision-language FT dataset.

Reads ``splits/full/{train,val,test}.jsonl`` plus the per-image label JSONs
and emits a Hugging Face ``DatasetDict`` whose rows are::

    {
      "image":  <PNG, datasets Image feature — decodes lazily to PIL>,
      "prompt": <the FIXED extraction instruction, identical for every row>,
      "target": <flat JSON string the eval harness grades against>,
      "sample_id": str, "tier": str, "split": str,   # passthrough metadata
    }

Target string
-------------
The target is ``json.dumps(flatten(raw["logical"]))`` with the dropped fields
removed — i.e. **exactly** what ``hcfa_eval.scoring`` grades against
(see ``hcfa_eval/scoring.py``: it scores ``flatten(raw["logical"])``). We
deliberately flatten the *logical* GT view rather than the raw PDF ``fields``
layer: the harness never scores the PDF field names, so training to them would
optimize a schema we don't measure. ``flatten()`` is imported from
``hcfa_eval.schema`` so the format can never drift from the scorer.

Prompt parity
-------------
The instruction lives in ONE place — ``PROMPT_TEMPLATE`` + ``build_prompt()`` —
and the resolved string is baked identically into every example. Serving code
must build its prompt the same way (``build_prompt(canonical_schema_keys())``)
so train and inference stay byte-for-byte identical.

CLI::

    python -m hcfa_synth.format_for_vlm --out data/hf_vlm
    python -m hcfa_synth.format_for_vlm --out data/hf_vlm --push --hub-repo-id you/hcfa-cms1500-vlm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from hcfa_eval.schema import flatten

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS_DIR = REPO / "splits" / "full"
DEFAULT_BATCH_DIR = REPO / "data" / "full"
DEFAULT_OUT_DIR = REPO / "data" / "hf_vlm"

# Source split files → Hugging Face split names (HF idiom uses "validation").
SPLIT_FILES = {"train": "train", "val": "validation", "test": "test"}

# Logical flat keys to exclude from the target. Prompt 1 fixed the checkbox
# /AP appearance rendering cleanly, so NO fields were dropped — this stays
# empty. Had the fix been infeasible, the button fields' *logical* keys
# (box_1_insurance_type, box_3_patient_sex, box_11a_insured_sex,
# box_20_outside_lab, box_25_tax_id_type, box_27_accept_assignment, etc.)
# would be listed here so the target and the scorer drop them together.
DROPPED_FIELDS: frozenset = frozenset()

# Single source of truth for the instruction. ``{schema}`` is filled with the
# canonical flat key list so the model sees exactly which keys to emit.
PROMPT_TEMPLATE = (
    "You are a precise medical-claim extractor. Read this CMS-1500 (HCFA) "
    "claim form image and return a SINGLE flat JSON object.\n"
    "Rules:\n"
    '- Output ONLY the JSON object — no prose, no code fences.\n'
    '- Include EXACTLY these keys, in this order.\n'
    '- Use "" (empty string) for any field that is blank or not present.\n'
    "- Do not invent values.\n"
    "Keys:\n"
    "{schema}"
)


def build_prompt(schema_keys: List[str]) -> str:
    """Render the fixed instruction for a given ordered key list.

    Used for BOTH training (here) and serving — keep them identical by always
    calling this with ``canonical_schema_keys()``.
    """
    template = {k: "" for k in schema_keys}
    return PROMPT_TEMPLATE.format(schema=json.dumps(template, indent=2, ensure_ascii=False))


def _read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_fields(json_path: Path) -> Dict[str, str]:
    """Flat str→str target for one label JSON: ``flatten(logical)`` minus drops."""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    flat = flatten(raw["logical"])
    return {k: v for k, v in flat.items() if k not in DROPPED_FIELDS}


def target_json(json_path: Path) -> str:
    """The assistant/target string — exactly what hcfa_eval scores against."""
    return json.dumps(target_fields(json_path), ensure_ascii=False)


def canonical_schema_keys(
    batch_dir: Path = DEFAULT_BATCH_DIR,
    *,
    sample_count: Optional[int] = None,
) -> List[str]:
    """Ordered union of every target key across the manifest, minus drops.

    Order is first-seen in form reading order. Service-line array indices
    expand to the widest sample seen, so the schema (and thus the prompt) is
    fixed for every example — the model emits "" for trailing unused lines.
    """
    manifest = _read_jsonl(batch_dir / "manifest.jsonl")
    if sample_count is not None:
        manifest = manifest[:sample_count]
    keys: List[str] = []
    seen: set = set()
    for row in manifest:
        for k in target_fields(batch_dir / row["json"]):
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def _split_columns(split_path: Path, batch_dir: Path, prompt: str) -> Dict[str, list]:
    cols: Dict[str, list] = {
        "image": [],
        "prompt": [],
        "target": [],
        "sample_id": [],
        "tier": [],
        "split": [],
    }
    for row in _read_jsonl(split_path):
        cols["image"].append(str((batch_dir / row["image"]).resolve()))
        cols["prompt"].append(prompt)
        cols["target"].append(target_json(batch_dir / row["json"]))
        cols["sample_id"].append(row["sample_id"])
        cols["tier"].append(row["tier"])
        cols["split"].append(row.get("split", split_path.stem))
    return cols


def _features():
    # Imported lazily so the rest of the module (constants, target builders)
    # is importable without the optional `datasets` dependency installed.
    from datasets import Features, Image, Value

    return Features(
        {
            "image": Image(),
            "prompt": Value("string"),
            "target": Value("string"),
            "sample_id": Value("string"),
            "tier": Value("string"),
            "split": Value("string"),
        }
    )


def build_split_dataset(split_path: Path, batch_dir: Path, prompt: str):
    """Build one ``datasets.Dataset`` for a single split file."""
    from datasets import Dataset

    return Dataset.from_dict(_split_columns(split_path, batch_dir, prompt), features=_features())


def build_dataset_dict(
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    batch_dir: Path = DEFAULT_BATCH_DIR,
):
    """Build the full ``DatasetDict`` (train/validation/test) from the splits.

    The prompt is computed once from the canonical schema and shared across
    every example in every split, so it is identical everywhere.
    """
    from datasets import DatasetDict

    prompt = build_prompt(canonical_schema_keys(batch_dir))
    dd = {}
    for file_stem, hf_name in SPLIT_FILES.items():
        split_path = splits_dir / f"{file_stem}.jsonl"
        if not split_path.exists():
            continue
        dd[hf_name] = build_split_dataset(split_path, batch_dir, prompt)
    if not dd:
        raise FileNotFoundError(f"no split files found under {splits_dir}")
    return DatasetDict(dd)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Format CMS-1500 splits into a VLM fine-tuning dataset.")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="dir with {train,val,test}.jsonl")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR, help="dir with manifest + per-image label JSONs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="save_to_disk destination")
    parser.add_argument("--push", action="store_true", help="also push the dataset to the Hugging Face Hub")
    parser.add_argument("--hub-repo-id", default=None, help="Hub repo id for --push, e.g. user/hcfa-cms1500-vlm")
    parser.add_argument("--private", action="store_true", help="push as a private Hub dataset")
    args = parser.parse_args(argv)

    if args.push and not args.hub_repo_id:
        parser.error("--push requires --hub-repo-id")

    dataset = build_dataset_dict(args.splits_dir, args.batch_dir)
    sizes = ", ".join(f"{name}={len(ds)}" for name, ds in dataset.items())
    print(f"Built dataset: {sizes}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.out))
    print(f"Saved to {args.out}  (load with datasets.load_from_disk)")

    if args.push:
        dataset.push_to_hub(args.hub_repo_id, private=args.private)
        print(f"Pushed to https://huggingface.co/datasets/{args.hub_repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
