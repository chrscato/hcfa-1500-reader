"""Build a Hugging Face dataset from a generated batch and (optionally) push it.

Reads `<data-dir>/manifest.jsonl` (produced by `python -m hcfa_synth generate`),
makes a tier-stratified train/validation/test split, and builds a DatasetDict whose
columns match what the v2 fine-tuning notebooks expect:

    image       : the rendered full-page form (datasets.Image, embedded on push)
    target      : json.dumps(flatten(logical))  — the whole-form flat JSON GT
    tier        : degradation tier
    sample_id   : zero-padded id
    prompt      : schema-style prompt (full canonical key list) — kept for v1/zero-shot
                  parity; the v2 region notebooks build their own prompt and ignore this.

Usage:
    # build + inspect only (no upload):
    python scripts/build_and_push_hf.py --data-dir data/v2 --dry-run

    # build + push (needs a WRITE token: `huggingface-cli login` or HF_TOKEN env):
    python scripts/build_and_push_hf.py --data-dir data/v2 --repo catochris/hcfa-1500-v2 --private

    # overwrite the v1 dataset instead (notebooks load this id unchanged):
    python scripts/build_and_push_hf.py --data-dir data/v2 --repo catochris/hcfa-1500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from hcfa_eval.dataset import _user_prompt_schema
from hcfa_eval.schema import flatten


def _read_manifest(data_dir: Path) -> List[dict]:
    manifest = data_dir / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"no manifest at {manifest} — did generation finish?")
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{manifest} is empty")
    return rows


def _load_logical(data_dir: Path, json_rel: str) -> Dict:
    # manifest stores Windows-style backslash paths; normalize.
    p = data_dir / json_rel.replace("\\", "/")
    return json.loads(p.read_text(encoding="utf-8"))["logical"]


def _canonical_keys(data_dir: Path, manifest: List[dict]) -> List[str]:
    """Union of flat keys across all samples, first-seen order (fixes service-line
    array width to the max seen). Used only for the schema `prompt` column."""
    seen, keys = set(), []
    for r in manifest:
        for k in flatten(_load_logical(data_dir, r["json"])):
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def _stratified_split(manifest: List[dict], val_frac: float, test_frac: float) -> Dict[str, List[dict]]:
    """Deterministic tier-stratified split. Sorted by sample_id so it's reproducible."""
    by_tier: Dict[str, List[dict]] = {}
    for r in sorted(manifest, key=lambda x: x["sample_id"]):
        by_tier.setdefault(r["tier"], []).append(r)

    out = {"train": [], "validation": [], "test": []}
    for tier, rows in by_tier.items():
        n = len(rows)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        # take val/test from the END so train stays the low-numbered, stable prefix
        out["test"].extend(rows[n - n_test:])
        out["validation"].extend(rows[n - n_test - n_val:n - n_test])
        out["train"].extend(rows[:n - n_test - n_val])
    return out


def _build_rows(data_dir: Path, rows: List[dict], prompt: str) -> List[dict]:
    built = []
    for r in rows:
        logical = _load_logical(data_dir, r["json"])
        img_path = (data_dir / r["image"].replace("\\", "/")).resolve()
        built.append({
            "image": str(img_path),                       # cast to datasets.Image below
            "target": json.dumps(flatten(logical), ensure_ascii=False),
            "tier": r["tier"],
            "sample_id": r["sample_id"],
            "prompt": prompt,
        })
    return built


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/v2", type=Path)
    ap.add_argument("--repo", default="catochris/hcfa-1500-v2",
                    help="HF dataset repo id to push to (use catochris/hcfa-1500 to overwrite v1)")
    ap.add_argument("--val-frac", default=0.10, type=float)
    ap.add_argument("--test-frac", default=0.10, type=float)
    ap.add_argument("--private", action="store_true", help="push as a private dataset")
    ap.add_argument("--dry-run", action="store_true", help="build + report only; do not push")
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict, Image  # lazy: only needed when actually building

    manifest = _read_manifest(args.data_dir)
    print(f"manifest: {len(manifest)} samples in {args.data_dir}")

    prompt = _user_prompt_schema(_canonical_keys(args.data_dir, manifest))
    splits = _stratified_split(manifest, args.val_frac, args.test_frac)

    dd = {}
    for split, rows in splits.items():
        ds = Dataset.from_list(_build_rows(args.data_dir, rows, prompt))
        ds = ds.cast_column("image", Image())
        dd[split] = ds
    dd = DatasetDict(dd)

    print("\nsplit sizes:")
    for split, ds in dd.items():
        from collections import Counter
        tiers = Counter(ds["tier"])
        print(f"  {split:11s}: {len(ds):5d}  tiers={dict(sorted(tiers.items()))}")
    print(f"\ntarget[0] (head): {dd['train'][0]['target'][:160]} ...")
    print(f"columns: {dd['train'].column_names}")

    if args.dry_run:
        print("\n--dry-run: built OK, not pushing. Re-run without --dry-run to upload.")
        return

    print(f"\npushing to https://huggingface.co/datasets/{args.repo} (private={args.private}) ...")
    dd.push_to_hub(args.repo, private=args.private)
    print("done.")


if __name__ == "__main__":
    main()
