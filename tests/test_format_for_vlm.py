"""Tests for the vision-language dataset formatter.

These exercise the target/prompt builders directly (no `datasets` needed) and,
when the library + generated data are present, round-trip one full HF example.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcfa_synth.format_for_vlm import (
    DROPPED_FIELDS,
    PROMPT_TEMPLATE,
    build_prompt,
    canonical_schema_keys,
    target_fields,
    target_json,
)

REPO = Path(__file__).resolve().parents[1]
BATCH = REPO / "data" / "full"
SPLITS = REPO / "splits" / "full"


def _has_data() -> bool:
    return (BATCH / "manifest.jsonl").exists() and (SPLITS / "train.jsonl").exists()


# ----- prompt template (no data / no datasets lib needed) --------------------


def test_prompt_is_fixed_and_embeds_keys():
    keys = ["box_3_patient_sex", "box_1_insurance_type"]
    p1 = build_prompt(keys)
    p2 = build_prompt(keys)
    assert p1 == p2  # deterministic → train/serve identical
    # every requested key is present in the embedded JSON template
    for k in keys:
        assert json.dumps(k) in p1
    # the template is the single source of truth
    assert "{schema}" in PROMPT_TEMPLATE


# ----- target string (needs generated label JSONs) --------------------------


def test_target_matches_eval_harness_flatten():
    if not _has_data():
        pytest.skip("synthetic data not generated")
    manifest = [
        json.loads(l)
        for l in (BATCH / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    json_path = BATCH / manifest[0]["json"]

    # target == flatten(logical) minus dropped fields, exactly as scored
    from hcfa_eval.schema import flatten

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    expected = {k: v for k, v in flatten(raw["logical"]).items() if k not in DROPPED_FIELDS}
    assert target_fields(json_path) == expected

    parsed = json.loads(target_json(json_path))
    assert parsed == expected
    assert all(isinstance(v, str) for v in parsed.values())


def test_dropped_fields_absent_from_target():
    if not _has_data():
        pytest.skip("synthetic data not generated")
    manifest = [
        json.loads(l)
        for l in (BATCH / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    fields = target_fields(BATCH / manifest[0]["json"])
    assert DROPPED_FIELDS.isdisjoint(fields)


# ----- full HF round-trip (needs `datasets`) --------------------------------


def test_example_round_trips_and_target_is_json(tmp_path):
    if not _has_data():
        pytest.skip("synthetic data not generated")
    datasets = pytest.importorskip("datasets")
    from PIL import Image as PILImage

    from hcfa_synth.format_for_vlm import build_split_dataset

    # tiny 2-row split so the test stays fast
    rows = [
        json.loads(l)
        for l in (SPLITS / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ][:2]
    mini = tmp_path / "mini.jsonl"
    mini.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    keys = canonical_schema_keys(BATCH, sample_count=20)
    prompt = build_prompt(keys)
    ds = build_split_dataset(mini, BATCH, prompt)

    assert len(ds) == 2
    assert isinstance(ds.features["image"], datasets.Image)

    ex = ds[0]
    # the Image feature decodes lazily to a PIL image
    assert isinstance(ex["image"], PILImage.Image)
    assert ex["image"].size[0] > 0 and ex["image"].size[1] > 0

    # target round-trips as a flat JSON object
    parsed = json.loads(ex["target"])
    assert isinstance(parsed, dict)
    assert len(parsed) > 20
    assert all(isinstance(v, str) for v in parsed.values())

    # prompt is identical across examples (fixed) and matches what we passed
    assert ds[0]["prompt"] == ds[1]["prompt"] == prompt

    # save_to_disk → load_from_disk preserves the example
    out = tmp_path / "hf_out"
    ds.save_to_disk(str(out))
    reloaded = datasets.load_from_disk(str(out))
    assert json.loads(reloaded[0]["target"]) == parsed
    assert reloaded[0]["prompt"] == prompt
    assert isinstance(reloaded[0]["image"], PILImage.Image)
