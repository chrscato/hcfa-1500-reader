"""Smoke tests for the eval harness — flatten / normalize / scoring."""

from __future__ import annotations

import json
from pathlib import Path

from hcfa_eval.dataset import build_row, derive_schema_keys
from hcfa_eval.normalize import compare, normalize
from hcfa_eval.schema import flatten, unflatten
from hcfa_eval.scoring import format_report, score
from hcfa_eval.splits import make_splits


# ----- flatten / unflatten ---------------------------------------------------


def test_flatten_scalars_and_nested():
    src = {
        "box_3_patient_birth": "05/17/1957",
        "box_5_patient_address": {"street": "8764 HOWARD FORGE", "city": "WEST DONALD"},
    }
    flat = flatten(src)
    assert flat["box_3_patient_birth"] == "05/17/1957"
    assert flat["box_5_patient_address.street"] == "8764 HOWARD FORGE"
    assert flat["box_5_patient_address.city"] == "WEST DONALD"


def test_flatten_array_of_objects_indexed():
    src = {"lines": [{"cpt": "Q3014"}, {"cpt": "J2792"}]}
    flat = flatten(src)
    assert flat["lines[0].cpt"] == "Q3014"
    assert flat["lines[1].cpt"] == "J2792"


def test_flatten_scalar_list_joined():
    src = {"diagnoses": ["M15.9", "F90.9"]}
    assert flatten(src)["diagnoses"] == "M15.9 | F90.9"


def test_flatten_blank_collapse():
    src = {"a": "", "b": None, "c": [], "d": {}}
    flat = flatten(src)
    assert flat == {"a": "", "b": "", "c": "", "d": ""}


def test_unflatten_roundtrip_shapes():
    src = {
        "lines": [{"cpt": "Q3014", "ch": "263.78"}, {"cpt": "J2792", "ch": "366.14"}],
        "patient": {"name": "ZAMORA, SARAH Y"},
    }
    flat = flatten(src)
    back = unflatten(flat)
    assert back["lines"][0]["cpt"] == "Q3014"
    assert back["patient"]["name"] == "ZAMORA, SARAH Y"


# ----- normalize / compare ---------------------------------------------------


def test_normalize_dates():
    assert normalize("box_3_patient_birth", "5/17/1957") == "05/17/1957"
    assert normalize("box_3_patient_birth", "05/17/57") == "05/17/1957"
    assert normalize("box_3_patient_birth", "1957-05-17") == "05/17/1957"


def test_normalize_phone_digits_only():
    assert normalize("box_5_patient_address.phone", "(456) 745-3407") == "4567453407"


def test_normalize_money_two_decimals():
    assert normalize("box_28_total_charge", "$629.92") == "629.92"
    assert normalize("box_28_total_charge", "629.9") == "629.90"


def test_normalize_blank_tokens_equivalent():
    assert normalize("anything", "") == normalize("anything", "N/A") == ""


def test_compare_returns_both_flags():
    exact, norm = compare("box_28_total_charge", "$629.92", "629.92")
    assert exact is False
    assert norm is True


# ----- splits ---------------------------------------------------------------


def test_make_splits_stratifies_and_is_deterministic():
    rows = []
    for tier in ["pristine", "fax", "worst"]:
        for i in range(30):
            rows.append({"sample_id": f"{tier}_{i:03d}", "tier": tier})
    s1 = make_splits(rows, seed=0)
    s2 = make_splits(rows, seed=0)
    # determinism
    assert [r["sample_id"] for r in s1["train"]] == [r["sample_id"] for r in s2["train"]]
    # stratification — each tier should appear in each split
    for split_name, items in s1.items():
        tiers_in_split = {r["tier"] for r in items}
        assert tiers_in_split == {"pristine", "fax", "worst"}, (split_name, tiers_in_split)
    # totals add up
    assert len(s1["train"]) + len(s1["val"]) + len(s1["test"]) == 90


# ----- end-to-end against real data -----------------------------------------


REPO = Path(__file__).resolve().parents[1]
FULL = REPO / "data" / "full"


def test_dataset_row_round_trip_real_sample():
    if not (FULL / "manifest.jsonl").exists():
        return  # skip if data not generated
    manifest = [json.loads(l) for l in (FULL / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    row = build_row(manifest[0], FULL, prompt_style="minimal")
    assert row["sample_id"] == manifest[0]["sample_id"]
    assert row["tier"] == manifest[0]["tier"]
    # assistant text must be parseable JSON and non-trivial
    target = json.loads(row["target_json"])
    assert isinstance(target, dict)
    assert len(target) > 20  # many fields
    # image path is absolute and exists
    assert Path(row["image_path"]).is_absolute()
    assert Path(row["image_path"]).exists()


def test_perfect_predictions_score_100(tmp_path):
    """Feed back the GT as predictions — every tier should score 1.0."""
    if not (FULL / "manifest.jsonl").exists():
        return
    splits = make_splits(
        [json.loads(l) for l in (FULL / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()],
        seed=0,
    )
    # tiny eval slice for speed
    test_slice = splits["test"][:6]
    split_path = tmp_path / "test_slice.jsonl"
    split_path.write_text("\n".join(json.dumps(r) for r in test_slice), encoding="utf-8")

    pred_lines = []
    for row in test_slice:
        raw = json.loads((FULL / row["json"]).read_text(encoding="utf-8"))
        pred_lines.append(json.dumps({"sample_id": row["sample_id"], "fields": flatten(raw["logical"])}))
    pred_path = tmp_path / "preds.jsonl"
    pred_path.write_text("\n".join(pred_lines), encoding="utf-8")

    rep = score(split_path, pred_path, FULL)
    text = format_report(rep)
    assert rep["overall_doc"].total == len(test_slice)
    assert rep["overall_doc"].all_correct == len(test_slice), text


def test_derive_schema_keys_grows_with_service_lines():
    if not (FULL / "manifest.jsonl").exists():
        return
    keys = derive_schema_keys(FULL, sample_count=50)
    # must include canonical singletons
    assert "box_1_insurance_type" in keys
    assert "box_3_patient_birth" in keys
    # must expand service-line indices beyond [0]
    indexed = [k for k in keys if k.startswith("box_24_service_lines[")]
    line_indices = {int(k.split("[")[1].split("]")[0]) for k in indexed}
    assert max(line_indices) >= 2
