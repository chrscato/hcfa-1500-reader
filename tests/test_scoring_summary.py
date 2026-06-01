"""Tests for the summary metrics layer in hcfa_eval.scoring.

These build tiny synthetic batches in tmp_path, so they don't need the full
generated dataset. The per-key `score()` path is covered in test_eval.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hcfa_eval.scoring import (
    CER_CLASSES,
    FIELD_CLASSES,
    char_error_rate,
    classify_field,
    format_summary,
    summarize,
    summary_csv_row,
    write_summary_csv,
)


# ----- field classification --------------------------------------------------


def test_classify_field_maps_each_class():
    cases = {
        "box_24_service_lines[0].procedure_code": "codes",
        "box_21_diagnoses": "codes",
        "box_24_service_lines[1].charges": "money",
        "box_28_total_charge": "money",
        "box_29_amount_paid": "money",
        "box_3_patient_birth": "dates",
        "box_24_service_lines[0].date_from": "dates",
        "box_33_billing_provider.npi": "npis",
        "box_24_service_lines[0].rendering_provider_npi": "npis",
        "box_2_patient_name.last": "names",
        "box_17_referring_provider_name": "names",
        "box_5_patient_address.street": "addresses",
        "insurance_carrier.city_state_zip": "addresses",
    }
    for key, cls in cases.items():
        assert classify_field(key) == cls, key

    # not in any class
    assert classify_field("box_3_patient_sex") is None
    # a phone living on an address is not "address text"
    assert classify_field("box_5_patient_address.phone") is None


def test_classes_are_disjoint():
    for key in [
        "box_24_service_lines[0].charges",
        "box_24_service_lines[0].procedure_code",
        "box_33_billing_provider.npi",
        "box_2_patient_name.last",
        "box_5_patient_address.zip",
        "box_3_patient_birth",
    ]:
        matches = [c for c in FIELD_CLASSES if classify_field(key) == c]
        assert len(matches) == 1, (key, matches)


# ----- character error rate --------------------------------------------------


def test_char_error_rate():
    assert char_error_rate("Q3014", "Q3014") == 0.0
    assert char_error_rate("", "") == 0.0
    assert char_error_rate("X", "") == 1.0  # output against empty target
    assert char_error_rate("", "ABC") == 1.0  # missed everything
    assert char_error_rate("ABD", "ABC") == pytest.approx(1 / 3)  # 1 substitution
    assert char_error_rate("AB", "ABC") == pytest.approx(1 / 3)  # 1 deletion
    assert char_error_rate("totally-different", "x") == 1.0  # capped at 1.0


# ----- synthetic batch helpers ----------------------------------------------


def _mk_batch(tmp_path: Path, samples):
    """samples: list of (sample_id, tier, logical_dict)."""
    batch = tmp_path / "batch"
    batch.mkdir(exist_ok=True)
    rows = []
    for sid, tier, logical in samples:
        (batch / f"{sid}.json").write_text(json.dumps({"logical": logical}), encoding="utf-8")
        rows.append({"sample_id": sid, "tier": tier, "json": f"{sid}.json"})
    split = tmp_path / "split.jsonl"
    split.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return batch, split


def _write_preds(tmp_path: Path, preds):
    """preds: list of dicts already shaped {sample_id, fields, [raw]}."""
    p = tmp_path / "preds.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in preds), encoding="utf-8")
    return p


# ----- summarize -------------------------------------------------------------


def test_summary_perfect_predictions(tmp_path):
    logical = {
        "box_28_total_charge": "629.92",
        "box_24_service_lines[0].procedure_code": "Q3014",
        "box_33_billing_provider.npi": "1234567893",
        "box_2_patient_name.last": "ZAMORA",
        "box_9_other_insured": "",  # blank GT
    }
    batch, split = _mk_batch(tmp_path, [("s1", "pristine", logical)])
    preds = _write_preds(tmp_path, [{"sample_id": "s1", "fields": dict(logical)}])

    s = summarize(split, preds, batch, model="perfect")
    assert s["overall"]["populated"]["norm_acc"] == 1.0
    assert s["overall"]["blank"]["norm_acc"] == 1.0
    assert s["by_tier_populated"]["pristine"]["norm_acc"] == 1.0
    for cls in CER_CLASSES:
        assert s["cer"][cls]["mean_cer"] == 0.0
    jv = s["json_validity"]
    assert jv["valid_rate"] == 1.0 and jv["parse_rate"] == 1.0


def test_summary_separates_populated_and_blank(tmp_path):
    # GT: one populated, one blank. Pred hallucinates into the blank field and
    # nails the populated one -> populated acc 1.0, blank acc 0.0.
    logical = {"box_2_patient_name.last": "ZAMORA", "box_9_other_insured": ""}
    batch, split = _mk_batch(tmp_path, [("s1", "pristine", logical)])
    preds = _write_preds(
        tmp_path,
        [{"sample_id": "s1", "fields": {"box_2_patient_name.last": "ZAMORA", "box_9_other_insured": "OOPS"}}],
    )
    s = summarize(split, preds, batch)
    assert s["overall"]["populated"]["norm_acc"] == 1.0
    assert s["overall"]["populated"]["total"] == 1
    assert s["overall"]["blank"]["norm_acc"] == 0.0  # hallucination caught
    assert s["overall"]["blank"]["total"] == 1


def test_summary_class_exact_and_cer_move_on_errors(tmp_path):
    logical = {
        "box_28_total_charge": "629.92",
        "box_24_service_lines[0].procedure_code": "Q3014",
    }
    batch, split = _mk_batch(tmp_path, [("s1", "pristine", logical)])
    # one-character error in each structured value
    preds = _write_preds(
        tmp_path,
        [{"sample_id": "s1", "fields": {"box_28_total_charge": "629.93", "box_24_service_lines[0].procedure_code": "Q3015"}}],
    )
    s = summarize(split, preds, batch)
    assert s["by_class"]["money"]["exact_acc"] == 0.0
    assert s["by_class"]["codes"]["exact_acc"] == 0.0
    assert s["cer"]["money"]["mean_cer"] > 0.0
    assert s["cer"]["codes"]["mean_cer"] > 0.0
    # CER stays small for a single-char slip
    assert s["cer"]["overall"]["mean_cer"] < 0.5


def test_summary_class_exact_ignores_blank_cells(tmp_path):
    # An unused service line (all blank) must not inflate class exact-match:
    # populated count for that class should reflect only the real value.
    logical = {
        "box_24_service_lines[0].procedure_code": "Q3014",
        "box_24_service_lines[1].procedure_code": "",  # blank/unused
    }
    batch, split = _mk_batch(tmp_path, [("s1", "pristine", logical)])
    preds = _write_preds(
        tmp_path,
        [{"sample_id": "s1", "fields": {"box_24_service_lines[0].procedure_code": "Q3014", "box_24_service_lines[1].procedure_code": ""}}],
    )
    s = summarize(split, preds, batch)
    assert s["by_class"]["codes"]["populated"] == 1
    assert s["by_class"]["codes"]["total"] == 2
    assert s["by_class"]["codes"]["exact_acc"] == 1.0


def test_summary_json_validity(tmp_path):
    logical = {"a": "1", "b": "2"}
    batch, split = _mk_batch(
        tmp_path, [("s1", "pristine", logical), ("s2", "fax", dict(logical))]
    )
    preds = _write_preds(
        tmp_path,
        [
            {"sample_id": "s1", "fields": {"a": "1", "b": "2"}},  # valid + complete
            {"sample_id": "s2", "fields": {"a": "3"}},  # missing key "b"
        ],
    )
    s = summarize(split, preds, batch)
    jv = s["json_validity"]
    assert jv["parse_rate"] == 1.0  # both lines parse
    assert jv["has_expected_keys_rate"] == 0.5  # s2 incomplete
    assert jv["valid_rate"] == 0.5
    assert jv["mean_key_coverage"] == pytest.approx(0.75)  # (2/2 + 1/2) / 2


def test_summary_json_validity_uses_raw_when_present(tmp_path):
    logical = {"a": "1"}
    batch, split = _mk_batch(tmp_path, [("s1", "pristine", logical)])
    # The wrapper line is valid JSON, but the model's raw text is not.
    preds = _write_preds(
        tmp_path, [{"sample_id": "s1", "fields": {"a": "1"}, "raw": "{ not valid json"}]
    )
    s = summarize(split, preds, batch)
    assert s["json_validity"]["parse_rate"] == 0.0  # judged on raw model output


def test_summary_counts_missing_predictions(tmp_path):
    logical = {"a": "1"}
    batch, split = _mk_batch(
        tmp_path, [("s1", "pristine", logical), ("s2", "fax", dict(logical))]
    )
    preds = _write_preds(tmp_path, [{"sample_id": "s1", "fields": {"a": "1"}}])
    s = summarize(split, preds, batch)
    assert s["missing_predictions"] == ["s2"]
    assert s["json_validity"]["valid_rate"] == 0.5  # s2 absent -> invalid


# ----- CSV output ------------------------------------------------------------


def test_csv_row_has_stable_schema(tmp_path):
    logical = {"box_28_total_charge": "629.92"}
    batch, split = _mk_batch(tmp_path, [("s1", "pristine", logical)])
    preds = _write_preds(tmp_path, [{"sample_id": "s1", "fields": dict(logical)}])
    row = summary_csv_row(summarize(split, preds, batch, model="m1"))
    assert row["model"] == "m1"
    # tier columns present for ALL canonical tiers (stable header)
    for tier in ["pristine", "clean_scan", "worn_scan", "fax", "phone_photo", "worst"]:
        assert f"tier_{tier}_norm_acc" in row
    # class + cer columns
    for cls in FIELD_CLASSES:
        assert f"class_{cls}_exact" in row
    for cls in CER_CLASSES:
        assert f"cer_{cls}" in row
    assert "json_valid_rate" in row
    # tiers with no populated cells render as blank, not a number
    assert row["tier_fax_norm_acc"] == ""


def test_write_summary_csv_appends_without_duplicate_header(tmp_path):
    logical = {"box_28_total_charge": "629.92"}
    batch, split = _mk_batch(tmp_path, [("s1", "pristine", logical)])
    preds = _write_preds(tmp_path, [{"sample_id": "s1", "fields": dict(logical)}])
    out = tmp_path / "runs.csv"
    write_summary_csv(summarize(split, preds, batch, model="a"), out)
    write_summary_csv(summarize(split, preds, batch, model="b"), out)

    rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
    assert [r["model"] for r in rows] == ["a", "b"]
    # exactly one header line
    assert out.read_text(encoding="utf-8").count("model,n_samples") == 1


def test_format_summary_is_printable(tmp_path):
    logical = {"box_28_total_charge": "629.92", "box_2_patient_name.last": "ZAMORA"}
    batch, split = _mk_batch(tmp_path, [("s1", "pristine", logical)])
    preds = _write_preds(tmp_path, [{"sample_id": "s1", "fields": dict(logical)}])
    text = format_summary(summarize(split, preds, batch, model="m"))
    assert "Summary: m" in text
    assert "Per-tier" in text and "field-class" in text and "JSON validity" in text
    text.encode("ascii")  # no non-ASCII that would break a plain terminal
