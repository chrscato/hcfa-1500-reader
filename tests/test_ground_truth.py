"""Tests for the ground-truth JSON emitter."""

import json

from hcfa_synth.ground_truth import build_ground_truth
from hcfa_synth.records import build_record


def test_ground_truth_is_json_serializable():
    rec = build_record(seed=1)
    gt = build_ground_truth(rec)
    s = json.dumps(gt)
    assert isinstance(s, str)
    assert json.loads(s) == gt


def test_schema_has_required_top_level_keys():
    rec = build_record(seed=1)
    gt = build_ground_truth(rec)
    assert set(gt.keys()) >= {"schema_version", "meta", "fields", "logical"}
    assert gt["schema_version"] == "1.0"


def test_fields_layer_matches_pdf_field_mapping():
    rec = build_record(seed=1)
    gt = build_ground_truth(rec)
    assert gt["fields"]["pt_name"].startswith(rec["patient"]["name"]["last"])
    assert gt["fields"]["cpt1"] == rec["service_lines"][0]["cpt"]
    assert gt["fields"]["t_charge"] == rec["billing"]["total_charge"]


def test_logical_layer_has_box_structure():
    rec = build_record(seed=1)
    gt = build_ground_truth(rec)
    logical = gt["logical"]
    # A representative sample of expected logical keys
    expected = {
        "box_1_insurance_type",
        "box_2_patient_name",
        "box_3_patient_birth",
        "box_21_diagnoses",
        "box_24_service_lines",
        "box_28_total_charge",
        "box_33_billing_provider",
    }
    assert expected.issubset(logical.keys())


def test_logical_diagnoses_match_record():
    rec = build_record(seed=5)
    gt = build_ground_truth(rec)
    assert gt["logical"]["box_21_diagnoses"] == rec["diagnoses"]


def test_logical_service_lines_drop_empty_modifiers():
    rec = build_record(seed=5)
    gt = build_ground_truth(rec)
    for line in gt["logical"]["box_24_service_lines"]:
        assert all(m != "" for m in line["modifiers"])


def test_dob_format_handles_2digit_year_century():
    rec = build_record(seed=1)
    rec["patient"]["dob"] = {"mm": "06", "dd": "15", "yy": "85"}  # 1985
    rec["insured"]["dob"] = {"mm": "01", "dd": "01", "yy": "10"}  # 2010
    gt = build_ground_truth(rec)
    assert gt["logical"]["box_3_patient_birth"] == "06/15/1985"
    assert gt["logical"]["box_11a_insured_birth"] == "01/01/2010"
