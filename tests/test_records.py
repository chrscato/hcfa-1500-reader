"""Tests for the synthetic record builder."""

from datetime import date

from hcfa_synth.npi import luhn_valid
from hcfa_synth.records import build_record


def test_record_is_deterministic_for_same_seed():
    a = build_record(seed=42)
    b = build_record(seed=42)
    assert a == b


def test_record_differs_with_different_seed():
    a = build_record(seed=1)
    b = build_record(seed=2)
    assert a != b


def test_service_dates_not_in_future():
    rec = build_record(seed=42)
    today = date.today()
    for line in rec["service_lines"]:
        yy = int(line["date_from"]["yy"])
        full_year = 2000 + yy if yy < 50 else 1900 + yy
        sd = date(full_year, int(line["date_from"]["mm"]), int(line["date_from"]["dd"]))
        assert sd <= today, f"service date {sd} is in the future"


def test_total_charge_equals_sum_of_lines():
    for seed in range(20):
        rec = build_record(seed=seed)
        lines_sum = sum(float(line["charges"]) for line in rec["service_lines"])
        total = float(rec["billing"]["total_charge"])
        assert abs(lines_sum - total) < 0.01


def test_all_npis_valid():
    for seed in range(20):
        rec = build_record(seed=seed)
        assert luhn_valid(rec["service_facility"]["npi"])
        assert luhn_valid(rec["billing_office"]["npi"])
        for line in rec["service_lines"]:
            assert luhn_valid(line["rendering_provider_npi"])
        ref_npi = rec["referring_provider"]["npi"]
        if ref_npi:
            assert luhn_valid(ref_npi)


def test_diagnosis_pointers_reference_populated_diagnoses():
    for seed in range(20):
        rec = build_record(seed=seed)
        n_diags = len(rec["diagnoses"])
        valid_pointers = set("ABCDEFGHIJKL"[:n_diags])
        for line in rec["service_lines"]:
            for pointer in line["diagnosis_pointers"]:
                assert pointer in valid_pointers


def test_line_count_in_range():
    for seed in range(50):
        rec = build_record(seed=seed)
        assert 1 <= len(rec["service_lines"]) <= 6


def test_diagnosis_count_in_range():
    for seed in range(50):
        rec = build_record(seed=seed)
        assert 1 <= len(rec["diagnoses"]) <= 12


def test_self_relationship_mirrors_patient_to_insured():
    # Sample enough seeds to find a "self" relationship and confirm mirroring.
    for seed in range(100):
        rec = build_record(seed=seed)
        if rec["patient"]["relationship_to_insured"] == "self":
            assert rec["insured"]["name"] == rec["patient"]["name"]
            assert rec["insured"]["dob"] == rec["patient"]["dob"]
            assert rec["insured"]["sex"] == rec["patient"]["sex"]
            return
    raise AssertionError("no 'self' relationship found in 100 seeds — RNG may be off")


def test_tax_id_format_matches_type():
    for seed in range(50):
        rec = build_record(seed=seed)
        billing = rec["billing"]
        if billing["tax_id_type"] == "SSN":
            parts = billing["tax_id"].split("-")
            assert len(parts) == 3 and len(parts[0]) == 3 and len(parts[1]) == 2 and len(parts[2]) == 4
        else:
            parts = billing["tax_id"].split("-")
            assert len(parts) == 2 and len(parts[0]) == 2 and len(parts[1]) == 7
