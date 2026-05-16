"""Tests for the PDF filler."""

import io

import pypdf

from hcfa_synth.pdf_fill import DEFAULT_TEMPLATE, fill_pdf, record_to_fields
from hcfa_synth.records import build_record


def test_template_exists():
    assert DEFAULT_TEMPLATE.exists(), f"template missing at {DEFAULT_TEMPLATE}"


def test_record_to_fields_returns_strings():
    rec = build_record(seed=1)
    fields = record_to_fields(rec)
    assert len(fields) > 100
    for name, value in fields.items():
        assert isinstance(value, str), f"{name}={value!r} is not str"


def test_filled_pdf_is_valid_and_round_trips():
    rec = build_record(seed=42)
    pdf_bytes = fill_pdf(rec)
    assert pdf_bytes.startswith(b"%PDF")

    r = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    fields = r.get_fields() or {}
    # A few representative fields should round-trip
    assert fields["pt_name"]["/V"] == record_to_fields(rec)["pt_name"]
    assert fields["cpt1"]["/V"] == rec["service_lines"][0]["cpt"]
    assert fields["diagnosis1"]["/V"] == rec["diagnoses"][0]
    assert fields["t_charge"]["/V"] == rec["billing"]["total_charge"]


def test_need_appearances_flag_is_set():
    rec = build_record(seed=1)
    pdf_bytes = fill_pdf(rec)
    r = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    acro = r.trailer["/Root"]["/AcroForm"]
    assert bool(acro["/NeedAppearances"]) is True
