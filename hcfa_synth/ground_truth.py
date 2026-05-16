"""Ground-truth JSON emitter.

Two-layer schema per sample:

  `fields`   — flat dict keyed by PDF field name. Matches what an extractor
               reading the raw AcroForm would naively see (radio export
               codes like "M" for spouse, "Tricare" instead of "TRICARE").

  `logical`  — nested structure organized by the 33 CMS-1500 boxes, with
               normalized values an extractor reading the rendered image
               would naturally produce.

Eval harnesses can score against either layer depending on what they're
benchmarking.
"""

from __future__ import annotations

from typing import Any, Dict

from hcfa_synth.pdf_fill import record_to_fields


def _format_dob(dob: Dict[str, str]) -> str:
    """MM/DD/YYYY — convert 2-digit year using century heuristic (≥50 → 19xx)."""
    yy = int(dob["yy"])
    yyyy = 2000 + yy if yy < 50 else 1900 + yy
    return f"{dob['mm']}/{dob['dd']}/{yyyy:04d}"


def _format_mmddyy_blank_safe(d: Dict[str, str]) -> str:
    if not d["mm"]:
        return ""
    return _format_dob(d)


def _logical_view(record: Dict[str, Any]) -> Dict[str, Any]:
    """Emit a human-readable, box-organized view of the record."""
    pt = record["patient"]
    insured = record["insured"]
    ins = record["insurance"]
    cond = record["condition"]
    d = record["dates"]
    ref = record["referring_provider"]
    bill = record["billing"]
    fac = record["service_facility"]
    bo = record["billing_office"]

    return {
        "box_1_insurance_type": ins["type"],
        "box_1a_insured_id": ins["insured_id"],
        "box_2_patient_name": {
            "first": pt["name"]["first"],
            "middle": pt["name"]["middle"],
            "last": pt["name"]["last"],
        },
        "box_3_patient_birth": _format_dob(pt["dob"]),
        "box_3_patient_sex": pt["sex"],
        "box_4_insured_name": {
            "first": insured["name"]["first"],
            "middle": insured["name"]["middle"],
            "last": insured["name"]["last"],
        },
        "box_5_patient_address": {
            "street": pt["address"]["street"],
            "city": pt["address"]["city"],
            "state": pt["address"]["state"],
            "zip": pt["address"]["zip"],
            "phone": f"({pt['phone']['area']}) {pt['phone']['number']}",
        },
        "box_6_relationship_to_insured": pt["relationship_to_insured"],
        "box_7_insured_address": {
            "street": insured["address"]["street"],
            "city": insured["address"]["city"],
            "state": insured["address"]["state"],
            "zip": insured["address"]["zip"],
            "phone": f"({insured['phone']['area']}) {insured['phone']['number']}",
        },
        "box_9_other_insured": record["other_insured"],
        "box_10_condition": {
            "employment_related": cond["employment_related"],
            "auto_accident": cond["auto_accident"],
            "auto_accident_state": cond["auto_accident_state"],
            "other_accident": cond["other_accident"],
        },
        "box_11_insured_policy_group": insured["policy_group"],
        "box_11a_insured_birth": _format_dob(insured["dob"]),
        "box_11a_insured_sex": insured["sex"],
        "box_11b_employer_or_school": insured["employer_or_school"],
        "box_11c_insurance_plan_name": insured["plan_name"],
        "box_11d_another_health_plan": insured["another_benefit_plan"],
        "box_14_current_illness_date": _format_mmddyy_blank_safe(d["current_illness"]),
        "box_15_other_date": _format_mmddyy_blank_safe(d["other_date"]),
        "box_16_unable_to_work_from": _format_mmddyy_blank_safe(d["unable_to_work_from"]),
        "box_16_unable_to_work_to": _format_mmddyy_blank_safe(d["unable_to_work_to"]),
        "box_17_referring_provider_name": ref["name"],
        "box_17a_qualifier": ref["qualifier"],
        "box_17a_other_id": ref["other_id"],
        "box_17b_npi": ref["npi"],
        "box_18_hospitalization_from": _format_mmddyy_blank_safe(d["hospitalization_from"]),
        "box_18_hospitalization_to": _format_mmddyy_blank_safe(d["hospitalization_to"]),
        "box_20_outside_lab": record["outside_lab"],
        "box_21_diagnoses": list(record["diagnoses"]),
        "box_21_icd_indicator": "0",
        "box_22_resubmission": record["resubmission"],
        "box_23_prior_authorization": record["prior_authorization"],
        "box_24_service_lines": [
            {
                "line_number": i + 1,
                "date_from": _format_dob(line["date_from"]),
                "date_to": _format_dob(line["date_to"]),
                "place_of_service": line["place_of_service"],
                "emg": line["emg"],
                "procedure_code": line["cpt"],
                "modifiers": [m for m in line["modifiers"] if m],
                "diagnosis_pointers": line["diagnosis_pointers"],
                "charges": line["charges"],
                "units": line["units"],
                "epsdt": line["epsdt"],
                "id_qualifier": line["id_qualifier"],
                "rendering_provider_other_id": line["rendering_provider_other_id"],
                "rendering_provider_npi": line["rendering_provider_npi"],
            }
            for i, line in enumerate(record["service_lines"])
        ],
        "box_25_federal_tax_id": bill["tax_id"],
        "box_25_tax_id_type": bill["tax_id_type"],
        "box_26_patient_account_number": bill["patient_account"],
        "box_27_accept_assignment": bill["accept_assignment"],
        "box_28_total_charge": bill["total_charge"],
        "box_29_amount_paid": bill["amount_paid"],
        "box_31_physician_signature": bill["physician_signature"],
        "box_31_physician_signature_date": bill["physician_signature_date"],
        "box_32_service_facility": {
            "name": fac["name"],
            "address": fac["address"],
            "npi": fac["npi"],
            "other_id": fac["other_id"],
        },
        "box_33_billing_provider": {
            "name": bo["name"],
            "address": bo["address"],
            "phone": f"({bo['phone']['area']}) {bo['phone']['number']}",
            "npi": bo["npi"],
            "other_id": bo["other_id"],
        },
        "insurance_carrier": {
            "name": ins["carrier_name"],
            "address_line1": ins["address_line1"],
            "address_line2": ins["address_line2"],
            "city_state_zip": ins["city_state_zip"],
        },
    }


def build_ground_truth(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build the per-sample ground-truth JSON object."""
    return {
        "schema_version": "1.0",
        "meta": record["meta"],
        "fields": record_to_fields(record),
        "logical": _logical_view(record),
    }
