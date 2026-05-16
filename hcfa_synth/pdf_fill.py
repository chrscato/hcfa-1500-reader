"""Fill the official CMS-1500 AcroForm PDF from a record dict.

The PDF has 252 named fields. This module owns the mapping from the
nested record (records.build_record) to the flat {field_name: value}
dict that pypdf consumes.

Checkbox/radio export values were discovered by inspecting /AP /N entries
on each /Btn field. See _RADIO_VALUES below.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict

import pypdf
from pypdf.generic import BooleanObject, NameObject

DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "form-cms1500.pdf"

# Logical → PDF radio export value
_RELATIONSHIP_RADIO = {"self": "S", "spouse": "M", "child": "C", "other": "O"}
_INSURANCE_TYPE_RADIO = {
    "Medicare": "Medicare",
    "Medicaid": "Medicaid",
    "TRICARE": "Tricare",
    "CHAMPVA": "Champva",
    "GroupHealthPlan": "Group",
    "FECABlkLung": "Feca",
    "Other": "Other",
}
_INSURED_SEX_RADIO = {"M": "MALE", "F": "FEMALE"}


def _fmt_name(name_parts: Dict[str, str]) -> str:
    """LAST, FIRST MI — the CMS-1500 convention for Box 2 and Box 4."""
    return f"{name_parts['last']}, {name_parts['first']} {name_parts['middle']}"


def _fmt_phone(phone: Dict[str, str]) -> str:
    return phone["number"]


def record_to_fields(record: Dict[str, Any]) -> Dict[str, str]:
    """Flatten a record into a {pdf_field_name: value} dict.

    Returns string values only. Empty strings clear a field; missing keys
    leave whatever default was in the template.
    """
    f: Dict[str, str] = {}

    # --- Insurance carrier (top right block) ---
    ins = record["insurance"]
    f["insurance_name"] = ins["carrier_name"]
    f["insurance_address"] = ins["address_line1"]
    f["insurance_address2"] = ins["address_line2"]
    f["insurance_city_state_zip"] = ins["city_state_zip"]

    # --- Box 1 / 1a ---
    f["insurance_type"] = _INSURANCE_TYPE_RADIO.get(ins["type"], "Other")
    f["insurance_id"] = ins["insured_id"]

    # --- Box 2 patient name ---
    pt = record["patient"]
    f["pt_name"] = _fmt_name(pt["name"])

    # --- Box 3 DOB + sex ---
    f["birth_mm"] = pt["dob"]["mm"]
    f["birth_dd"] = pt["dob"]["dd"]
    f["birth_yy"] = pt["dob"]["yy"]
    f["sex"] = pt["sex"]

    # --- Box 4 insured's name ---
    insured = record["insured"]
    f["ins_name"] = _fmt_name(insured["name"])

    # --- Box 5 patient address/phone ---
    pt_addr = pt["address"]
    f["pt_street"] = pt_addr["street"]
    f["pt_city"] = pt_addr["city"]
    f["pt_state"] = pt_addr["state"]
    f["pt_zip"] = pt_addr["zip"]
    f["pt_AreaCode"] = pt["phone"]["area"]
    f["pt_phone"] = _fmt_phone(pt["phone"])

    # --- Box 6 relationship ---
    f["rel_to_ins"] = _RELATIONSHIP_RADIO.get(pt["relationship_to_insured"], "O")

    # --- Box 7 insured address/phone ---
    ins_addr = insured["address"]
    f["ins_street"] = ins_addr["street"]
    f["ins_city"] = ins_addr["city"]
    f["ins_state"] = ins_addr["state"]
    f["ins_zip"] = ins_addr["zip"]
    f["ins_phone area"] = insured["phone"]["area"]
    f["ins_phone"] = _fmt_phone(insured["phone"])

    # --- Box 9 / 9a / 9d other insured ---
    other = record["other_insured"]
    if other:
        f["other_ins_name"] = other["name"]
        f["other_ins_policy"] = other["policy_or_group"]
        f["other_ins_plan_name"] = other["plan_name"]

    # --- Box 10 condition ---
    cond = record["condition"]
    f["employment"] = cond["employment_related"]
    f["pt_auto_accident"] = cond["auto_accident"]
    f["accident_place"] = cond["auto_accident_state"]
    f["other_accident"] = cond["other_accident"]

    # --- Box 11 insured policy info ---
    f["ins_policy"] = insured["policy_group"]
    f["ins_dob_mm"] = insured["dob"]["mm"]
    f["ins_dob_dd"] = insured["dob"]["dd"]
    f["ins_dob_yy"] = insured["dob"]["yy"]
    f["ins_sex"] = _INSURED_SEX_RADIO.get(insured["sex"], "MALE")
    f["ins_benefit_plan"] = insured["another_benefit_plan"]
    f["ins_plan_name"] = insured["plan_name"]

    # --- Box 12 / 13 signatures ---
    f["pt_signature"] = pt["signature"]
    f["pt_date"] = pt["signature_date"]
    f["ins_signature"] = insured["signature"]

    # --- Box 14 / 15 / 16 / 18 dates ---
    d = record["dates"]
    f["cur_ill_mm"] = d["current_illness"]["mm"]
    f["cur_ill_dd"] = d["current_illness"]["dd"]
    f["cur_ill_yy"] = d["current_illness"]["yy"]
    f["sim_ill_mm"] = d["other_date"]["mm"]
    f["sim_ill_dd"] = d["other_date"]["dd"]
    f["sim_ill_yy"] = d["other_date"]["yy"]
    f["work_mm_from"] = d["unable_to_work_from"]["mm"]
    f["work_dd_from"] = d["unable_to_work_from"]["dd"]
    f["work_yy_from"] = d["unable_to_work_from"]["yy"]
    f["work_mm_end"] = d["unable_to_work_to"]["mm"]
    f["work_dd_end"] = d["unable_to_work_to"]["dd"]
    f["work_yy_end"] = d["unable_to_work_to"]["yy"]
    f["hosp_mm_from"] = d["hospitalization_from"]["mm"]
    f["hosp_dd_from"] = d["hospitalization_from"]["dd"]
    f["hosp_yy_from"] = d["hospitalization_from"]["yy"]
    f["hosp_mm_end"] = d["hospitalization_to"]["mm"]
    f["hosp_dd_end"] = d["hospitalization_to"]["dd"]
    f["hosp_yy_end"] = d["hospitalization_to"]["yy"]

    # --- Box 17 referring provider ---
    ref = record["referring_provider"]
    f["ref_physician"] = ref["name"]
    f["physician number 17a1"] = ref["qualifier"]
    f["physician number 17a"] = ref["other_id"]
    f["id_physician"] = ref["npi"]

    # --- Box 20 outside lab ---
    lab = record["outside_lab"]
    f["lab"] = "YES" if lab["yes"] else "NO"
    f["charge"] = lab["charges"]

    # --- Box 21 diagnoses + ICD indicator ---
    f["99icd"] = "0"  # 0 = ICD-10-CM
    for i, code in enumerate(record["diagnoses"], start=1):
        if i > 12:
            break
        f[f"diagnosis{i}"] = code

    # --- Box 22 / 23 ---
    resub = record["resubmission"]
    f["medicaid_resub"] = resub["code"]
    f["original_ref"] = resub["original_ref_number"]
    f["prior_auth"] = record["prior_authorization"]

    # --- Box 24 service lines ---
    for i, line in enumerate(record["service_lines"], start=1):
        if i > 6:
            break
        f[f"sv{i}_mm_from"] = line["date_from"]["mm"]
        f[f"sv{i}_dd_from"] = line["date_from"]["dd"]
        f[f"sv{i}_yy_from"] = line["date_from"]["yy"]
        f[f"sv{i}_mm_end"] = line["date_to"]["mm"]
        f[f"sv{i}_dd_end"] = line["date_to"]["dd"]
        f[f"sv{i}_yy_end"] = line["date_to"]["yy"]
        f[f"place{i}"] = line["place_of_service"]
        f[f"emg{i}"] = line["emg"]
        f[f"cpt{i}"] = line["cpt"]
        f[f"mod{i}"] = line["modifiers"][0]
        f[f"mod{i}a"] = line["modifiers"][1]
        f[f"mod{i}b"] = line["modifiers"][2]
        f[f"mod{i}c"] = line["modifiers"][3]
        f[f"diag{i}"] = line["diagnosis_pointers"]
        f[f"ch{i}"] = line["charges"]
        f[f"day{i}"] = line["units"]
        f[f"epsdt{i}"] = line["epsdt"]
        f[f"type{i}"] = line["id_qualifier"]
        f[f"local{i}a"] = line["rendering_provider_other_id"]
        f[f"local{i}"] = line["rendering_provider_npi"]

    # --- Box 25 / 26 / 27 / 28 / 29 ---
    bill = record["billing"]
    f["tax_id"] = bill["tax_id"]
    f["ssn"] = bill["tax_id_type"]
    f["pt_account"] = bill["patient_account"]
    f["assignment"] = bill["accept_assignment"]
    f["t_charge"] = bill["total_charge"]
    f["amt_paid"] = bill["amount_paid"]

    # --- Box 31 physician signature ---
    f["physician_signature"] = bill["physician_signature"]
    f["physician_date"] = bill["physician_signature_date"]

    # --- Box 32 service facility ---
    fac = record["service_facility"]
    fac_addr = fac["address"]
    f["fac_name"] = fac["name"]
    f["fac_street"] = fac_addr["street"]
    f["fac_location"] = f"{fac_addr['city']}, {fac_addr['state']} {fac_addr['zip']}"

    # --- Box 33 billing provider ---
    bo = record["billing_office"]
    bo_addr = bo["address"]
    f["doc_name"] = bo["name"]
    f["doc_street"] = bo_addr["street"]
    f["doc_location"] = f"{bo_addr['city']}, {bo_addr['state']} {bo_addr['zip']}"
    f["doc_phone area"] = bo["phone"]["area"]
    f["doc_phone"] = _fmt_phone(bo["phone"])
    f["pin"] = bo["npi"]
    f["grp"] = bo["other_id"]

    return f


def fill_pdf(record: Dict[str, Any], template_path: Path | str = DEFAULT_TEMPLATE) -> bytes:
    """Fill the CMS-1500 template with a record and return the PDF bytes."""
    template_path = Path(template_path)
    reader = pypdf.PdfReader(str(template_path))
    writer = pypdf.PdfWriter(clone_from=reader)

    field_values = record_to_fields(record)

    # Page 2 of the CMS-1500 has no fields; only update pages that do.
    for page in writer.pages:
        annots = page.get("/Annots")
        if annots:
            writer.update_page_form_field_values(page, field_values)

    # /NeedAppearances tells viewers/renderers to (re)generate field
    # appearance streams from the values we just set.
    if "/AcroForm" not in writer._root_object:
        writer._root_object[NameObject("/AcroForm")] = writer._root_object.get("/AcroForm")
    acroform = writer._root_object["/AcroForm"]
    acroform[NameObject("/NeedAppearances")] = BooleanObject(True)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
