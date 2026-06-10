"""Fill the official CMS-1500 AcroForm PDF from a record dict.

The PDF has 252 named fields. This module owns the mapping from the
nested record (records.build_record) to the flat {field_name: value}
dict that pypdf consumes.

Checkbox/radio export values were discovered by inspecting /AP /N entries
on each /Btn field. See _RADIO_VALUES below.
"""

from __future__ import annotations

import io
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pypdf
from pypdf.generic import ArrayObject, BooleanObject, FloatObject, NameObject

DEFAULT_TEMPLATE = Path(__file__).resolve().parent.parent / "form-cms1500.pdf"

_OFF_STATE = NameObject("/Off")

_TEXT_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


def _year_box(date_dict: Dict[str, str], year_digits: int) -> str:
    """Render a split year box as 2- or 4-digit. Blank stays blank.

    Writing 4 digits into the narrow YY cell intentionally overflows it — a
    common look on real claims that the v1 model never saw (it only ever read
    2-digit years). Ground truth is unaffected; it always carries the full year.
    """
    if not date_dict.get("mm"):
        return date_dict.get("yy", "")
    if year_digits == 4:
        return date_dict.get("yyyy") or date_dict.get("yy", "")
    return date_dict.get("yy", "")


def _restyle_text_date(canonical: str, rng: random.Random) -> str:
    """Vary a canonical MM/DD/YYYY free-text date: separator, 2- vs 4-digit
    year, optional leading-zero stripping. Ground truth stays canonical."""
    m = _TEXT_DATE_RE.match(canonical or "")
    if not m:
        return canonical
    mm, dd, yyyy = m.groups()
    sep = rng.choice(["/", "/", "-", "."])
    year = yyyy if rng.random() < 0.6 else yyyy[2:]
    if rng.random() < 0.30:
        mm, dd = str(int(mm)), str(int(dd))
    else:
        mm, dd = f"{int(mm):02d}", f"{int(dd):02d}"
    return f"{mm}{sep}{dd}{sep}{year}"

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


def record_to_fields(record: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, str]:
    """Flatten a record into a {pdf_field_name: value} dict.

    Returns string values only. Empty strings clear a field; missing keys
    leave whatever default was in the template.

    `rng` enables *render-time display variety* — 2- vs 4-digit years in the
    split date boxes and separator/format variety in free-text dates. This
    changes only how values look on the rasterized form, never the ground
    truth: call with `rng=None` (the default) to get the canonical values that
    ground_truth.py scores against.
    """
    f: Dict[str, str] = {}

    # Per-form display choices (only when rendering, not for ground truth).
    year_digits = 4 if (rng is not None and rng.random() < 0.35) else 2

    def _yr(date_dict: Dict[str, str]) -> str:
        return _year_box(date_dict, year_digits)

    def _txt_date(canonical: str) -> str:
        return _restyle_text_date(canonical, rng) if rng is not None else canonical

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
    f["birth_yy"] = _yr(pt["dob"])
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
    f["ins_dob_yy"] = _yr(insured["dob"])
    f["ins_sex"] = _INSURED_SEX_RADIO.get(insured["sex"], "MALE")
    f["ins_benefit_plan"] = insured["another_benefit_plan"]
    f["ins_plan_name"] = insured["plan_name"]

    # --- Box 12 / 13 signatures ---
    f["pt_signature"] = pt["signature"]
    f["pt_date"] = _txt_date(pt["signature_date"])
    f["ins_signature"] = insured["signature"]

    # --- Box 14 / 15 / 16 / 18 dates ---
    d = record["dates"]
    f["cur_ill_mm"] = d["current_illness"]["mm"]
    f["cur_ill_dd"] = d["current_illness"]["dd"]
    f["cur_ill_yy"] = _yr(d["current_illness"])
    f["sim_ill_mm"] = d["other_date"]["mm"]
    f["sim_ill_dd"] = d["other_date"]["dd"]
    f["sim_ill_yy"] = _yr(d["other_date"])
    f["work_mm_from"] = d["unable_to_work_from"]["mm"]
    f["work_dd_from"] = d["unable_to_work_from"]["dd"]
    f["work_yy_from"] = _yr(d["unable_to_work_from"])
    f["work_mm_end"] = d["unable_to_work_to"]["mm"]
    f["work_dd_end"] = d["unable_to_work_to"]["dd"]
    f["work_yy_end"] = _yr(d["unable_to_work_to"])
    f["hosp_mm_from"] = d["hospitalization_from"]["mm"]
    f["hosp_dd_from"] = d["hospitalization_from"]["dd"]
    f["hosp_yy_from"] = _yr(d["hospitalization_from"])
    f["hosp_mm_end"] = d["hospitalization_to"]["mm"]
    f["hosp_dd_end"] = d["hospitalization_to"]["dd"]
    f["hosp_yy_end"] = _yr(d["hospitalization_to"])

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
        f[f"sv{i}_yy_from"] = _yr(line["date_from"])
        f[f"sv{i}_mm_end"] = line["date_to"]["mm"]
        f[f"sv{i}_dd_end"] = line["date_to"]["dd"]
        f[f"sv{i}_yy_end"] = _yr(line["date_to"])
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
    f["physician_date"] = _txt_date(bill["physician_signature_date"])

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


def _sync_button_appearance_states(writer: pypdf.PdfWriter) -> None:
    """Point each button widget's /AS at the on-state matching the field's /V.

    pypdf sets /V on a /Btn (checkbox/radio) field but leaves every kid
    widget's /AS at /Off. pypdfium2 rasterizes the appearance named by /AS,
    so without this the checked mark is invisible even though the value is
    correct. The on-state /AP /N appearance streams already exist in the
    template — we only retarget /AS to the one matching /V (and force the
    rest to /Off).
    """
    acroform = writer._root_object.get("/AcroForm")
    if not acroform:
        return
    for ref in acroform.get("/Fields", []):
        field = ref.get_object()
        if field.get("/FT") != "/Btn":
            continue
        value = field.get("/V")
        # pypdf writes /V as a TextStringObject ("M"), but /AP /N keys are
        # PDF names ("/M"). Normalize to a name with exactly one leading slash.
        on_state = "/" + str(value).lstrip("/") if value not in (None, "") else None
        kids = field.get("/Kids")
        widgets = [k.get_object() for k in kids] if kids else [field]
        for widget in widgets:
            ap = widget.get("/AP")
            states = list(ap["/N"].keys()) if ap and "/N" in ap else []
            if on_state is not None and on_state in states:
                widget[NameObject("/AS")] = NameObject(on_state)
            else:
                widget[NameObject("/AS")] = _OFF_STATE


def _apply_field_jitter(
    writer: pypdf.PdfWriter,
    rng: random.Random,
    prob: float,
    max_shift_pt: float = 2.5,
) -> None:
    """Nudge text-field widget rectangles by a few PDF points.

    With /NeedAppearances set, the renderer redraws each value inside its
    (now shifted) /Rect, so a downward/upward nudge makes digits cross the
    printed cell gridlines and a sideways nudge pushes long values over their
    box edge. This reproduces the misaligned, overflowing handwriting/print
    seen on real scanned claims — something v1's pixel-perfect forms lacked.

    Only /Tx (text) widgets are touched; buttons/radios are left alone so the
    checkbox-appearance sync below stays valid.
    """
    for page in writer.pages:
        annots = page.get("/Annots")
        if not annots:
            continue
        for ref in annots:
            widget = ref.get_object()
            if widget.get("/Subtype") != "/Widget":
                continue
            ft = widget.get("/FT")
            if ft is None and "/Parent" in widget:
                ft = widget["/Parent"].get_object().get("/FT")
            if ft != "/Tx":
                continue
            if rng.random() > prob:
                continue
            rect = widget.get("/Rect")
            if not rect or len(rect) != 4:
                continue
            x0, y0, x1, y1 = (float(v) for v in rect)
            dx = rng.uniform(-max_shift_pt, max_shift_pt)
            dy = rng.uniform(-max_shift_pt, max_shift_pt)
            widget[NameObject("/Rect")] = ArrayObject(
                [FloatObject(x0 + dx), FloatObject(y0 + dy),
                 FloatObject(x1 + dx), FloatObject(y1 + dy)]
            )


def fill_pdf(
    record: Dict[str, Any],
    template_path: Path | str = DEFAULT_TEMPLATE,
    *,
    rng: Optional[random.Random] = None,
    jitter: float = 0.0,
) -> bytes:
    """Fill the CMS-1500 template with a record and return the PDF bytes.

    `rng` enables render-time display variety (year-digit and date-format
    variety) and seeds the optional field jitter. `jitter` is the per-field
    probability of a positional nudge (0 disables it). Both default off so
    ground-truth generation and existing callers/tests are unaffected.
    """
    template_path = Path(template_path)
    reader = pypdf.PdfReader(str(template_path))
    writer = pypdf.PdfWriter(clone_from=reader)

    field_values = record_to_fields(record, rng=rng)

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

    # Retarget button widgets' /AS so the checked marks actually rasterize.
    _sync_button_appearance_states(writer)

    # Positional jitter must run after values are set (so the widgets exist)
    # and only on text fields, leaving the button /AS sync above intact.
    if jitter > 0 and rng is not None:
        _apply_field_jitter(writer, rng, prob=jitter)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
