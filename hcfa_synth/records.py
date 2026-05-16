"""Synthetic CMS-1500 record builder.

`build_record(seed)` returns a nested dict modeling one filled claim form.
This is the single source of truth — both PDF filling and ground-truth JSON
emission consume this object, so the two cannot drift apart.

Structure mirrors the 33 logical boxes of the CMS-1500, not the PDF field
names (which are messy). Mapping happens downstream in pdf_fill.py and
ground_truth.py.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from faker import Faker

from hcfa_synth.codes import load_hcpcs_codes, load_icd10_codes, load_pos_codes
from hcfa_synth.npi import generate_npi

INSURANCE_TYPES = [
    "Medicare",
    "Medicaid",
    "TRICARE",
    "CHAMPVA",
    "GroupHealthPlan",
    "FECABlkLung",
    "Other",
]

RELATIONSHIP_TO_INSURED = ["self", "spouse", "child", "other"]


def _mmddyy(d: date) -> Dict[str, str]:
    return {"mm": f"{d.month:02d}", "dd": f"{d.day:02d}", "yy": f"{d.year % 100:02d}"}


def _mmddyyyy(d: date) -> str:
    return f"{d.month:02d}/{d.day:02d}/{d.year:04d}"


def _money(amount: float) -> str:
    return f"{amount:.2f}"


def _person_name(fake: Faker, sex: str) -> Dict[str, str]:
    first = fake.first_name_male() if sex == "M" else fake.first_name_female()
    return {
        "first": first.upper(),
        "middle": fake.random_uppercase_letter(),
        "last": fake.last_name().upper(),
    }


def _address(fake: Faker) -> Dict[str, str]:
    return {
        "street": fake.street_address().upper(),
        "city": fake.city().upper(),
        "state": fake.state_abbr(),
        "zip": fake.zipcode(),
    }


def _phone(fake: Faker) -> Dict[str, str]:
    area = f"{fake.random_int(200, 999)}"
    number = f"{fake.random_int(200, 999)}-{fake.random_int(1000, 9999)}"
    return {"area": area, "number": number}


def _service_line(
    fake: Faker,
    rng: random.Random,
    service_date: date,
    rendering_npi: str,
    diagnosis_count: int,
) -> Dict[str, Any]:
    hcpcs = load_hcpcs_codes()
    pos = load_pos_codes()
    code, _desc = rng.choice(hcpcs)
    pos_code, _ = rng.choice(pos)

    # Optional modifiers — most lines have 0-1, occasionally up to 4.
    modifier_pool = ["25", "26", "50", "51", "52", "59", "76", "77", "LT", "RT", "TC", "GA"]
    n_mods = rng.choices([0, 1, 2], weights=[60, 30, 10])[0]
    mods = rng.sample(modifier_pool, n_mods) + [""] * (4 - n_mods)

    # Diagnosis pointer — references positions A..L of the diagnosis list.
    available_pointers = "ABCDEFGHIJKL"[:diagnosis_count]
    n_pointers = rng.choices([1, 2, 3, 4], weights=[60, 25, 10, 5])[0]
    n_pointers = min(n_pointers, len(available_pointers))
    pointers = "".join(rng.sample(available_pointers, n_pointers))

    units = rng.choices([1, 1, 1, 2, 3, 4, 5], weights=[50, 20, 10, 8, 6, 4, 2])[0]
    unit_charge = round(rng.uniform(35.0, 450.0), 2)
    charges = round(unit_charge * units, 2)

    return {
        "date_from": _mmddyy(service_date),
        "date_to": _mmddyy(service_date),
        "place_of_service": pos_code,
        "emg": "",
        "cpt": code,
        "modifiers": mods,
        "diagnosis_pointers": pointers,
        "charges": _money(charges),
        "units": str(units),
        "epsdt": "",
        "id_qualifier": "",
        "rendering_provider_other_id": "",
        "rendering_provider_npi": rendering_npi,
    }


def build_record(seed: Optional[int] = None) -> Dict[str, Any]:
    """Construct one synthetic CMS-1500 record.

    Cross-field guarantees:
      * DOB < service date <= today
      * Box 28 total charges == sum of line charges
      * Diagnosis pointers only reference populated diagnoses
      * Tax ID format matches selected type (SSN xxx-xx-xxxx, EIN xx-xxxxxxx)
      * If patient is insured (self), insured demographics mirror patient
    """
    rng = random.Random(seed)
    fake = Faker("en_US")
    if seed is not None:
        Faker.seed(seed)

    # --- Patient demographics ---
    patient_sex = rng.choice(["M", "F"])
    today = date.today()
    patient_dob = today - timedelta(days=rng.randint(365 * 1, 365 * 90))
    patient_name = _person_name(fake, patient_sex)
    patient_address = _address(fake)
    patient_phone = _phone(fake)
    relationship = rng.choices(RELATIONSHIP_TO_INSURED, weights=[60, 25, 12, 3])[0]

    # --- Insured (Box 4 / 7 / 11) ---
    if relationship == "self":
        insured_name = dict(patient_name)
        insured_address = dict(patient_address)
        insured_phone = dict(patient_phone)
        insured_dob = patient_dob
        insured_sex = patient_sex
    else:
        insured_sex = rng.choice(["M", "F"])
        insured_dob = today - timedelta(days=rng.randint(365 * 18, 365 * 75))
        insured_name = _person_name(fake, insured_sex)
        insured_address = _address(fake)
        insured_phone = _phone(fake)

    # --- Insurance carrier ---
    carriers = [
        "BLUE CROSS BLUE SHIELD", "AETNA", "CIGNA", "UNITED HEALTHCARE",
        "HUMANA", "ANTHEM", "MEDICARE", "MEDICAID", "TRICARE",
        "KAISER PERMANENTE", "MOLINA HEALTHCARE", "CENTENE",
    ]
    insurance_carrier = rng.choice(carriers)
    insurance = {
        "carrier_name": insurance_carrier,
        "address_line1": fake.street_address().upper(),
        "address_line2": rng.choice(["", f"STE {rng.randint(100, 999)}", f"PO BOX {rng.randint(1000, 99999)}"]),
        "city_state_zip": f"{fake.city().upper()}, {fake.state_abbr()} {fake.zipcode()}",
        "type": rng.choice(INSURANCE_TYPES),
        "insured_id": f"{fake.random_uppercase_letter()}{fake.random_uppercase_letter()}{rng.randint(100000000, 999999999)}",
    }

    # --- Other insured (Box 9) — populated ~30% of the time ---
    has_other_insured = rng.random() < 0.30
    other_insured = None
    if has_other_insured:
        other_name = _person_name(fake, rng.choice(["M", "F"]))
        other_insured = {
            "name": f"{other_name['last']}, {other_name['first']} {other_name['middle']}",
            "policy_or_group": f"{rng.randint(10000, 99999)}",
            "plan_name": rng.choice(carriers),
        }

    # --- Condition (Box 10) ---
    is_employment = rng.random() < 0.10
    is_auto = rng.random() < 0.08
    is_other_accident = rng.random() < 0.05
    condition = {
        "employment_related": "YES" if is_employment else "NO",
        "auto_accident": "YES" if is_auto else "NO",
        "auto_accident_state": fake.state_abbr() if is_auto else "",
        "other_accident": "YES" if is_other_accident else "NO",
    }

    # --- Dates (Boxes 14, 15, 16, 18) ---
    # Service date is the anchor; everything else is offset from it.
    service_date = today - timedelta(days=rng.randint(1, 180))
    current_illness = service_date - timedelta(days=rng.randint(0, 14))
    other_date_present = rng.random() < 0.20
    work_dates_present = is_employment or rng.random() < 0.05
    hosp_dates_present = rng.random() < 0.10

    def _blank_date() -> Dict[str, str]:
        return {"mm": "", "dd": "", "yy": ""}

    dates = {
        "current_illness": _mmddyy(current_illness),
        "other_date": _mmddyy(service_date - timedelta(days=rng.randint(7, 60))) if other_date_present else _blank_date(),
        "unable_to_work_from": _mmddyy(service_date - timedelta(days=rng.randint(1, 30))) if work_dates_present else _blank_date(),
        "unable_to_work_to": _mmddyy(service_date + timedelta(days=rng.randint(7, 60))) if work_dates_present else _blank_date(),
        "hospitalization_from": _mmddyy(service_date - timedelta(days=rng.randint(0, 5))) if hosp_dates_present else _blank_date(),
        "hospitalization_to": _mmddyy(service_date + timedelta(days=rng.randint(0, 7))) if hosp_dates_present else _blank_date(),
    }

    # --- Providers (Box 17, 31, 32, 33) ---
    rendering_npi = generate_npi(rng)
    referring_provider_present = rng.random() < 0.40
    referring_npi = generate_npi(rng) if referring_provider_present else ""
    referring_name = ""
    if referring_provider_present:
        ref = _person_name(fake, rng.choice(["M", "F"]))
        referring_name = f"DR. {ref['first']} {ref['last']}"

    facility_npi = generate_npi(rng, organization=True)
    billing_npi = generate_npi(rng, organization=True)

    # --- Diagnoses (Box 21) — 1 to 12 ICD-10s ---
    icd_pool = load_icd10_codes()
    n_diagnoses = rng.choices(range(1, 13), weights=[5, 25, 25, 15, 10, 8, 5, 3, 2, 1, 0.5, 0.5])[0]
    diagnoses = [code for code, _ in rng.sample(icd_pool, n_diagnoses)]

    # --- Service lines (Box 24) — 1 to 6 lines ---
    n_lines = rng.choices(range(1, 7), weights=[20, 30, 25, 15, 7, 3])[0]
    service_lines: List[Dict[str, Any]] = []
    for i in range(n_lines):
        sd = service_date - timedelta(days=rng.randint(0, 3))
        service_lines.append(_service_line(fake, rng, sd, rendering_npi, n_diagnoses))

    total_charge = sum(float(line["charges"]) for line in service_lines)
    amount_paid = round(total_charge * rng.choice([0.0, 0.0, 0.0, 0.2, 0.5]), 2)

    # --- Billing provider footer (Box 25-33) ---
    tax_id_type = rng.choice(["SSN", "EIN"])
    if tax_id_type == "SSN":
        tax_id = f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"
    else:
        tax_id = f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"

    physician_name_parts = _person_name(fake, rng.choice(["M", "F"]))
    physician_signature = f"{physician_name_parts['first']} {physician_name_parts['last']} MD"

    facility_address = _address(fake)
    billing_address = _address(fake)
    billing_phone = _phone(fake)

    record: Dict[str, Any] = {
        "meta": {
            "seed": seed,
            "generated_on": today.isoformat(),
        },
        "insurance": insurance,
        "patient": {
            "name": patient_name,
            "dob": _mmddyy(patient_dob),
            "sex": patient_sex,
            "address": patient_address,
            "phone": patient_phone,
            "relationship_to_insured": relationship,
            "account_number": f"ACC-{rng.randint(100000, 999999)}",
            "signature": "Signature on File",
            "signature_date": _mmddyyyy(service_date),
        },
        "insured": {
            "name": insured_name,
            "address": insured_address,
            "phone": insured_phone,
            "dob": _mmddyy(insured_dob),
            "sex": insured_sex,
            "policy_group": f"GRP{rng.randint(10000, 99999)}",
            "employer_or_school": fake.company().upper(),
            "plan_name": insurance_carrier,
            "another_benefit_plan": "YES" if has_other_insured else "NO",
            "signature": "Signature on File",
        },
        "other_insured": other_insured,
        "condition": condition,
        "dates": dates,
        "referring_provider": {
            "name": referring_name,
            "qualifier": "DN" if referring_provider_present else "",
            "other_id": "",
            "npi": referring_npi,
        },
        "outside_lab": {
            "yes": False,
            "charges": "",
        },
        "diagnoses": diagnoses,
        "resubmission": {
            "code": "",
            "original_ref_number": "",
        },
        "prior_authorization": "",
        "service_lines": service_lines,
        "billing": {
            "tax_id": tax_id,
            "tax_id_type": tax_id_type,
            "patient_account": f"ACC-{rng.randint(100000, 999999)}",
            "accept_assignment": rng.choices(["YES", "NO"], weights=[85, 15])[0],
            "total_charge": _money(total_charge),
            "amount_paid": _money(amount_paid),
            "physician_signature": physician_signature,
            "physician_signature_date": _mmddyyyy(service_date + timedelta(days=rng.randint(0, 7))),
        },
        "service_facility": {
            "name": rng.choice([
                "SPRINGFIELD MEDICAL CENTER", "CITY GENERAL HOSPITAL",
                "RIVERSIDE CLINIC", "OAK STREET FAMILY PRACTICE",
                "VALLEY HEALTH GROUP", "METRO URGENT CARE",
            ]),
            "address": facility_address,
            "npi": facility_npi,
            "other_id": "",
        },
        "billing_office": {
            "name": physician_signature,
            "address": billing_address,
            "phone": billing_phone,
            "npi": billing_npi,
            "other_id": "",
        },
    }

    return record
