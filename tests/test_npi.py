"""Tests for NPI generation and Luhn validation."""

import random

from hcfa_synth.npi import generate_npi, luhn_valid


# Known-good test vectors from the CMS NPI check-digit spec document.
# https://www.cms.gov/Regulations-and-Guidance/Administrative-Simplification/NationalProvIdentStand/Downloads/NPIcheckdigit.pdf
KNOWN_VALID_NPIS = [
    "1234567893",  # Example from CMS spec
]
KNOWN_INVALID_NPIS = [
    "1234567890",
    "0000000000",
    "1111111111",
]


def test_known_valid_npi_passes():
    for npi in KNOWN_VALID_NPIS:
        assert luhn_valid(npi), f"{npi} should pass Luhn"


def test_known_invalid_npi_fails():
    for npi in KNOWN_INVALID_NPIS:
        assert not luhn_valid(npi), f"{npi} should fail Luhn"


def test_generated_individual_npis_are_valid():
    rng = random.Random(42)
    for _ in range(200):
        npi = generate_npi(rng)
        assert len(npi) == 10
        assert npi[0] == "1"
        assert luhn_valid(npi), f"generated NPI {npi} failed Luhn"


def test_generated_organization_npis_are_valid():
    rng = random.Random(42)
    for _ in range(200):
        npi = generate_npi(rng, organization=True)
        assert len(npi) == 10
        assert npi[0] == "2"
        assert luhn_valid(npi), f"generated org NPI {npi} failed Luhn"


def test_generation_is_deterministic_for_same_seed():
    a = [generate_npi(random.Random(7)) for _ in range(10)]
    b = [generate_npi(random.Random(7)) for _ in range(10)]
    assert a == b


def test_rejects_non_numeric_or_wrong_length():
    assert not luhn_valid("12345")
    assert not luhn_valid("12345678901")
    assert not luhn_valid("abcdefghij")
    assert not luhn_valid("")
