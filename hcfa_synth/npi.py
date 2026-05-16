"""NPI (National Provider Identifier) generation and validation.

NPIs are 10-digit numbers issued by CMS. Validation uses the ISO/IEC 7812
Luhn algorithm on the NPI prepended with the healthcare issuer prefix '80840'.

NPIs currently begin with 1 (individual provider) or 2 (organization).
References: CMS NPPES, https://www.cms.gov/Regulations-and-Guidance/Administrative-Simplification/NationalProvIdentStand/Downloads/NPIcheckdigit.pdf
"""

from __future__ import annotations

import random

NPI_PREFIX = "80840"
INDIVIDUAL_FIRST_DIGITS = ("1",)
ORGANIZATION_FIRST_DIGITS = ("2",)


def _luhn_check_digit(digits: str) -> int:
    """Return the Luhn check digit for the given numeric string.

    Doubles every second digit from the right; digits >= 10 are reduced by
    summing their decimal digits (equivalent to subtracting 9). The check
    digit makes the resulting total a multiple of 10.
    """
    total = 0
    # Walk left-to-right; the rightmost position of `digits` is position
    # (len-1). The check digit will be appended at position len, so we
    # double digits whose distance-from-the-end (with check appended) is odd.
    # That's equivalent to doubling at indices where (len - i) is even.
    for i, ch in enumerate(digits):
        d = int(ch)
        if (len(digits) - i) % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def luhn_valid(npi: str) -> bool:
    """Check whether a 10-digit NPI passes the Luhn check with the 80840 prefix."""
    if len(npi) != 10 or not npi.isdigit():
        return False
    body, check = npi[:-1], int(npi[-1])
    return _luhn_check_digit(NPI_PREFIX + body) == check


def generate_npi(rng: random.Random | None = None, *, organization: bool = False) -> str:
    """Generate a 10-digit NPI that passes the Luhn check.

    Args:
        rng: optional Random instance for deterministic output.
        organization: if True, emits a Type 2 (organizational) NPI starting with 2.
    """
    rng = rng or random.Random()
    first = rng.choice(ORGANIZATION_FIRST_DIGITS if organization else INDIVIDUAL_FIRST_DIGITS)
    body = first + "".join(str(rng.randint(0, 9)) for _ in range(8))
    check = _luhn_check_digit(NPI_PREFIX + body)
    return body + str(check)
