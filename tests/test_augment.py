"""Tests for the augmentation tier transforms."""

import pytest
from PIL import Image

from hcfa_synth.augment import TIER_NAMES, apply_tier
from hcfa_synth.pdf_fill import fill_pdf
from hcfa_synth.records import build_record
from hcfa_synth.render import pdf_to_png


@pytest.fixture(scope="module")
def base_image() -> Image.Image:
    rec = build_record(seed=0)
    pdf_bytes = fill_pdf(rec)
    # Lower DPI for test speed; augmentations work at any resolution.
    return pdf_to_png(pdf_bytes, dpi=100)


@pytest.mark.parametrize("tier", TIER_NAMES)
def test_each_tier_returns_rgb_image_same_size(base_image, tier):
    out = apply_tier(base_image, tier, seed=1)
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"
    assert out.size == base_image.size


@pytest.mark.parametrize("tier", TIER_NAMES)
def test_tier_is_deterministic(base_image, tier):
    a = apply_tier(base_image, tier, seed=7)
    b = apply_tier(base_image, tier, seed=7)
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("tier", [t for t in TIER_NAMES if t != "pristine"])
def test_non_pristine_tiers_change_pixels(base_image, tier):
    out = apply_tier(base_image, tier, seed=1)
    assert out.tobytes() != base_image.tobytes()


def test_pristine_is_passthrough(base_image):
    out = apply_tier(base_image, "pristine", seed=99)
    assert out.tobytes() == base_image.tobytes()


def test_unknown_tier_raises():
    img = Image.new("RGB", (100, 100), "white")
    with pytest.raises(ValueError, match="unknown tier"):
        apply_tier(img, "not-a-tier", seed=1)
