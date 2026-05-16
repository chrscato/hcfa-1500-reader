"""Tests for the PDF→PNG renderer."""

from PIL import Image

from hcfa_synth.pdf_fill import fill_pdf
from hcfa_synth.records import build_record
from hcfa_synth.render import pdf_to_png


def test_render_at_default_dpi_produces_expected_size():
    rec = build_record(seed=1)
    pdf_bytes = fill_pdf(rec)
    img = pdf_to_png(pdf_bytes, dpi=300)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    # CMS-1500 is 8.5" × 11" → at 300 DPI ≈ 2550 × 3300 (±2 from rounding)
    assert abs(img.size[0] - 2550) <= 2
    assert abs(img.size[1] - 3300) <= 2


def test_render_at_lower_dpi():
    rec = build_record(seed=1)
    pdf_bytes = fill_pdf(rec)
    img = pdf_to_png(pdf_bytes, dpi=150)
    assert abs(img.size[0] - 1275) <= 2
    assert abs(img.size[1] - 1650) <= 2


def test_render_is_not_blank():
    """If form fields were filled, the rendered image should have notable
    pixel variance — not just a solid color."""
    rec = build_record(seed=1)
    pdf_bytes = fill_pdf(rec)
    img = pdf_to_png(pdf_bytes, dpi=150)
    extrema = img.convert("L").getextrema()
    # min should be near black (form lines), max should be near white
    assert extrema[0] < 50 and extrema[1] > 200
