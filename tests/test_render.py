"""Tests for the PDF→PNG renderer."""

import io

import pypdf
from pypdf.generic import NameObject
from PIL import Image, ImageChops

from hcfa_synth.pdf_fill import fill_pdf
from hcfa_synth.records import build_record
from hcfa_synth.render import pdf_to_png

_PAGE_HEIGHT_PT = 792  # CMS-1500 mediabox is 612 × 792 pt (8.5" × 11")


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


def _selected_widget_rect(pdf_bytes, field_name):
    """Return the /Rect (PDF points) of the checked widget of a /Btn field.

    The checked widget is the kid whose /AS is not /Off.
    """
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    for ref in reader.trailer["/Root"]["/AcroForm"]["/Fields"]:
        field = ref.get_object()
        if field.get("/T") != field_name:
            continue
        for kid in field.get("/Kids", []):
            widget = kid.get_object()
            if widget.get("/AS") not in (None, "/Off"):
                return [float(x) for x in widget["/Rect"]]
    raise AssertionError(f"no checked widget found for field {field_name!r}")


def _all_off_variant(pdf_bytes):
    """Clone a filled PDF but force every button widget's /AS to /Off.

    Renders identically to the filled form *except* for the checkbox/radio
    marks, so a pixel diff isolates exactly those marks.
    """
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    writer = pypdf.PdfWriter(clone_from=reader)
    for ref in writer._root_object["/AcroForm"]["/Fields"]:
        field = ref.get_object()
        if field.get("/FT") != "/Btn":
            continue
        for kid in field.get("/Kids", [field]):
            kid.get_object()[NameObject("/AS")] = NameObject("/Off")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _changed_pixels(on_img, off_img, rect_pt, dpi):
    """Count pixels that differ between on/off renders inside a widget rect."""
    scale = dpi / 72.0
    x0, y0, x1, y1 = rect_pt
    margin = 4  # the X mark can stroke a hair past the widget border
    box = (
        int(x0 * scale) - margin,
        int((_PAGE_HEIGHT_PT - y1) * scale) - margin,  # PDF y-up → image y-down
        int(x1 * scale) + margin,
        int((_PAGE_HEIGHT_PT - y0) * scale) + margin,
    )
    diff = ImageChops.difference(on_img.crop(box), off_img.crop(box))
    return sum(1 for p in diff.getdata() if p > 20)


def test_checkbox_marks_render_visibly():
    """Regression for the checkbox/radio "blank mark" bug: setting /V alone
    left every widget's /AS at /Off, so pypdfium2 rasterized nothing. After
    syncing /AS to the on-state, the sex and accident marks must appear.

    We diff the filled render against an all-off render of the same form;
    any changed pixels in a widget rect are the checkbox/radio mark itself.
    """
    rec = build_record(seed=42)
    on_pdf = fill_pdf(rec)
    off_pdf = _all_off_variant(on_pdf)

    dpi = 300
    on_img = pdf_to_png(on_pdf, dpi=dpi).convert("L")
    off_img = pdf_to_png(off_pdf, dpi=dpi).convert("L")

    for field_name in ("sex", "pt_auto_accident"):
        rect = _selected_widget_rect(on_pdf, field_name)
        changed = _changed_pixels(on_img, off_img, rect, dpi)
        assert changed > 50, f"{field_name} mark did not render (changed={changed})"
