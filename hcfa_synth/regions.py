"""Region geometry + ground-truth splitting for the CMS-1500 form.

v2 trains two region-specialized models instead of one whole-page model:

  * "fields"  — the single-value labeled boxes, read from the TOP band
                (boxes 1-23: carrier, patient/insured demographics, dates,
                diagnoses, referring provider) and the BOTTOM band
                (boxes 25-33: tax id, account, charges, physician, facility,
                billing provider).
  * "service" — the box 24 service-line table (variable number of rows),
                read from the SERVICE band.

Cropping a band gives the model far more effective resolution on small glyphs
(NPIs, codes, 10-digit ids) than a downsampled full page, and shrinks each
model's output schema — both of which directly target v1's weak spots
(small-number errors and JSON parse failures).

The band cuts are derived *from the template's own field rectangles* (the
service rows define where the middle band starts/ends), so they stay correct
if the template is ever re-fielded. Pixel coordinates depend on render DPI —
pass the same dpi used to rasterize the page.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pypdf
from PIL import Image

from hcfa_synth.pdf_fill import DEFAULT_TEMPLATE

_BASE_DPI = 72.0

PxBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1), y-down image pixels
PtRect = Tuple[float, float, float, float]  # (x0, y0, x1, y1), y-up PDF points

# Field names belonging to the box-24 service-line table. Anchored so that
# e.g. "diag1" (a service diagnosis pointer) matches but "diagnosis1" (box 21)
# does not, and "ch1" (line charge) matches but "charge" (box 20) does not.
_SERVICE_FIELD_RE = re.compile(
    r"^(sv\d|place\d|emg\d|cpt\d|mod\d|diag\d|ch\d|day\d|epsdt\d|type\d|local\d)"
)

REGIONS: Tuple[str, ...] = ("top", "service", "bottom")

# Which bands feed which v2 model. The v2 deployment trains THREE single-band models
# (top / bottom / service) served as three concurrent Modal calls; each owns exactly one
# band. ("fields" is the legacy 2-model layout where top+bottom shared one model.)
MODEL_BANDS: Dict[str, Tuple[str, ...]] = {
    "top": ("top",),
    "bottom": ("bottom",),
    "service": ("service",),
    "fields": ("top", "bottom"),  # legacy: top+bottom in one model
}

# Top-level GT keys (logical view) that live in the BOTTOM band. Everything
# that is not one of these and is not the service-line array is in the TOP band.
_BOTTOM_BOXES = frozenset({
    "box_25_federal_tax_id",
    "box_25_tax_id_type",
    "box_26_patient_account_number",
    "box_27_accept_assignment",
    "box_28_total_charge",
    "box_29_amount_paid",
    "box_31_physician_signature",
    "box_31_physician_signature_date",
    "box_32_service_facility",
    "box_33_billing_provider",
})
_SERVICE_BOX = "box_24_service_lines"


# --------------------------------------------------------------------------- #
# Template geometry
# --------------------------------------------------------------------------- #
def page_size_pt(template: Path | str = DEFAULT_TEMPLATE) -> Tuple[float, float]:
    """(width, height) of page 1 in PDF points."""
    reader = pypdf.PdfReader(str(template))
    box = reader.pages[0].mediabox
    return float(box.width), float(box.height)


def _widget_field_name(widget) -> str | None:
    name = widget.get("/T")
    node = widget
    while name is None and "/Parent" in node:
        node = node["/Parent"].get_object()
        name = node.get("/T")
    return str(name) if name is not None else None


def extract_field_rects_pt(
    template: Path | str = DEFAULT_TEMPLATE,
) -> Dict[str, PtRect]:
    """Map each form field name to the union /Rect of its widgets, in PDF points.

    Radio groups have several kid widgets; we union them so the field's rect
    covers all its on-page marks.
    """
    reader = pypdf.PdfReader(str(template))
    rects: Dict[str, PtRect] = {}
    for page in reader.pages:
        annots = page.get("/Annots")
        if not annots:
            continue
        for ref in annots:
            widget = ref.get_object()
            if widget.get("/Subtype") != "/Widget":
                continue
            rect = widget.get("/Rect")
            if not rect or len(rect) != 4:
                continue
            name = _widget_field_name(widget)
            if name is None:
                continue
            x0, y0, x1, y1 = (float(v) for v in rect)
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)
            if name in rects:
                px0, py0, px1, py1 = rects[name]
                rects[name] = (min(px0, x0), min(py0, y0), max(px1, x1), max(py1, y1))
            else:
                rects[name] = (x0, y0, x1, y1)
    return rects


def _pt_rect_to_px(rect_pt: PtRect, dpi: float, page_h_pt: float) -> PxBox:
    """Convert a y-up PDF-point rect to a y-down image-pixel box."""
    scale = dpi / _BASE_DPI
    x0, y0, x1, y1 = rect_pt
    return (
        int(round(x0 * scale)),
        int(round((page_h_pt - y1) * scale)),  # PDF y-up -> image y-down
        int(round(x1 * scale)),
        int(round((page_h_pt - y0) * scale)),
    )


def region_bands_px(
    dpi: int = 300,
    template: Path | str = DEFAULT_TEMPLATE,
    overlap_frac: float = 0.015,
) -> Dict[str, PxBox]:
    """Full-width top/service/bottom pixel bands for the given render DPI.

    The service band spans the box-24 rows (derived from the sv*/cpt*/... field
    rects); the top band runs from the page top to the service band, and the
    bottom band from the service band to the page bottom. Bands overlap
    slightly (`overlap_frac` of page height) so glyphs straddling a cut are not
    sliced in half.
    """
    page_w_pt, page_h_pt = page_size_pt(template)
    scale = dpi / _BASE_DPI
    page_w = int(round(page_w_pt * scale))
    page_h = int(round(page_h_pt * scale))

    rects = extract_field_rects_pt(template)
    service_px = [
        _pt_rect_to_px(r, dpi, page_h_pt)
        for name, r in rects.items()
        if _SERVICE_FIELD_RE.match(name)
    ]
    if not service_px:
        raise RuntimeError("no box-24 service fields found in template")

    svc_top = min(b[1] for b in service_px)
    svc_bottom = max(b[3] for b in service_px)
    pad = int(round(overlap_frac * page_h))

    def _clamp(v: int) -> int:
        return max(0, min(page_h, v))

    return {
        "top": (0, 0, page_w, _clamp(svc_top + pad)),
        "service": (0, _clamp(svc_top - pad), page_w, _clamp(svc_bottom + pad)),
        "bottom": (0, _clamp(svc_bottom - pad), page_w, page_h),
    }


def crop_regions(
    image: Image.Image,
    dpi: int = 300,
    template: Path | str = DEFAULT_TEMPLATE,
    overlap_frac: float = 0.015,
) -> Dict[str, Image.Image]:
    """Crop a rendered full-page form into {region: PIL image}.

    The image must be the page rasterized at `dpi`. If it was resized after
    rendering, scale the bands accordingly first.
    """
    bands = region_bands_px(dpi, template, overlap_frac)
    return {region: image.crop(box) for region, box in bands.items()}


# --------------------------------------------------------------------------- #
# Ground-truth splitting (operates on the flat GT dict from schema.flatten)
# --------------------------------------------------------------------------- #
def region_of_key(flat_key: str) -> str:
    """Which band a flat GT key (e.g. 'box_33_billing_provider.npi') belongs to."""
    top_level = flat_key.split(".", 1)[0].split("[", 1)[0]
    if top_level == _SERVICE_BOX:
        return "service"
    if top_level in _BOTTOM_BOXES:
        return "bottom"
    return "top"


def split_flat_gt(flat: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Partition a flat GT dict into {region: flat-subset}, preserving order."""
    out: Dict[str, Dict[str, str]] = {r: {} for r in REGIONS}
    for key, value in flat.items():
        out[region_of_key(key)][key] = value
    return out


def gt_for_model(flat: Dict[str, str], model: str) -> Dict[str, str]:
    """Flat GT subset for a v2 model ('fields' = top+bottom, 'service')."""
    bands = set(MODEL_BANDS[model])
    return {k: v for k, v in flat.items() if region_of_key(k) in bands}
