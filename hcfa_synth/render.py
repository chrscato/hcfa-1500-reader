"""PDF → PNG rendering via pypdfium2 (no system poppler required)."""

from __future__ import annotations

import io
from typing import Optional

import pypdfium2 as pdfium
from PIL import Image

# pypdfium2 renders at 72 DPI when scale=1.0. Convert dpi → scale.
_BASE_DPI = 72


def pdf_to_png(pdf_bytes: bytes, dpi: int = 300, page_index: int = 0) -> Image.Image:
    """Render one page of a PDF to a PIL RGB image.

    Args:
        pdf_bytes: raw PDF bytes (e.g. from pdf_fill.fill_pdf).
        dpi: render resolution. 300 is good for OCR; 150 for previews.
        page_index: 0-based page to render.

    Form fields (the data we just filled) require init_forms() + the
    may_draw_forms flag to be drawn into the raster output.
    """
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    pdf.init_forms()
    scale = dpi / _BASE_DPI
    page = pdf[page_index]
    bitmap = page.render(scale=scale, may_draw_forms=True)
    img = bitmap.to_pil().convert("RGB")
    return img


def render_filled_pdf(
    pdf_bytes: bytes,
    *,
    dpi: int = 300,
    page_index: int = 0,
) -> Image.Image:
    """Alias for pdf_to_png with keyword-only options. Convenience entry point."""
    return pdf_to_png(pdf_bytes, dpi=dpi, page_index=page_index)
