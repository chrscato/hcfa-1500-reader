"""Modal serverless worker: CMS-1500 (HCFA) PDF -> machine-readable JSON.

Flow (async, S3-key in / S3-keys out):

    VM uploads PDF to S3
      -> POST /submit {bucket, pdf_key}      (returns {call_id})
         -> Extractor.extract (A10G, scale-to-zero):
              S3 PDF -> pymupdf render @ 300 DPI
              -> classify pass (same model, short prompt): is this page a
                 CMS-1500 at all? Uploads can contain EOBs, cover sheets, ...
                 Non-bills skip extraction; result carries doc_type/is_bill.
                 Fail-open: an unparseable classify answer extracts anyway.
                 {"force": true} on /submit skips the gate (reviewer override).
              -> merged Qwen2.5-VL (greedy) -> extract_json -> {flat, logical}
              -> writes <out_prefix>.json and <out_prefix>.png back to S3
      -> GET /result?call_id=...             (poll: {status, result})

Parity with training (do NOT drift these):
  * MINIMAL_PROMPT and the user/image message structure are byte-for-byte the
    prompt the model was fine-tuned on (notebook cell 9 used PROMPT_MODE="minimal").
  * Greedy decode (do_sample=False) + stop at first '}' mirrors the eval cell and
    overrides the base model's generation_config (which ships do_sample=True).
  * The processor's pixel budget (min/max_pixels) travels inside the merged repo,
    so preprocessing matches the training resolution automatically.
  * extract_json() and unflatten() are copied verbatim from the notebook /
    hcfa_eval.schema so the served output matches what the harness scored.

Deploy:
    modal secret create huggingface-secret HF_TOKEN=hf_xxx
    modal secret create aws-s3 AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-1
    modal deploy serving/modal_app.py

Local smoke test (one form, prints result):
    modal run serving/modal_app.py --bucket my-bucket --pdf-key inbox/claim123.pdf
"""

from __future__ import annotations

import io
import json
import re

import modal

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL_REPO = "catochris/qwen2.5vl-3b-hcfa"   # private merged model on the Hub
MODEL_DIR = "/model"                          # baked into the image at build time
RENDER_DPI = 300                              # matches the 2550x3300 synthetic forms
MAX_NEW_TOKENS = 2048                         # cap; generation stops at first '}'

# Trained prompt — must stay identical to notebook cell 9 (PROMPT_MODE="minimal").
MINIMAL_PROMPT = (
    "Extract every field from this CMS-1500 (HCFA) claim form as a single flat JSON "
    'object. Use "" for any blank field. Return only the JSON.'
)

# First-pass gate: zero-shot doc-type classification with a fixed menu. This is
# NOT the trained prompt — it leans on the base model's general instruction
# following, so it's free to change (and may need tuning if the fine-tune turns
# out too narrow to answer anything but MINIMAL_PROMPT).
DOC_TYPES = (
    "hcfa_1500", "ub04", "eob", "medical_record",
    "cover_sheet", "id_card", "correspondence", "other",
)
CLASSIFY_PROMPT = (
    "What kind of document is this page? Choose exactly one type:\n"
    "- hcfa_1500: CMS-1500 / HCFA health insurance claim form (red-ink grid, boxes 1-33)\n"
    "- ub04: UB-04 / CMS-1450 institutional claim form\n"
    "- eob: explanation of benefits or remittance advice\n"
    "- medical_record: clinical notes, operative report, progress notes\n"
    "- cover_sheet: fax or mail cover page\n"
    "- id_card: insurance or identification card\n"
    "- correspondence: letter or other correspondence\n"
    "- other: anything else\n"
    "Also rate how confident you are: high, medium, or low.\n"
    'Return only JSON: {"doc_type": "<type>", "confidence": "<high|medium|low>"}'
)
CLASSIFY_MAX_NEW_TOKENS = 48

# Non-bill summary: once the gate decides a page ISN'T a claim form, we still
# want to know what it is. Free-text, a sentence or two — notes, reports, EOBs,
# cover sheets. Like CLASSIFY_PROMPT this leans on base-model instruction
# following (NOT the frozen extraction prompt), so it's free to tune.
SUMMARIZE_PROMPT = (
    "This page is not a CMS-1500 claim form. In 1-2 sentences, say what kind of "
    "document it is and summarize its key content (e.g. a medical record, "
    "operative report, EOB/remittance, cover sheet, or letter). Note any patient "
    "name, dates of service, or provider you can read. Return only the summary."
)
SUMMARIZE_MAX_NEW_TOKENS = 256

HF_SECRET = modal.Secret.from_name("huggingface-secret")  # HF_TOKEN (read access)
AWS_SECRET = modal.Secret.from_name("aws-s3")             # AWS_ACCESS_KEY_ID / SECRET / DEFAULT_REGION


# --------------------------------------------------------------------------- #
# Image — deps + bake the merged model into the image layer (fast cold starts)
# --------------------------------------------------------------------------- #
def _download_model():
    import os
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_REPO,
        local_dir=MODEL_DIR,
        token=os.environ["HF_TOKEN"],
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==5.9.0",
        "accelerate>=0.34",
        "qwen-vl-utils==0.0.11",
        "pillow>=10",
        "pymupdf>=1.24",
        "boto3>=1.34",
        "huggingface_hub>=0.25",
        "hf_transfer>=0.1.8",
        "fastapi[standard]",
        # Match the transformers version that SAVED the merged model (recorded in
        # its config.json as transformers_version=5.9.0). The v5 config schema
        # (nested text_config) won't parse on 4.x — loading crashes with
        # `'dict' object has no attribute 'to_dict'`.
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Bake the 7.5 GB merged model into the image so containers start without a
    # 7.5 GB download every cold start. Re-runs on a new MODEL_REPO/commit.
    .run_function(_download_model, secrets=[HF_SECRET])
)

app = modal.App("hcfa-extractor", image=image)


# --------------------------------------------------------------------------- #
# Output helpers — copied verbatim from the notebook / hcfa_eval.schema so the
# served JSON matches exactly what the eval harness scored. Keep in sync.
# --------------------------------------------------------------------------- #
def extract_json(text: str):
    """Parse the model's JSON, salvaging truncated output (keep complete pairs)."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?", "", t).strip().strip("`").strip()
    start = t.find("{")
    if start == -1:
        return None
    frag = t[start:]
    try:
        return json.loads(frag)
    except Exception:
        pass
    cut = frag.rfind('",')
    if cut != -1:
        try:
            return json.loads(frag[: cut + 1] + "}")
        except Exception:
            pass
    return None


_INDEX_RE = re.compile(r"^(.*)\[(\d+)\]$")


def _set_path(root, path, value):
    parts = path.split(".")
    cur = root
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        m = _INDEX_RE.match(part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if key not in cur or not isinstance(cur[key], list):
                cur[key] = []
            while len(cur[key]) <= idx:
                cur[key].append({})
            if is_last:
                cur[key][idx] = value
            else:
                if not isinstance(cur[key][idx], dict):
                    cur[key][idx] = {}
                cur = cur[key][idx]
        else:
            if is_last:
                cur[part] = value
            else:
                if part not in cur or not isinstance(cur[part], dict):
                    cur[part] = {}
                cur = cur[part]


def unflatten(flat: dict) -> dict:
    """Inverse of hcfa_eval.schema.flatten — rebuild the nested `logical` view."""
    root: dict = {}
    for path, value in flat.items():
        _set_path(root, path, value)
    return root


def _render_pdf_first_page(pdf_bytes: bytes):
    """Render page 0 of a PDF to an RGB PIL image at RENDER_DPI."""
    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("PDF has no pages")
    pix = doc.load_page(0).get_pixmap(dpi=RENDER_DPI)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _load_page_image(data: bytes, key: str):
    """Get an RGB image from a single page. PDFs are rendered at RENDER_DPI;
    image inputs (PNG/JPG/TIFF) are loaded directly. The front door already
    split multi-page PDFs/TIFFs, so each call handles exactly one page."""
    from PIL import Image

    if key.lower().endswith(".pdf"):
        return _render_pdf_first_page(data)
    return Image.open(io.BytesIO(data)).convert("RGB")


# --------------------------------------------------------------------------- #
# GPU worker
# --------------------------------------------------------------------------- #
@app.cls(
    gpu="A10G",
    secrets=[AWS_SECRET],
    timeout=600,            # generous: cold start + render + generate
    scaledown_window=300,   # stay warm 5 min after a request, then scale to zero
    max_containers=5,       # ceiling, not a reservation: scales 0->5 with demand and
                            # back to 0 when idle. Single pages use 1 GPU (same cost as
                            # max=1); multi-page docs fan out across up to 5 GPUs. Stays
                            # under the account's 10-GPU cap no matter what.
)
class Extractor:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            dtype=torch.bfloat16,         # v5 arg name (was torch_dtype); A10G is Ampere -> bf16 OK
            device_map="cuda",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR)
        self.processor.tokenizer.padding_side = "left"

    def _generate(self, image, prompt=MINIMAL_PROMPT, max_new_tokens=MAX_NEW_TOKENS,
                  stop_strings=("}",)):
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, return_tensors="pt"
        ).to("cuda")

        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                    # greedy — deterministic extraction
                # flat JSON ends at one '}' (default); free-text summary passes
                # None so it isn't cut off at a stray brace.
                stop_strings=list(stop_strings) if stop_strings else None,
                tokenizer=self.processor.tokenizer,  # required for stop_strings
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def _classify(self, image):
        """First pass: what is this page? Returns (doc_type, confidence, raw).

        doc_type is one of DOC_TYPES, or "unknown" when the answer didn't parse
        or named a type outside the menu. confidence is high/medium/low (low
        when missing). Callers fail open: "unknown" — and any non-bill verdict
        the model itself rates "low" — still gets extracted, because a wasted
        extraction beats a silently dropped real claim.
        """
        raw = self._generate(image, CLASSIFY_PROMPT, CLASSIFY_MAX_NEW_TOKENS)
        parsed = extract_json(raw)
        doc_type = str((parsed or {}).get("doc_type", "")).strip().lower()
        confidence = str((parsed or {}).get("confidence", "")).strip().lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "low"
        return (doc_type if doc_type in DOC_TYPES else "unknown"), confidence, raw

    def _summarize(self, image) -> str:
        """Non-bill pages: a short free-text description of what the page is and
        its key content. Free-form, so generation runs without the '}' stop."""
        return self._generate(image, SUMMARIZE_PROMPT, SUMMARIZE_MAX_NEW_TOKENS,
                              stop_strings=None).strip()

    @modal.method()
    def extract(self, bucket: str, pdf_key: str, out_prefix: str | None = None,
                force: bool = False) -> dict:
        import boto3

        if out_prefix is None:
            base = pdf_key.rsplit("/", 1)[-1]
            base = base[:-4] if base.lower().endswith(".pdf") else base
            out_prefix = f"extractions/{base}"
        json_key = f"{out_prefix}.json"
        png_key = f"{out_prefix}.png"

        s3 = boto3.client("s3")
        pdf_bytes = s3.get_object(Bucket=bucket, Key=pdf_key)["Body"].read()

        image = _load_page_image(pdf_bytes, pdf_key)

        # First pass: only spend the full extraction on pages that look like a
        # CMS-1500. force=True (reviewer override) skips straight to extraction.
        # Skipping requires a confident non-bill verdict — uncertain pages
        # extract anyway and carry their doc_type for the reviewer.
        doc_type, confidence, classify_raw = ("", "", "") if force else self._classify(image)
        is_bill = force or doc_type in ("hcfa_1500", "unknown") or confidence == "low"

        summary = ""
        if is_bill:
            raw = self._generate(image)
            fields = extract_json(raw)
            parse_ok = isinstance(fields, dict)
            fields = fields if parse_ok else {}
        else:
            # not a claim form: skip field extraction, but summarize what it is
            # so the non-bill worklane has something to show (notes, reports...).
            summary = self._summarize(image)
            raw, fields, parse_ok = classify_raw, {}, True

        result = {
            "sample_id": out_prefix.rsplit("/", 1)[-1],
            "source": f"s3://{bucket}/{pdf_key}",
            "model": MODEL_REPO,
            "doc_type": doc_type,              # "" when forced (gate skipped)
            "doc_confidence": confidence,      # classifier's own high/medium/low
            "is_bill": is_bill,
            "parse_ok": parse_ok,
            "summary": summary,                # non-bill free-text; "" for bills
            "fields": fields,                  # flat str->str (scoring-ready)
            "logical": unflatten(fields),      # nested view
            "raw": raw,                         # full decode (debugging / salvage)
        }

        # rendered page (what the model actually saw) for human review / audit
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        s3.put_object(Bucket=bucket, Key=png_key, Body=buf.getvalue(), ContentType="image/png")
        s3.put_object(
            Bucket=bucket,
            Key=json_key,
            Body=json.dumps(result, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

        return {
            "bucket": bucket,
            "json_key": json_key,
            "png_key": png_key,
            "doc_type": doc_type,
            "is_bill": is_bill,
            "parse_ok": parse_ok,
            "num_fields": sum(1 for v in fields.values() if v),
        }


# --------------------------------------------------------------------------- #
# Async HTTP surface for the VM: submit -> poll. Lightweight (no GPU).
# --------------------------------------------------------------------------- #
@app.function()
@modal.fastapi_endpoint(method="POST")
def submit(payload: dict) -> dict:
    """Body: {"bucket", "pdf_key", "out_prefix"?, "force"?} -> {"call_id": ...}.

    force=true skips the classification gate and extracts unconditionally
    (used by the front door's "extract anyway" reviewer override)."""
    call = Extractor().extract.spawn(
        payload["bucket"], payload["pdf_key"], payload.get("out_prefix"),
        bool(payload.get("force", False)),
    )
    return {"call_id": call.object_id}


@app.function()
@modal.fastapi_endpoint(method="GET")
def result(call_id: str) -> dict:
    """?call_id=... -> {"status": "running"} | {"status": "done", "result": {...}}."""
    fc = modal.FunctionCall.from_id(call_id)
    try:
        return {"status": "done", "result": fc.get(timeout=0)}
    except TimeoutError:
        return {"status": "running"}


# --------------------------------------------------------------------------- #
# CLI smoke test: `modal run serving/modal_app.py --bucket B --pdf-key K`
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main(bucket: str, pdf_key: str, out_prefix: str = None, force: bool = False):
    res = Extractor().extract.remote(bucket, pdf_key, out_prefix, force)
    print(json.dumps(res, indent=2))
