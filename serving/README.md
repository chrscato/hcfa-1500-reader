# Serving — Modal worker

Turns an uploaded CMS-1500 PDF into machine-readable JSON using the fine-tuned
`catochris/qwen2.5vl-3b-hcfa` model, on a scale-to-zero A10G GPU.

```
VM uploads PDF -> S3
  POST /submit {bucket, pdf_key}            -> {call_id}
     Extractor.extract  (A10G, cold-starts on demand):
        S3 PDF -> render @300 DPI -> Qwen2.5-VL (greedy) -> JSON
        writes <out_prefix>.json + <out_prefix>.png to S3
  GET  /result?call_id=...                  -> {status, result}
```

## One-time setup

1. **Install + auth Modal** (on the machine that deploys — your VM or laptop):
   ```bash
   pip install modal
   modal token new
   ```
2. **Create secrets** (the model repo is private; S3 needs creds):
   ```bash
   modal secret create huggingface-secret HF_TOKEN=hf_xxx        # read token is enough
   modal secret create aws-s3 \
       AWS_ACCESS_KEY_ID=AKIA... \
       AWS_SECRET_ACCESS_KEY=... \
       AWS_DEFAULT_REGION=us-east-1
   ```
   The IAM principal needs `s3:GetObject` on the input prefix and `s3:PutObject`
   on the output prefix of your bucket.

## Deploy

```bash
modal deploy serving/modal_app.py
```

First deploy bakes the 7.5 GB model into the image (a few minutes). Deploy
prints the public URLs for `submit` and `result`.

## Use from the VM

```python
import requests, time

BASE = "https://<your-workspace>--hcfa-extractor-submit.modal.run"
RESULT = "https://<your-workspace>--hcfa-extractor-result.modal.run"

# VM has already put the PDF at s3://my-bucket/inbox/claim123.pdf
call_id = requests.post(BASE, json={
    "bucket": "my-bucket",
    "pdf_key": "inbox/claim123.pdf",
    # "out_prefix": "extractions/claim123",   # optional; defaults to extractions/<filename>
}).json()["call_id"]

while True:
    r = requests.get(RESULT, params={"call_id": call_id}).json()
    if r["status"] == "done":
        print(r["result"])   # {json_key, png_key, parse_ok, num_fields}
        break
    time.sleep(3)
```

Outputs land in S3:
- `extractions/claim123.json` — `{fields, logical, raw, parse_ok, source, model}`
- `extractions/claim123.png` — the rendered page the model actually saw (audit trail)

## Smoke test (no VM needed)

```bash
modal run serving/modal_app.py --bucket my-bucket --pdf-key inbox/claim123.pdf
```

## Parity notes (why this matches the eval numbers)

- Uses the **exact** training prompt (`MINIMAL_PROMPT`) and message structure.
- **Greedy** decode (`do_sample=False`) — overrides the base model's
  `generation_config` default of `do_sample=True`, so output is deterministic.
- Renders at **300 DPI**; the processor's baked-in `max_pixels` then downscales
  identically to training, so the model sees training-resolution input.
- `extract_json` / `unflatten` are copied from the notebook / `hcfa_eval.schema`
  so served JSON matches what the harness scored. If you change them there,
  change them here too.

## Tuning

- **Cost**: `scaledown_window=300` keeps a worker warm 5 min after the last
  request. Lower it to scale to zero faster (more cold starts), raise it for
  steadier traffic.
- **Multi-page scans**: the worker renders page 0 only (HCFA is one page). If
  real uploads are multi-page, add page selection in `_render_pdf_first_page`.
- **Parse failures**: `result.parse_ok=false` means the model's JSON didn't
  parse even after salvage. The raw decode is saved in the `.json` for triage —
  this is the same 25% failure mode flagged in eval; worth monitoring.
