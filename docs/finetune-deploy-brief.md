# HCFA-1500 Reader — Fine-tune & Production Brief

Briefing doc for researching the **fine-tuning + production deployment** of a
visual-LLM extractor for CMS-1500 (HCFA) claim forms. Scope is deliberately narrow:
**model choice → training setup → serving setup.** Everything below is anchored to
what already exists in this repo.

---

## 1. Current state (what's already built)

- **Synthetic data generator** (`hcfa_synth`): produces CMS-1500 images + labels.
- **~500 samples**, rendered across **6 degradation tiers**:
  `pristine`, `clean_scan`, `fax`, `phone_photo`, `worn_scan`, `worst`
  (~83–84 each). Rendered via `pypdfium2`; degradation via OpenCV.
- **Splits** (`splits/full/`): **404 train / 48 val / 48 test** (`*.jsonl` manifests
  pointing at `data/full/<tier>/<id>.png` + `<id>.json`).
- **Label format**: per-image `<id>.json` with `fields: {...}` — a **flat dict of
  ~100+ keys** (e.g. `pt_name`, `birth_mm`, `cpt1`, `mod1a`, `ch1`, `diagnosis1`,
  `local2`, ...). One form = one flat record.
- **Eval harness** (`hcfa_eval`): `schema.py` flattens nested GT to dot-paths
  (`box_24_service_lines[0].procedure_code`) / joins scalar lists with ` | `;
  `scoring.py` does **per-key scoring**. Designed around a model that **emits flat
  JSON**. Empty/missing → `""` (one universal "blank" token).
- **Design intent already in code**: schema.py comment states the target is a
  **"3B VLM"** — flat JSON chosen because it's easier for a small VLM to produce,
  easier to constrain at decode, and trivially scoreable per-key.

### Hard constraints / facts
- **Fine-tune budget**: **$25 of rented GPU hours.**
- **Production shape**: user uploads **PDF or PNG → queue → serverless GPU spins up
  → processes → returns structured JSON.**
- **Data is 100% synthetic** (Faker distributions + synthetic degradation). No real
  forms in the training set yet.
- **Platform**: Windows dev box; training/serving will be on rented Linux GPUs.

---

## 2. The task, framed precisely

Fixed-template, dense **document field extraction**:
- The CMS-1500 layout is **fixed** — ~100 known field boxes in known positions.
- Output is a **fixed flat JSON schema** (already defined by the label files).
- The difficulty is **not reasoning** — it's (a) **exact character transcription**
  of codes/dates/money/IDs, and (b) **robustness to degradation** (skew, fax noise,
  phone-photo perspective) where naive template-registration breaks.

This framing matters for model choice: this is a **transfer/fine-tune-on-one-task**
problem, not a general chat-VLM problem.

---

## 3. Pipeline architecture to evaluate (OCR + VLM hybrid)

Do **not** treat OCR vs VLM as either/or. For a dense fixed template the strong
baseline is a **hybrid**:

```
PDF/PNG in
  → render to image @ ~300 DPI  (pypdfium2, already in repo)
  → OCR pass → tokens + bounding boxes        [exact characters]
  → prompt = [image] + [OCR tokens w/ coords as auxiliary text]
  → fine-tuned VLM → flat JSON                [layout + field assignment]
  → constrained / validated decode            [per-field regex/grammar]
  → score + validate → emit
```

Rationale to verify during research:
- **VLMs are weak at exact digit strings** (`489.38`, NPI, CPT `V2300`). OCR grounds
  them — the model copies exact tokens instead of hallucinating.
- **The VLM earns its keep on the degraded tiers** (`fax`, `phone_photo`, `worst`)
  where box-cropping/registration fails.
- **Resolution is the real lever, not param count.** ~100 small-text fields need
  legible input — research native-high-res (≥896px) vs tiling/pan-and-scan.
- The **flat-JSON schema enables constrained decoding** (dates = digits, `sex ∈ {M,F}`,
  money = `\d+\.\d{2}`). Worth testing as accuracy insurance.

Alternative architectures worth a paragraph of research each:
- **Pure VLM** (image → JSON, no OCR): simpler; test if exact-transcription accuracy
  is good enough on degraded tiers.
- **Layout models** (LayoutLMv3 / Donut / DocTR): OCR+layout encoders, often smaller
  and cheaper than generative VLMs; Donut is OCR-free. Compare against VLM on the
  same eval harness.
- **Template registration + per-box OCR**: near-deterministic on clean tiers, fails
  on `worst`/`phone_photo`. Could be a cheap fast-path with VLM fallback.

---

## 4. Open-source models to research (runnable + fine-tunable locally)

All of these are on Hugging Face, open-weights, and fine-tunable with QLoRA on a
24GB GPU. Verify current versions/licenses at research time (space moves fast).

### Primary candidates (small VLMs, ~2–4B, document-oriented)
| Model | ~Params | Why it's a candidate | Watch for |
|---|---|---|---|
| **PaliGemma 2** | 3B / 10B / 28B | **Purpose-built for fine-tuning** on vision tasks (OCR, doc extraction). Native **896×896**. 3B is the lead candidate. Gemma family (matches original intent). | Base model — needs fine-tune to be useful; check license terms. |
| **Qwen2.5-VL (3B / 7B)** | 3B / 7B | Strong document/OCR benchmarks, dynamic resolution, bounding-box/grounding output. Very active ecosystem + tooling. | 7B may exceed budget/serving sweet spot; check exact version. |
| **InternVL2.5 / InternVL3 (2B / 4B)** | 2–4B | Competitive doc-VLM scores at small sizes; high-res tiling. | License nuance per release. |
| **SmolVLM2 (2.2B)** | ~2.2B | Tiny, cheap to serve, fast cold start. Good baseline floor. | May trade accuracy on dense fields. |
| **Gemma 3 (4B)** | 4B | Generalist multimodal chat; pan-and-scan for high res. | Chat model bent to narrow task; not doc-specialized. |

### Non-generative / specialist alternatives (cheaper, narrower)
| Model | Type | Why consider |
|---|---|---|
| **Donut** (naver-clava) | OCR-free doc transformer | Designed exactly for image→structured-JSON form parsing. Small. Strong baseline to beat. |
| **LayoutLMv3** | OCR+layout encoder | Token-classification framing of field extraction; small + cheap. |
| **DocTR / PaddleOCR** | OCR engines | The OCR half of the hybrid; PaddleOCR strong on dense print. |

### Recommendation to validate
Start with **PaliGemma 2 3B** (matches the "3B VLM" design intent, native high-res,
built for this) and benchmark **Qwen2.5-VL 3B** + **Donut** as the two comparators on
the existing eval harness. Pick on **per-field accuracy on the degraded tiers**, not
on clean.

---

## 5. Fine-tune setup to research

- **Method**: **QLoRA** (4-bit base + LoRA adapters). Full fine-tune is unnecessary
  and won't fit cheaply.
- **VRAM**: a 3–4B VLM under QLoRA fits in **24 GB**.
- **GPU for training**: single **RTX 4090 (24GB)** or **L4 (24GB)** on Runpod/Vast,
  ~**$0.35–0.70/hr**.
- **Budget math**: 404 examples × ~3–5 epochs of QLoRA ≈ **1–3 hrs/run**. $25 buys
  **~40–60 GPU-hrs** on a 4090 → room for **multiple runs / sweeps**, even a 12B
  comparison. **Training is data-constrained, not budget-constrained.**
- **Cost cliff**: A100 80GB (~$1.2–1.9/hr) / H100 (~$2–3/hr) buy nothing here — QLoRA
  3B doesn't need the VRAM. **Stay on 24GB-class.**
- **Frameworks to evaluate**: HF `transformers` + `peft` + `trl` (`SFTTrainer`), or
  **Unsloth** (faster/less VRAM, check VLM support for chosen model), or
  **Llama-Factory** (config-driven multimodal SFT).
- **Data formatting work needed**: convert `splits/full/train.jsonl` + label JSONs
  (+ optional OCR pass) into the chat/instruction format the chosen model expects
  (image + prompt → flat-JSON response). The eval harness's flatten() defines the
  target string.

### Things to decide during research
- Native high-res vs tiling for the chosen model (affects field legibility most).
- Whether to inject OCR tokens+coords into the training prompt (train/serve must match).
- Constrained decoding library (Outlines / XGrammar / lm-format-enforcer) for the
  flat-JSON schema.
- Epochs / LoRA rank / LR — watch **overfitting to synthetic distribution** (only 404
  examples, all synthetic). Use the `val` split early-stop.

---

## 6. Production / deployment setup to research

Shape is correct: **upload → queue → serverless GPU → JSON out.**

- **Serverless GPU platforms**: **Modal**, **Runpod Serverless**, **Beam**, Replicate.
  Want **scale-to-zero** + **per-second billing** for bursty upload traffic.
- **Inference GPU**: **L4 (24GB)** or **A10G** — the cost-effective tier,
  ~$0.60–0.90/hr or per-second. A 3B model = ~6GB fp16 / ~3–4GB in 4-bit.
- **Cold start is the real prod cost**, not throughput. Smaller model = faster weight
  load = cheaper bursts. Another vote for **3B over 12B**. Research: keeping a warm
  pool vs pure scale-to-zero; baking weights into the image vs volume mount.
- **OCR placement**: run OCR (PaddleOCR/Tesseract) **on CPU in the same container**, or
  call a managed API (Azure Document Intelligence / AWS Textract). **Don't burn GPU on
  OCR.**
- **Serving runtime to evaluate**: **vLLM** (has VLM support for several of these),
  TGI, or plain `transformers` for a first cut. Confirm vLLM supports the chosen
  model's vision path.
- **Inference cost framing**: at 3B on L4 with per-second billing, per-form cost is
  dominated by cold start + image-token count, not raw FLOPs. Measure forms/sec warm
  and cold-start seconds for the chosen model.

### "Right size GPU" boundary (summary)
- **3–4B params on 24GB-class GPUs** is the cost/effectiveness sweet spot.
- Above 24GB VRAM → paying for nothing on this task.
- Above ~4B params → hurts cold start / prod cost, only justified if 3–4B can't hit
  the accuracy bar **with OCR grounding** (prove it on the eval harness first).

---

## 7. Big risk to keep front-of-mind

**The training data is 100% synthetic.** 404 examples is enough for QLoRA on a fixed
template, but the model will fit Faker's value distributions and your own degradation
operators — not real-world scanner/fax/photo artifacts, handwriting, stamps, or odd
payer layouts. **Before trusting production numbers, validate on a handful of real
(de-identified) forms** through the existing eval harness. Consider this the gating
metric, not synthetic-test accuracy.

---

## 8. Concrete decision checklist

- [ ] Pick architecture: hybrid OCR+VLM (lead) vs pure VLM vs Donut/layout — bench all
      three on the eval harness, degraded tiers weighted.
- [ ] Pick model: PaliGemma 2 3B (lead) vs Qwen2.5-VL 3B vs Donut.
- [ ] Pick training framework: trl/peft vs Unsloth vs Llama-Factory.
- [ ] Build the data formatter (splits + labels + optional OCR → chat/JSON format).
- [ ] QLoRA run on a 4090/L4; early-stop on `val`; sweep within $25.
- [ ] Add constrained decoding for the flat schema.
- [ ] Validate on real de-identified forms.
- [ ] Serve on Modal/Runpod serverless, L4/A10G, scale-to-zero; measure cold start.
- [ ] OCR on CPU in-container or managed API.
