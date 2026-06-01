---
license: cc-by-4.0
pretty_name: HCFA-1500 Synthetic Claim Forms (CMS-1500)
language:
- en
task_categories:
- image-to-text
- visual-question-answering
tags:
- document-ai
- vlm
- ocr
- key-information-extraction
- medical-claims
- cms-1500
- hcfa-1500
- synthetic
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# HCFA-1500 Synthetic Claim Forms

500 synthetic **CMS-1500 (HCFA-1500)** health-insurance claim forms rendered across six
image-degradation tiers, each paired with a **flat JSON** ground-truth record. Built for
fine-tuning and benchmarking **vision-language models** on dense, fixed-template
**document field extraction** (image → structured JSON).

The data is **100% synthetic** — there is **no real PHI**. Patient/provider identities and
addresses come from [Faker](https://faker.readthedocs.io/); medical codes are drawn from
**public-domain** CMS reference sets (HCPCS Level II, ICD-10-CM, Place-of-Service).

- **Source code & generator:** https://github.com/chrscato/hcfa-1500-reader
- **Task:** read a CMS-1500 form image and emit a flat JSON object of ~118 field values
- **Difficulty:** exact character transcription (codes, dates, money, NPIs) + robustness
  to scan/fax/photo degradation

## Quick start

```python
from datasets import load_dataset

ds = load_dataset("catochris/hcfa-1500")
ex = ds["train"][0]
ex["image"]      # PIL.Image (PNG, ~2550×3300, 300 DPI)
ex["prompt"]     # fixed extraction instruction (lists the schema keys)
ex["target"]     # the flat-JSON answer string
```

## Dataset structure

### Columns

| column | type | description |
|---|---|---|
| `image` | `Image` | rendered CMS-1500 form, 8.5"×11" @ 300 DPI (~2550×3300 PNG), embedded |
| `prompt` | `string` | a fixed instruction that lists all ~118 schema keys to extract |
| `target` | `string` | the ground-truth answer: a flat JSON object, ~118 keys, `""` for blank fields |
| `sample_id` | `string` | stable id (e.g. `00007`) |
| `tier` | `string` | degradation tier (see below) |
| `split` | `string` | `train` / `val` / `test` |

`prompt` and `target` are plain strings so the dataset is model-agnostic — wrap them in
whatever chat template your VLM expects. The `prompt` lists the full key set (handy for
zero-shot eval); for fine-tuning a model that learns the schema you can swap in a shorter
instruction (keep training and serving identical).

### Splits

Stratified by tier (deterministic), ~81/10/10%:

| split | rows | per tier |
|---|---|---|
| train | 404 | ~67–68 each |
| validation | 48 | 8 each |
| test | 48 | 8 each |
| **total** | **500** | ~83–84 each |

### Degradation tiers

Each clean render is degraded with OpenCV to mimic real-world capture:

| tier | what it simulates |
|---|---|
| `pristine` | clean digital render (passthrough) |
| `clean_scan` | modern office scanner |
| `worn_scan` | old MFP / multi-generation photocopy |
| `fax` | low-res faxed copy |
| `phone_photo` | phone-snapped printout (perspective, uneven lighting) |
| `worst` | stacked worst-case artifacts |

Report accuracy **per tier** — the `pristine` → `worst` gap is the robustness number that
matters for production.

### Target schema

`target` is a flat dict whose keys are dot/index paths over the 33 logical CMS-1500 boxes,
e.g.:

```json
{
  "box_1_insurance_type": "Medicare",
  "box_1a_insured_id": "RD492655486",
  "box_2_patient_name.last": "FOSTER",
  "box_3_patient_birth": "05/03/1960",
  "box_21_diagnoses": "E86.0 | R63.0",
  "box_24_service_lines[0].procedure_code": "J0129",
  "box_24_service_lines[0].charges": "263.78",
  "box_33_billing_provider.npi": "1234567893"
}
```

Conventions: scalar lists (diagnoses, modifiers) are joined with ` | `; service lines are
indexed (`box_24_service_lines[0..]`); every blank/missing field is the empty string `""`
(one universal "blank" token). The repo's `hcfa_eval` package flattens/normalizes/scores
this exact format (per-key, per-tier, per-field-class, CER, JSON-validity).

## Dataset creation

1. **Records** — `Faker` draws patient/insured/provider identities, dates, money, and NPIs
   (Luhn-valid); procedure/diagnosis/POS codes are sampled from public CMS reference CSVs.
2. **Fill** — values are written into the official CMS-1500 AcroForm (`pypdf`).
3. **Render** — the filled PDF is rasterized to PNG at 300 DPI (`pypdfium2`).
4. **Degrade** — each page is pushed through one of the six tiers (`OpenCV`).
5. **Label** — the same record object is emitted as ground truth, so image and label
   cannot drift.

## Considerations & limitations

- **Synthetic, not real.** The model will fit Faker's value distributions and these
  specific degradation operators — **not** real scanner/fax/photo artifacts, handwriting,
  stamps, or unusual payer layouts. **Validate on real, de-identified forms before
  trusting production numbers.** Treat that as the gating metric, not synthetic-test
  accuracy.
- **No PHI.** All identities are fabricated; any resemblance to real people is coincidental.
- **Small.** 500 samples — enough for QLoRA on a fixed template, but watch overfitting and
  use the `val` split for early stopping.
- **Codes are sampled, not clinically coherent.** Procedure/diagnosis pairings are random,
  not medically validated.

## Licensing

Released under **CC-BY-4.0** — free to use, share, and adapt (including commercially) with
attribution. The data is fully synthetic and uses only public-domain CMS reference codes
(HCPCS Level II, ICD-10-CM, POS); the CMS-1500 is the standard NUCC claim-form template.
The generator source code is MIT-licensed in the linked repository.

## Citation

```bibtex
@misc{hcfa1500_synth_2026,
  title  = {HCFA-1500 Synthetic Claim Forms (CMS-1500)},
  author = {Cato, Chris},
  year   = {2026},
  howpublished = {Hugging Face Datasets},
  url    = {https://huggingface.co/datasets/catochris/hcfa-1500}
}
```
