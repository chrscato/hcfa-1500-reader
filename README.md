<div align="center">

<img src="docs/assets/logo.png" alt="HCFA-1500 Reader logo" width="120" height="120" />

# HCFA-1500 Reader

**An open toolkit for synthetic CMS-1500 (HCFA) claim forms — generate, evaluate, and fine-tune vision-language models for medical-claim extraction.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-catochris%2Fhcfa--1500-yellow)](https://huggingface.co/datasets/catochris/hcfa-1500)
[![Tests](https://img.shields.io/badge/tests-86%20passing-brightgreen.svg)](#testing)
[![PHI-free](https://img.shields.io/badge/data-100%25%20synthetic%20·%20PHI--free-success.svg)](#disclaimer)

</div>

---

Real CMS-1500 forms carry protected health information (PHI), so they can't be shared, and document-extraction models for them can't be benchmarked in the open. **HCFA-1500 Reader** removes that blocker: it generates unlimited, PHI-free synthetic claim forms with exact ground truth, scores extractor output against them, and fine-tunes a small vision-language model end to end.

It ships three components:

| Package / asset | What it does |
|---|---|
| 🏭 **`hcfa_synth`** | Generates paired `(image, ground_truth.json)` samples across six realistic degradation tiers — pristine digital through worst-case fax/photo. |
| 📊 **`hcfa_eval`** | Scores predictions against ground truth: per-key, per-tier, per-field-class, character error rate, and JSON validity. |
| 🤖 **Fine-tuning notebook** | QLoRA fine-tunes **Qwen2.5-VL-3B** on the published dataset and reports results through the eval harness. |

---

## Table of contents

- [Quickstart](#quickstart)
- [The dataset](#the-dataset)
- [Difficulty tiers](#difficulty-tiers)
- [Ground-truth schema](#ground-truth-schema)
- [Evaluating an extractor](#evaluating-an-extractor)
- [Fine-tuning a VLM](#fine-tuning-a-vlm)
- [Realism guarantees](#realism-guarantees)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)
- [License](#license)

## Quickstart

```bash
git clone https://github.com/chrscato/hcfa-1500-reader.git
cd hcfa-1500-reader
pip install -e .

# generate 20 forms across all tiers
python -m hcfa_synth generate --count 20 --tiers all --seed 42 --out ./data/sanity
```

Output layout:

```
data/sanity/
├── pristine/    00000.png  00000.json  00000.pdf  ...
├── clean_scan/  ...
├── worn_scan/   ...
├── fax/         ...
├── phone_photo/ ...
├── worst/       ...
└── manifest.jsonl     # one line per sample with metadata
```

Every tier is **deterministic given a seed**, so runs are reproducible.

## The dataset

A ready-made 500-sample benchmark is published on the Hugging Face Hub:

**🤗 [`catochris/hcfa-1500`](https://huggingface.co/datasets/catochris/hcfa-1500)** — 404 train / 48 validation / 48 test, stratified across all six tiers, images embedded, with a flat-JSON target per form.

```python
from datasets import load_dataset

ds = load_dataset("catochris/hcfa-1500")
ds["train"][0]["image"]   # PIL image
ds["train"][0]["target"]  # flat-JSON ground truth
```

Or regenerate the full set locally:

```bash
python -m hcfa_synth generate --count 500 --tiers all --seed 0 --out ./data/full --no-pdf
```

(~5 min at 300 DPI, ~2.5 GB.) Twelve ready-to-browse samples (2 per tier) also live in [`examples/`](./examples).

## Difficulty tiers

| Tier | Resembles | Key transforms |
|---|---|---|
| `pristine` | Native digital submission | None — direct render |
| `clean_scan` | Modern office scanner | ±1° skew, mild noise, slight fade |
| `worn_scan` | Old MFP / photocopy | ±3° skew, JPEG 60, fade, speckles |
| `fax` | Faxed claim form | Binary B&W, vertical streaks, downsample |
| `phone_photo` | Phone-snapped printout | Perspective warp, lighting gradient, shadow, blur |
| `worst` | Stress test | Crinkles, staple holes, coffee stain, redaction bars |

The `pristine` → `worst` accuracy gap is the robustness signal that matters for production — where template registration breaks but a VLM can still read.

## Ground-truth schema

Each `.json` has two layers:

- **`fields`** — flat dict keyed by the official PDF field name (e.g. `pt_name`, `cpt1`). Matches what an extractor reading the raw AcroForm sees.
- **`logical`** — nested by the 33 CMS-1500 boxes with normalized values (e.g. `box_2_patient_name.first`, `box_24_service_lines[]`). Matches what an extractor reading the rendered image would naturally produce.

The eval harness flattens `logical` to dot/index paths (`box_24_service_lines[0].procedure_code`) and scores per key; blanks normalize to `""` (one universal "empty" token).

## Evaluating an extractor

`hcfa_eval` turns predictions into a structured report — not just one accuracy number:

```bash
hcfa-eval score \
  --batch data/full \
  --split splits/full/test.jsonl \
  --preds my_predictions.jsonl \
  --model my-extractor \
  --summary-csv runs.csv
```

It reports **populated vs blank accuracy**, **per-tier** accuracy, **per-field-class** exact-match (codes, money, dates, NPIs, names, addresses), **character error rate** on the structured classes, and **JSON-validity rate** — plus a one-row CSV per run for comparing models.

## Fine-tuning a VLM

Format the splits into a portable vision-language dataset and publish it:

```bash
# build a local HF DatasetDict (train/validation/test), images embedded
python -m hcfa_synth.format_for_vlm --out data/hf_vlm

# authenticate once, then build + push in one step
huggingface-cli login            # or: export HF_TOKEN=...
python -m hcfa_synth.format_for_vlm --out data/hf_vlm --push --hub-repo-id <user>/hcfa-1500
```

> Use `push_to_hub` (above), **not** `huggingface_hub.upload_folder(".")` — the latter uploads the whole repo as loose files, which `load_dataset` can't read.

Then open **[`notebooks/finetune_qwen2_5_vl_hcfa.ipynb`](./notebooks/finetune_qwen2_5_vl_hcfa.ipynb)** in Colab. It loads the dataset from the Hub, QLoRA-fine-tunes **Qwen2.5-VL-3B**, and scores the `test` split through `hcfa_eval`. The notebook auto-detects the GPU (T4 / L4 / A100) and sets resolution, epochs, and precision accordingly — pick a runtime and run top to bottom.

## Realism guarantees

- NPIs are **Luhn-valid** with the `80840` prefix (per CMS spec)
- ICD-10 codes drawn from a bundled CMS sample (`hcfa_synth/codes/icd10_sample.csv`, 247 codes)
- Procedure codes from **HCPCS Level II** (`hcpcs_sample.csv`, 132 codes — public-domain, avoids CPT licensing)
- POS codes from the full place-of-service codeset (`pos_codes.csv`, 51 codes)
- Service dates ≤ today; DOB earlier than service date
- Box 28 total = sum of line charges
- Diagnosis pointers (A–L) only reference populated diagnoses
- Tax ID format matches type (SSN `xxx-xx-xxxx`, EIN `xx-xxxxxxx`)
- "Self" relationship mirrors patient demographics into insured fields

## Project structure

```
hcfa-1500-reader/
├── form-cms1500.pdf              # Official CMS-1500 AcroForm (252 fields)
├── hcfa_synth/                   # Synthetic data generator
│   ├── codes/                    #   ICD-10, HCPCS, POS reference CSVs
│   ├── npi.py                    #   Luhn-valid NPI generation
│   ├── records.py                #   Faker-driven record builder
│   ├── pdf_fill.py               #   AcroForm field mapper + filler (pypdf)
│   ├── render.py                 #   PDF→PNG renderer (pypdfium2)
│   ├── augment.py                #   Difficulty-tier transforms (OpenCV)
│   ├── ground_truth.py           #   JSON emitter
│   ├── format_for_vlm.py         #   Splits → Hugging Face VLM dataset
│   ├── pipeline.py               #   End-to-end orchestrator
│   └── __main__.py               #   CLI (hcfa-synth)
├── hcfa_eval/                    # Evaluation harness
│   ├── schema.py                 #   flatten / unflatten ground truth
│   ├── normalize.py              #   field-type-aware normalization
│   ├── scoring.py                #   per-key / per-tier / per-class / CER / validity
│   ├── splits.py                 #   stratified train/val/test
│   └── cli.py                    #   CLI (hcfa-eval)
├── notebooks/                    # QLoRA fine-tuning notebook (Qwen2.5-VL-3B)
├── examples/                     # Small committed sample set (~30 MB)
├── tests/                        # 86 tests across all modules
└── docs/                         # Workplan + dataset card
```

## Testing

```bash
pip install -e ".[vlm]"   # optional extra pulls in `datasets` for the VLM tests
pytest -q
```

86 tests cover record building, NPI validity, PDF fill, rendering, augmentation, the eval harness, and the VLM formatter.

## Roadmap

- [ ] Handwriting tier (v0.2)
- [ ] Map the Box 32 facility NPI (one of the numeric-named PDF fields)
- [ ] Optional OCR-grounding pass for the fine-tuning pipeline
- [ ] Validation against real, de-identified forms

See [`docs/HCFA_OSS_WORKPLAN.md`](./docs/HCFA_OSS_WORKPLAN.md) for the broader plan.

## Contributing

Contributions are welcome. A good flow:

1. Open an issue describing the change (new tier, code-set, schema field, scorer metric, …).
2. Fork and branch from `main`.
3. Add or update tests and keep `pytest -q` green.
4. Open a pull request.

New degradation tiers live in `hcfa_synth/augment.py`; new scoring metrics in `hcfa_eval/scoring.py`.

## Disclaimer

All data produced by this project is **100% synthetic** — identities, addresses, dates, and codes are fabricated (via [Faker](https://faker.readthedocs.io/) and public CMS code samples) and contain **no real PHI**. Medical codes are sampled, not clinically validated. This toolkit is for benchmarking and research; it is **not** a clinical or billing system, and models trained only on this synthetic data should be validated on real, de-identified forms before any production use.

## License

Code is released under the [MIT License](#license). The published dataset ([`catochris/hcfa-1500`](https://huggingface.co/datasets/catochris/hcfa-1500)) is licensed **CC-BY-4.0**.

<div align="center">
<sub>Built for open, reproducible medical-document extraction research.</sub>
</div>
