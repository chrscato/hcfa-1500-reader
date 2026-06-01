# hcfa-synth

Synthetic CMS-1500 (HCFA) claim form generator. Produces paired `(image, ground_truth.json)` samples across a difficulty spectrum — pristine digital through worst-case scanned/photographed — for benchmarking medical-claim extraction pipelines.

## Why

Real CMS-1500 forms contain PHI; you can't ship them. This generator emits unlimited PHI-free synthetic forms with matching ground truth, so extraction models can be trained and benchmarked in the open.

## Quickstart

```bash
pip install -e .
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

## Difficulty tiers

| Tier | Resembles | Key transforms |
|---|---|---|
| `pristine` | Native digital submission | None — direct render |
| `clean_scan` | Modern office scanner | ±1° skew, mild noise, slight fade |
| `worn_scan` | Old MFP / photocopy | ±3° skew, JPEG 60, fade, speckles |
| `fax` | Faxed claim form | Binary B&W, vertical streaks, downsample |
| `phone_photo` | Phone-snapped printout | Perspective warp, lighting gradient, shadow, blur |
| `worst` | Stress test | Crinkles, staple holes, coffee stain, redaction bars |

All tiers are deterministic given a seed.

## Ground-truth schema

Each `.json` has two layers:

- **`fields`** — flat dict keyed by the official PDF field name (e.g. `pt_name`, `cpt1`). Matches what an extractor reading the raw AcroForm sees.
- **`logical`** — nested by the 33 CMS-1500 boxes with normalized values (e.g. `box_2_patient_name.first`, `box_24_service_lines[]`). Matches what an extractor reading the rendered image would naturally produce.

Eval harnesses can score against either layer.

## Realism guarantees

- NPIs are **Luhn-valid** with the `80840` prefix (per CMS spec)
- ICD-10 codes drawn from the bundled CMS sample (`hcfa_synth/codes/icd10_sample.csv`, 247 codes)
- Procedure codes from HCPCS Level II (`hcpcs_sample.csv`, 132 codes — avoids CPT licensing)
- POS codes from the full place-of-service codeset (`pos_codes.csv`, 51 codes)
- Service dates ≤ today; DOB earlier than service date
- Box 28 total = sum of line charges
- Diagnosis pointers (A–L) only reference populated diagnoses
- Tax ID format matches type (SSN `xxx-xx-xxxx`, EIN `xx-xxxxxxx`)
- "Self" relationship mirrors patient demographics into insured fields

## Examples

12 ready-to-browse samples (2 per tier, 200 DPI) live in [`examples/`](./examples). Open any `.png` next to its `.json` to see the form and the ground truth side-by-side.

## Full benchmark dataset

The 500-sample benchmark dataset is published on HuggingFace Datasets — see [TODO: HuggingFace link, published in Week 4].

To regenerate the full dataset locally:

```bash
python -m hcfa_synth generate --count 500 --tiers all --seed 0 --out ./data/full --no-pdf
```

This takes ~5 minutes at 300 DPI and produces ~2.5 GB.

## Fine-tuning

Format the splits into a vision-language dataset (one fixed prompt + flat-JSON target
per image, images embedded so it's portable) and publish it to the Hub:

```bash
# build a local HF DatasetDict (train/validation/test)
python -m hcfa_synth.format_for_vlm --out data/hf_vlm

# authenticate once, then build + push in one step
huggingface-cli login            # or: export HF_TOKEN=...
python -m hcfa_synth.format_for_vlm --out data/hf_vlm --push --hub-repo-id <user>/hcfa-1500
```

> Use `push_to_hub` (above), **not** `huggingface_hub.upload_folder(".")` — the latter
> uploads the whole repo as loose files, which `load_dataset` can't read.

Then open [`notebooks/finetune_qwen2_5_vl_hcfa.ipynb`](./notebooks/finetune_qwen2_5_vl_hcfa.ipynb)
in Colab: it loads the dataset from the Hub, QLoRA-fine-tunes **Qwen2.5-VL-3B**, and
scores the `test` split with the `hcfa_eval` harness (per-tier, per-class, CER, JSON
validity).

## Project structure

```
hcfa-1500-reader/
├── form-cms1500.pdf           # Official CMS-1500 AcroForm (252 fields)
├── hcfa_synth/
│   ├── codes/                 # ICD-10, HCPCS, POS reference CSVs
│   ├── npi.py                 # Luhn-valid NPI generation
│   ├── records.py             # Faker-driven record builder
│   ├── pdf_fill.py            # AcroForm field mapper + filler (pypdf)
│   ├── render.py              # PDF→PNG renderer (pypdfium2)
│   ├── augment.py             # Difficulty-tier transforms
│   ├── ground_truth.py        # JSON emitter
│   ├── pipeline.py            # End-to-end orchestrator
│   └── __main__.py            # CLI
├── examples/                  # Small committed sample set (~30 MB)
├── tests/                     # 54 tests across all modules
└── docs/HCFA_OSS_WORKPLAN.md  # 4-week project plan
```

## Known limitations (v0)

- Box 32 facility NPI is unmapped (one of the numeric-named PDF fields).
- Handwriting tier deferred to v0.2.

See `docs/HCFA_OSS_WORKPLAN.md` for the broader 4-week project plan.

## License

MIT
