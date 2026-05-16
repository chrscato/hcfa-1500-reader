# HCFA Open Source Extraction — 4-Week MVP Workplan

**Goal:** Ship a credible open-source CMS-1500 (HCFA) extraction pipeline with a public benchmark, live demo, and launch post — in 4 weeks, ~100 hours total. Bias to fast-and-quick over polished. Iterate in public after launch.

**Stack:** Python · Anthropic API (Claude vision) for v1 extraction · Pillow/OpenCV · WeasyPrint or ReportLab · Faker · FastAPI/Gradio · GitHub + HuggingFace Space.

---

## At-a-glance timeline

| Week | Phase | Primary goal | Headline deliverable |
|---|---|---|---|
| Week 0 (weekend) | Prep | Domain fluency + repo live | Empty repo, API smoke-tested |
| Week 1 | Synthetic data | Unlimited PHI-free training/test data | 500 (image, JSON) pairs + public `hcfa-synth` repo |
| Week 2 | Extraction MVP | End-to-end pipeline with measurable accuracy | Working extractor + first benchmark numbers |
| Week 3 | Robustness + validation | Survives real-world scans + domain rules enforced | Augmented data, validators, AWS Textract comparison |
| Week 4 | Launch | Public release with demo and writeup | HF Space + blog post + Show HN |

---

## Week 0 — Weekend prep (~6 hours)

| Slot | Task | What you do | Deliverable | Hrs | Claude Code helps? |
|---|---|---|---|---|---|
| Sat AM | Read CMS-1500 spec | Download form-instructions PDF from cms.gov; study all 33 boxes | Notes/cheatsheet on box meanings | 2 | Low (you read it) |
| Sat PM | Format quick-reference | Look up NPI Luhn check, ICD-10 format (letter + 2-7 alphanum), CPT (5 digits), 837P structure overview | One-pager with format rules | 1 | Medium |
| Sun AM | Repo + environment | Create GitHub repo, Python venv, install pillow opencv-python anthropic faker weasyprint reportlab fastapi gradio | Empty repo with `README.md` skeleton, deps locked in `pyproject.toml` | 1 | High |
| Sun PM | API smoke test | Get sample blank CMS-1500 from cms.gov, hand-fill one digital copy, send to Claude API with "extract all fields as JSON" | Working API call, gut-feel calibration | 2 | High |

**Exit criteria:** repo exists, you can extract fields from one form via API, you understand what the 33 boxes mean.

---

## Week 1 — Synthetic data foundation (~25 hours)

> **The unlock.** Real claim data is PHI; you can't ship it. A solid synthetic generator is itself a notable open-source contribution and the prerequisite for everything downstream. Spend the time here.

| Day | Task | What you do | Deliverable | Hrs | Claude Code helps? |
|---|---|---|---|---|---|
| Mon | Form template render | HTML + CSS approximating CMS-1500 layout, render to PNG via WeasyPrint | `render_form.py` produces blank template | 4 | High |
| Tue | Data injection | Faker for patient/provider demographics; load sample NPIs, ICD-10s, CPTs from public CSVs | `populate_form.py` injects realistic data | 4 | High |
| Wed | Ground-truth JSON | Every render emits matching JSON of all 33 fields | Paired `(image.png, ground_truth.json)` output | 3 | High |
| Thu | Box 24 service lines | The 6-row × 12-column service grid with codes, modifiers, dates, charges | Box 24 rendering with valid CPT/modifier combos | 4 | High |
| Fri | Generate dataset | Crank out 500 paired samples | Full v0 synthetic dataset | 2 | Medium |
| Sat | Visual QA | Eyeball 50 random samples — do they pass the squint test? Tune layout. | Refined templates | 4 | Medium |
| Sun | Publish `hcfa-synth` | Split out as standalone package, write its own README, push public | Public repo #1: `hcfa-synth` | 4 | High |

**Exit criteria:** running `python -m hcfa_synth generate --count 500 --output ./data/` produces 500 form images + JSON ground truth in under 5 minutes. The package is independently usable by anyone who wants to benchmark CMS-1500 extraction.

---

## Week 2 — Extraction MVP (~25 hours)

> **Build the test harness FIRST.** Every prompt change is worthless if you can't measure it. The benchmark loop is your most valuable artifact in the entire project.

| Day | Task | What you do | Deliverable | Hrs | Claude Code helps? |
|---|---|---|---|---|---|
| Mon | Eval harness | Script that takes (predicted_json, ground_truth_json) → per-field F1 + overall accuracy table | `benchmark.py` printing per-field scores | 3 | High |
| Tue | Single-prompt baseline | One Claude vision call: "extract all 33 fields." Run on 500 forms. Measure. | `extract_baseline.py` + first numbers | 3 | High |
| Wed | Zone decomposition | Split into 5 prompts: header, patient block, insurance, services (Box 24), provider footer | `extract_zoned.py` with parallel API calls | 4 | High |
| Thu | Box 24 specialist | Dedicated prompt + structured output schema for the service-line grid | Improved Box 24 accuracy | 4 | High |
| Fri | Field-level tuning | For the 5 lowest-accuracy fields, write specialized prompts (handwriting hint, format hint, etc.) | Updated benchmark | 4 | Medium |
| Sat | Pipeline wiring | `image_in → deskew → zoned_extraction → JSON_merge → JSON_out` end-to-end | `pipeline.py` callable from CLI | 4 | High |
| Sun | First public number | Update README with current accuracy table, note weak fields honestly | Benchmark table v1 | 3 | Medium |

**Exit criteria:** `python -m hcfa.pipeline form.png` outputs valid JSON. Overall field-level accuracy ≥ 85% on clean synthetic forms. Per-field accuracy table is in the README.

---

## Week 3 — Robustness + validation (~25 hours)

| Day | Task | What you do | Deliverable | Hrs | Claude Code helps? |
|---|---|---|---|---|---|
| Mon | Scan artifact augmentation | Add to synth generator: skew (-5° to +5°), gaussian noise, JPEG compression, fade, fax streaks, staple holes | "Dirty" dataset variant | 4 | High |
| Tue | Re-benchmark dirty | Measure accuracy drop on augmented vs clean | Comparison table | 2 | Medium |
| Wed | Format validators | Implement: NPI Luhn check, ICD-10 regex, CPT 5-digit check, place-of-service codeset | `validators.py` module | 3 | High |
| Thu | Cross-field validators | DOB ≤ service date ≤ today, ZIP/state consistency, charge math | Extended validators | 3 | High |
| Fri | Confidence scoring | Combine: VLM logprob proxy + validator passes + cross-field consistency → per-field 0-1 score | `confidence.py` | 4 | High |
| Sat | Routing logic | High-confidence → JSON output; low-confidence → review queue stub (just JSON for v1) | `routing.py` complete | 3 | High |
| Sun | AWS Textract benchmark | Run same 500 augmented forms through AWS Textract Forms API, score with same harness | Apples-to-apples comparison table | 6 | Medium (AWS setup is fiddly) |

**Exit criteria:** Pipeline produces JSON with per-field confidence scores. README has accuracy table comparing clean vs dirty data, and a Textract head-to-head comparison. You have at least one field where you match or beat Textract — that's your launch hook.

---

## Week 4 — Launch (~25 hours)

| Day | Task | What you do | Deliverable | Hrs | Claude Code helps? |
|---|---|---|---|---|---|
| Mon | 837P EDI export | Add EDI X12 837P serializer alongside JSON output (give CC the spec PDF) | `edi_export.py` produces valid 837P | 5 | High |
| Tue | HuggingFace Space | Gradio UI: upload form → see JSON + confidence highlights → deploy to HF Space (free CPU tier) | Public live demo URL | 5 | High |
| Wed | Demo video | 90-second screen recording: drop synthetic form in, watch JSON appear with confidence overlay | `demo.mp4` (or YouTube link) | 3 | Low |
| Thu | README polish | Architecture diagram, benchmark tables, install instructions, "what this does NOT do" section | Launch-ready README | 4 | High |
| Fri | Blog post | Technical writeup: "I spent 4 weeks beating AWS Textract on medical claim forms with open-source models" | Published personal blog post | 4 | High |
| Sat | Launch | Show HN (link to blog post, repo, demo), r/MachineLearning, r/healthIT, targeted Twitter DMs to ~10 healthtech-AI folks | Launched | 2 | Medium |
| Sun | Triage + respond | Reply to comments and issues, capture feedback for v0.2 backlog | `BACKLOG.md` + closed loops | 2 | High |

**Exit criteria:** Repo is public, README is honest and benchmark-led, demo works end-to-end on HF Space, blog post is live, Show HN is submitted.

---

## Investment summary

| Resource | Estimate |
|---|---|
| Calendar time | 4 weeks |
| Total hours | ~106 hours (~25/week) |
| Claude API spend (extraction during dev) | $30-80 |
| AWS Textract benchmark spend | $5-15 |
| GPU rental for OSS model swap-in (one-time) | $5-15 |
| HuggingFace Space (free CPU tier sufficient for v1) | $0 |
| Domain + miscellaneous | $0-20 |
| **Total cash outlay** | **~$100-200** |

---

## Tools & resources reference

| Category | Item | Purpose | Cost |
|---|---|---|---|
| Spec | CMS-1500 form instructions (cms.gov) | Field semantics | Free |
| Spec | 837P Implementation Guide | EDI serialization | Free |
| Data | NPPES NPI Registry download | Sample valid NPIs | Free |
| Data | ICD-10-CM code set (CMS) | Sample valid diagnoses | Free |
| Data | HCPCS/CPT code lists | Sample valid procedures | Free (CPT has license caveats — use HCPCS for OSS) |
| VLM | Anthropic API (Claude Sonnet 4.6 vision) | Primary extraction during dev | ~$0.01-0.05/form |
| VLM swap | Qwen2.5-VL 7B (HuggingFace) | Final OSS benchmark | Free model, GPU rental |
| GPU | RunPod / vast.ai | One-day OSS bench run | ~$0.40-1.00/hr |
| Image | Pillow + OpenCV | Preprocessing | Free |
| Render | WeasyPrint or ReportLab | Synthetic form generation | Free |
| Faker | `faker` Python library | Demographics | Free |
| API | FastAPI | Optional service mode | Free |
| Demo | Gradio + HuggingFace Spaces | Live demo | Free CPU tier |
| Comparison | AWS Textract | Benchmark target | ~$1.50/1000 pages |
| Comparison | Google Document AI (optional) | Second benchmark target | ~$0.60/page |
| IDE | Claude Code | Force multiplier | API usage |

---

## Critical path — don't skip these

| Priority | Item | Why |
|---|---|---|
| **P0** | Synthetic generator before extractor | PHI blocks real data; nothing else works without this |
| **P0** | Eval harness before prompt tuning | Untested prompt changes are vibes, not progress |
| **P0** | Honest benchmark numbers in README | This is the entire credibility play |
| **P0** | At least one real OCR comparison (Textract) | Without it, no one believes the numbers |
| **P1** | Scan-artifact augmentation | Lab-only results don't impress this audience |
| **P1** | Box 24 (service lines) extraction | This is what proprietary OCR struggles with too |
| **P1** | Validation layer | Demonstrates you understand the domain |
| **P2** | 837P EDI export | Nice signal of seriousness; cheap with CC + spec |
| **P2** | Live HF Space demo | High launch impact, low effort |
| **P3** | Human-in-the-loop UI | Defer to v0.2 — write a stub |
| **P3** | Self-hosted OSS VLM in production | Defer; do final-bench swap only |

---

## Watch-outs (where projects like this stall)

| Trap | Why it kills you | Counter-move |
|---|---|---|
| Building a plugin/config framework | Premature abstraction, weeks lost | Hardcode everything; refactor only when forced |
| Self-hosting OSS VLM in week 1 | Infra rabbit hole, contributes nothing to launch | Use API; one-day swap at the end |
| Fine-tuning a model | Adds weeks for marginal gain at this scale | Prompt engineering is enough for MVP |
| Touching real claim data | PHI = career risk, also illegal in most contexts | Synthetic only, no exceptions |
| Polishing before benchmarking | You don't know what to polish | Benchmark first, polish what scores worst |
| Expanding scope to UB-04 / dental | Loses the focused launch story | CMS-1500 only for v1; expand post-launch |
| Waiting for "ready" before launching | The launch generates the feedback that defines v0.2 | Ship at minimum bar; iterate in public |
| Skipping the demo video | README without video gets 10× fewer engagement | 90-second screen recording is non-negotiable |

---

## Claude Code working patterns for this project

| Pattern | When to use | Why |
|---|---|---|
| Pin `CLAUDE.md` at repo root | From day 1 | Captures architecture decisions and domain context for every CC invocation |
| Plan mode | Any task touching form layout or EDI structure | Spatial/format errors are subtle and hard to debug |
| Direct mode | Boilerplate (Faker hookups, FastAPI routes, README sections) | Fast and reliable on conventional code |
| Provide spec PDFs as context | EDI 837P serializer, ICD-10 validators | Dense format work CC handles well with the source |
| Push back on abstractions | Whenever CC suggests config systems or plugins | Goal is 2k LOC repo, not 20k |
| Have CC write the test harness BEFORE the feature | Especially the eval harness (Week 2 Mon) | Untested code in this domain is a liability |

---

## After launch — v0.2 backlog seeds

These are deferred from MVP. Don't build them in the 4-week window.

| Idea | Triggers it | Effort |
|---|---|---|
| Self-hosted OSS VLM serving (vLLM + Qwen) | Privacy-conscious user requests | 1-2 weeks |
| Active-learning loop on review queue corrections | First real users producing labeled data | 2-3 weeks |
| UB-04 (institutional claims) extension | Demand from users who do facility billing | 2-4 weeks |
| Production-grade human review UI | Someone wants to actually deploy this | 2-3 weeks |
| Fine-tuned Qwen on synthetic data | Beating frontier APIs on accuracy | 2-4 weeks + GPU |
| HIPAA / SOC2 documentation pack | First serious commercial inquiry | 4+ weeks |

---

*Build fast. Ship at "embarrassed-but-honest" quality. Let the launch tell you what to build next.*
