"""Score the *deployed* pipeline on the sanity samples, using PNG inputs.

Why PNG, not PDF: real-world claims arrive as scans/photos, and the per-tier
degradation (fax/phone/worn) lives in the rendered PNG — which is also what the
model trained on. Feeding the clean text PDFs would render crisp at 300 DPI and
make the per-tier numbers meaningless.

Flow:
  1. pick N samples per tier from data/sanity
  2. upload each PNG to S3 and call the live Modal worker (image path)
  3. read each result JSON back, build preds + split + GT batch dir
  4. score with hcfa_eval.scoring (the same harness the notebook used)

Run from the repo root:
    python serving/score_samples.py            # 4 per tier (all of sanity)
    python serving/score_samples.py 2          # 2 per tier
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv

load_dotenv(REPO / "frontdoor" / ".env")

import boto3
import modal

from hcfa_eval.scoring import format_summary, summarize, write_summary_csv

BUCKET = os.environ.get("S3_BUCKET") or os.environ["AWS_STORAGE_BUCKET_NAME"]
REGION = (
    os.environ.get("AWS_DEFAULT_REGION")
    or os.environ.get("AWS_S3_REGION_NAME")
    or "us-east-2"
)
PER_TIER = int(sys.argv[1]) if len(sys.argv) > 1 else 4
# Keep in-flight jobs well under the account's concurrent-GPU cap (default 10).
# A small window reuses warm containers instead of cold-starting many GPUs.
CONCURRENCY = int(os.environ.get("SCORE_CONCURRENCY", "3"))
MODEL_LABEL = "qwen2.5vl-3b-hcfa @png(deployed)"

s3 = boto3.client("s3", region_name=REGION)
Extractor = modal.Cls.from_name("hcfa-extractor", "Extractor")
SANITY = REPO / "data" / "sanity"


def select():
    rows = [json.loads(l) for l in (SANITY / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_tier = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
    picked = []
    for tier, trows in by_tier.items():
        picked.extend(trows[:PER_TIER])
    return picked


def main():
    selected = select()
    print(f"Scoring {len(selected)} samples ({PER_TIER}/tier) through the deployed worker...\n")

    ext = Extractor()

    # Upload + run with a bounded in-flight window (<= CONCURRENCY) so we stay
    # under the concurrent-GPU cap and reuse warm containers. spawn() queues the
    # job; we drain the oldest once the window is full.
    print(f"running {len(selected)} jobs, {CONCURRENCY} in flight at a time "
          f"(first cold-starts ~60-90s)...")
    calls, done = [], 0

    def _drain(c):
        nonlocal done
        done += 1
        try:
            c["call"].get()
            print(f"  [{done}/{len(selected)}] {c['sid']} ({c['tier']}) done")
        except Exception as e:  # noqa: BLE001
            c["error"] = str(e)
            print(f"  [{done}/{len(selected)}] {c['sid']} FAILED: {e}")

    inflight = []
    for r in selected:
        sid = r["sample_id"]
        png = SANITY / r["image"].replace("\\", "/")
        key = f"scoring/inputs/{sid}.png"
        s3.upload_file(str(png), BUCKET, key)
        out_prefix = f"scoring/results/{sid}"
        c = {"sid": sid, "tier": r["tier"], "out_prefix": out_prefix, "json": r["json"],
             "call": ext.extract.spawn(BUCKET, key, out_prefix)}
        calls.append(c)
        inflight.append(c)
        if len(inflight) >= CONCURRENCY:
            _drain(inflight.pop(0))
    for c in inflight:
        _drain(c)

    # 3) build preds.jsonl, test_split.jsonl, and GT batch dir
    work = Path(tempfile.mkdtemp(prefix="hcfa_score_"))
    batch = work / "gt"
    batch.mkdir()
    split_rows, pred_lines = [], []
    for c in calls:
        sid = c["sid"]
        gt = json.loads((SANITY / c["json"].replace("\\", "/")).read_text(encoding="utf-8"))
        (batch / f"{sid}.json").write_text(json.dumps({"logical": gt["logical"]}), encoding="utf-8")
        split_rows.append({"sample_id": sid, "tier": c["tier"], "json": f"{sid}.json"})

        fields, raw = {}, ""
        if "error" not in c:
            res = json.loads(s3.get_object(Bucket=BUCKET, Key=f"{c['out_prefix']}.json")["Body"].read())
            fields = res.get("fields", {})
            raw = res.get("raw", "")
        pred_lines.append(json.dumps({"sample_id": sid, "fields": fields, "raw": raw}))

    split_path = work / "test_split.jsonl"
    pred_path = work / "preds.jsonl"
    split_path.write_text("\n".join(json.dumps(r) for r in split_rows), encoding="utf-8")
    pred_path.write_text("\n".join(pred_lines), encoding="utf-8")

    # 4) score
    summary = summarize(split_path, pred_path, batch, model=MODEL_LABEL)
    print("\n" + format_summary(summary))

    out_csv = REPO / "serving" / "scoring_runs.csv"
    write_summary_csv(summary, out_csv)
    print(f"\nartifacts: {work}\nappended summary -> {out_csv}")


if __name__ == "__main__":
    main()
