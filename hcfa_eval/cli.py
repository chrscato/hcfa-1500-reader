"""CLI: split, preview, score.

Examples:
  hcfa-eval split   --batch data/full
  hcfa-eval preview --batch data/full --split splits/full/train.jsonl --n 1
  hcfa-eval score   --batch data/full --split splits/full/test.jsonl --preds preds.jsonl
  hcfa-eval keys    --batch data/full > schema_keys.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hcfa_eval.dataset import build_row, derive_schema_keys
from hcfa_eval.scoring import (
    format_report,
    format_summary,
    score,
    summarize,
    write_summary_csv,
)
from hcfa_eval.splits import split_manifest


def _cmd_split(args: argparse.Namespace) -> int:
    batch = Path(args.batch)
    manifest = batch / "manifest.jsonl"
    if not manifest.exists():
        print(f"manifest not found: {manifest}")
        return 1
    out = Path(args.out) if args.out else Path("splits") / batch.name
    paths, splits = split_manifest(manifest, out, seed=args.seed)
    for name, p in paths.items():
        print(f"  {name}: {len(splits[name])} rows -> {p}")
    return 0


def _cmd_preview(args: argparse.Namespace) -> int:
    batch = Path(args.batch)
    split_path = Path(args.split)
    rows = [json.loads(l) for l in split_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    schema_keys = derive_schema_keys(batch) if args.prompt == "schema" else None
    for r in rows[: args.n]:
        chat = build_row(r, batch, prompt_style=args.prompt, schema_keys=schema_keys)
        print(json.dumps(
            {
                "sample_id": chat["sample_id"],
                "tier": chat["tier"],
                "image_path": chat["image_path"],
                "system": chat["messages"][0]["content"][0]["text"][:80] + "...",
                "user_text_preview": chat["messages"][1]["content"][1]["text"][:200],
                "target_json_preview": chat["target_json"][:200] + "...",
                "target_json_len": len(chat["target_json"]),
            },
            indent=2,
        ))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    rep = score(Path(args.split), Path(args.preds), Path(args.batch))
    print(format_report(rep, top_field_misses=args.top_misses))
    if args.json_out:
        # Convert non-serializable dataclasses to dicts for downstream use
        def _fs_to_dict(fs):
            return {
                "exact_hits": fs.exact_hits,
                "norm_hits": fs.norm_hits,
                "total": fs.total,
                "exact_acc": fs.exact_acc,
                "norm_acc": fs.norm_acc,
                "top_misses": [
                    {"gt": g, "pred": p, "count": c}
                    for (g, p), c in fs.misses.most_common(10)
                ],
            }
        payload = {
            "n_samples": rep["n_samples"],
            "missing_predictions": rep["missing_predictions"],
            "by_tier_field": {
                t: {k: _fs_to_dict(v) for k, v in fs.items()}
                for t, fs in rep["by_tier_field"].items()
            },
            "overall_field": {k: _fs_to_dict(v) for k, v in rep["overall_field"].items()},
            "by_tier_doc": {
                t: {
                    "total": ds.total,
                    "all_correct": ds.all_correct,
                    "mean_field_acc": sum(ds.per_doc_field_acc) / ds.total if ds.total else 0.0,
                }
                for t, ds in rep["by_tier_doc"].items()
            },
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {args.json_out}")

    # High-level summary (populated/blank, per-tier, per-class, CER, validity)
    summary = summarize(
        Path(args.split), Path(args.preds), Path(args.batch), model=args.model
    )
    print("\n" + format_summary(summary))
    if args.summary_csv:
        write_summary_csv(summary, Path(args.summary_csv))
        print(f"\nAppended summary row to {args.summary_csv}")
    return 0


def _cmd_keys(args: argparse.Namespace) -> int:
    keys = derive_schema_keys(Path(args.batch), sample_count=args.sample_count)
    for k in keys:
        print(k)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hcfa-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("split", help="stratified train/val/test split")
    sp.add_argument("--batch", required=True)
    sp.add_argument("--out", default=None)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=_cmd_split)

    pp = sub.add_parser("preview", help="print N chat-format rows")
    pp.add_argument("--batch", required=True)
    pp.add_argument("--split", required=True)
    pp.add_argument("--n", type=int, default=1)
    pp.add_argument("--prompt", choices=["minimal", "schema"], default="minimal")
    pp.set_defaults(fn=_cmd_preview)

    sc = sub.add_parser("score", help="score predictions.jsonl against a split")
    sc.add_argument("--batch", required=True)
    sc.add_argument("--split", required=True)
    sc.add_argument("--preds", required=True)
    sc.add_argument("--top-misses", type=int, default=15)
    sc.add_argument("--json-out", default=None)
    sc.add_argument("--model", default="", help="model name to label the summary / CSV row")
    sc.add_argument("--summary-csv", default=None, help="append a one-row summary to this CSV")
    sc.set_defaults(fn=_cmd_score)

    kk = sub.add_parser("keys", help="print canonical flat schema keys")
    kk.add_argument("--batch", required=True)
    kk.add_argument("--sample-count", type=int, default=50)
    kk.set_defaults(fn=_cmd_keys)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
