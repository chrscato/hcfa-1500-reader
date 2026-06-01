"""Score predictions vs. ground truth.

Inputs:
  - predictions: list of {sample_id, fields: flat_dict} (model output)
  - split rows : list of manifest rows with `tier` and json path

For each sample we score every key in the UNION of pred + gt:
  - missing key on either side counts as ""
  - both exact and normalized matches are recorded

Aggregations:
  - per-tier  : mean per-field exact + normalized accuracy
  - per-field : mean per-tier exact + normalized accuracy
  - doc-level : fraction of samples with ALL normalized fields correct
  - confusion : for each field, top-5 (gt → pred) miss pairs

A "field" here is a flat dot/index path. Service-line array indices are
treated as their own fields (e.g., `box_24_service_lines[2].procedure_code`).
Samples that don't reach that index get "" on both sides → auto-match.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hcfa_eval.normalize import (
    compare,
    is_date_field,
    is_money_field,
    normalize,
)
from hcfa_eval.schema import flatten


@dataclass
class FieldStats:
    exact_hits: int = 0
    norm_hits: int = 0
    total: int = 0
    misses: Counter = field(default_factory=Counter)  # (gt_norm, pred_norm) -> count

    def add(self, exact: bool, norm: bool, gt: str, pred: str) -> None:
        self.total += 1
        if exact:
            self.exact_hits += 1
        if norm:
            self.norm_hits += 1
        else:
            self.misses[(gt, pred)] += 1

    @property
    def exact_acc(self) -> float:
        return self.exact_hits / self.total if self.total else 0.0

    @property
    def norm_acc(self) -> float:
        return self.norm_hits / self.total if self.total else 0.0


@dataclass
class DocStats:
    total: int = 0
    all_correct: int = 0
    per_doc_field_acc: List[float] = field(default_factory=list)


def _load_gt_flat(batch_dir: Path, json_rel: str) -> Dict[str, str]:
    raw = json.loads((batch_dir / json_rel).read_text(encoding="utf-8"))
    return flatten(raw["logical"])


def _load_predictions(pred_path: Path) -> Dict[str, Dict[str, str]]:
    """Read predictions.jsonl: each line {sample_id, fields: {...}}."""
    out: Dict[str, Dict[str, str]] = {}
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["sample_id"]] = {k: ("" if v is None else str(v)) for k, v in row["fields"].items()}
    return out


def _load_split(split_path: Path) -> List[dict]:
    return [json.loads(l) for l in split_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score(
    split_path: Path,
    pred_path: Path,
    batch_dir: Path,
) -> dict:
    """Return a nested report dict. Pure data — render with `report.format_report`."""
    split = _load_split(split_path)
    preds = _load_predictions(pred_path)

    by_tier_field: Dict[str, Dict[str, FieldStats]] = defaultdict(lambda: defaultdict(FieldStats))
    overall_field: Dict[str, FieldStats] = defaultdict(FieldStats)
    by_tier_doc: Dict[str, DocStats] = defaultdict(DocStats)
    overall_doc = DocStats()
    missing_preds: List[str] = []

    for row in split:
        sid = row["sample_id"]
        tier = row["tier"]
        gt_flat = _load_gt_flat(batch_dir, row["json"])
        pred_flat = preds.get(sid)
        if pred_flat is None:
            missing_preds.append(sid)
            pred_flat = {}

        keys = set(gt_flat) | set(pred_flat)
        doc_hits = 0
        for k in keys:
            gt_v = gt_flat.get(k, "")
            pred_v = pred_flat.get(k, "")
            exact, norm = compare(k, pred_v, gt_v)
            from hcfa_eval.normalize import normalize as _n
            by_tier_field[tier][k].add(exact, norm, _n(k, gt_v), _n(k, pred_v))
            overall_field[k].add(exact, norm, _n(k, gt_v), _n(k, pred_v))
            if norm:
                doc_hits += 1
        doc_acc = doc_hits / len(keys) if keys else 1.0
        by_tier_doc[tier].total += 1
        by_tier_doc[tier].per_doc_field_acc.append(doc_acc)
        overall_doc.total += 1
        overall_doc.per_doc_field_acc.append(doc_acc)
        if doc_hits == len(keys):
            by_tier_doc[tier].all_correct += 1
            overall_doc.all_correct += 1

    return {
        "by_tier_field": by_tier_field,
        "overall_field": overall_field,
        "by_tier_doc": by_tier_doc,
        "overall_doc": overall_doc,
        "missing_predictions": missing_preds,
        "n_samples": len(split),
    }


def format_report(report: dict, *, top_field_misses: int = 15) -> str:
    """Render a human-readable text report from a score() result."""
    lines: List[str] = []
    n = report["n_samples"]
    miss = len(report["missing_predictions"])
    lines.append(f"Scored {n} samples ({miss} missing predictions)")
    lines.append("")

    # ----- per-tier summary
    lines.append("=== Per-tier accuracy ===")
    lines.append(f"{'tier':<14}{'docs':>6}{'all_correct':>14}{'mean_field_acc':>18}{'norm_field_acc':>18}")
    for tier in sorted(report["by_tier_doc"].keys()):
        ds = report["by_tier_doc"][tier]
        all_corr = ds.all_correct / ds.total if ds.total else 0.0
        mean_acc = sum(ds.per_doc_field_acc) / ds.total if ds.total else 0.0
        # mean normalized field acc across all (tier, field) cells
        cells = report["by_tier_field"][tier].values()
        norm_acc = sum(c.norm_acc for c in cells) / len(cells) if cells else 0.0
        lines.append(f"{tier:<14}{ds.total:>6}{all_corr:>14.3f}{mean_acc:>18.3f}{norm_acc:>18.3f}")
    od = report["overall_doc"]
    overall_all = od.all_correct / od.total if od.total else 0.0
    overall_mean = sum(od.per_doc_field_acc) / od.total if od.total else 0.0
    lines.append(f"{'OVERALL':<14}{od.total:>6}{overall_all:>14.3f}{overall_mean:>18.3f}")
    lines.append("")

    # ----- worst fields overall
    lines.append(f"=== Worst {top_field_misses} fields (by normalized accuracy) ===")
    lines.append(f"{'field':<48}{'n':>6}{'exact':>9}{'norm':>9}")
    ranked = sorted(
        report["overall_field"].items(),
        key=lambda kv: (kv[1].norm_acc, -kv[1].total),
    )
    for name, s in ranked[:top_field_misses]:
        lines.append(f"{name[:48]:<48}{s.total:>6}{s.exact_acc:>9.3f}{s.norm_acc:>9.3f}")
    lines.append("")

    # ----- top confusions for the bottom few fields
    lines.append("=== Top miss pairs (worst fields only) ===")
    for name, s in ranked[:5]:
        if not s.misses:
            continue
        lines.append(f"-- {name}")
        for (gt, pred), c in s.misses.most_common(5):
            gt_d = gt or "<blank>"
            pred_d = pred or "<blank>"
            lines.append(f"     {c:>4}x  gt={gt_d!r:<28}  pred={pred_d!r}")

    return "\n".join(lines)


# ===========================================================================
# Summary metrics
# ---------------------------------------------------------------------------
# Everything above scores per flat key. The functions below roll those cells
# up into the higher-level numbers a model card / run-comparison CSV wants:
# populated vs blank accuracy, per-tier accuracy, per-field-class exact match,
# character error rate on the structured classes, and JSON validity. The
# original `score()` is untouched — this is purely additive.
# ===========================================================================

# Difficulty order (mirrors hcfa_synth.augment.TIER_NAMES). Kept local so the
# eval package stays importable without the synth package. Unknown tiers seen
# in a split are appended after these, in first-seen order.
DEFAULT_TIER_ORDER: List[str] = [
    "pristine",
    "clean_scan",
    "worn_scan",
    "fax",
    "phone_photo",
    "worst",
]


# ----- field-class assignment ------------------------------------------------
# Classes are described in the task by PDF field name (cpt*/ch*/local*/pin/…);
# here we map them onto the flattened *logical* keys the scorer actually sees.

def _is_code_field(key: str) -> bool:
    """CPT/HCPCS procedure codes and ICD diagnosis codes (PDF cpt*/diagnosis*)."""
    k = key.lower()
    return k.endswith("procedure_code") or k.endswith("diagnoses")


def _is_npi_field(key: str) -> bool:
    """NPIs — rendering (PDF local*), billing (pin), facility, referring."""
    k = key.lower()
    return k.endswith(".npi") or k.endswith("_npi")


def _is_name_field(key: str) -> bool:
    k = key.lower()
    return (
        k.endswith(".first")
        or k.endswith(".middle")
        or k.endswith(".last")
        or k.endswith(".name")
        or k.endswith("_name")
    )


def _is_address_field(key: str) -> bool:
    k = key.lower()
    if k.endswith(".phone"):  # an address may carry a phone — that's not address text
        return False
    return "address" in k or k.endswith("city_state_zip")


# Order matters: first match wins, so each key lands in at most one class.
_CLASS_PREDICATES: List[Tuple[str, "callable"]] = [
    ("dates", is_date_field),
    ("money", is_money_field),
    ("npis", _is_npi_field),
    ("codes", _is_code_field),
    ("names", _is_name_field),
    ("addresses", _is_address_field),
]

FIELD_CLASSES: List[str] = [name for name, _ in _CLASS_PREDICATES]

# Structured string classes where a soft character-level score is meaningful.
CER_CLASSES: List[str] = ["codes", "money", "npis"]


def classify_field(key: str) -> Optional[str]:
    """Return the field-class name for `key`, or None if it belongs to none."""
    for name, pred in _CLASS_PREDICATES:
        if pred(key):
            return name
    return None


# ----- character error rate --------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Edit distance (insertions/deletions/substitutions)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def char_error_rate(pred: str, gt: str) -> float:
    """Normalized edit distance = edits / len(reference). 0.0 == identical.

    Capped at 1.0. With an empty reference, returns 0.0 if pred is also empty
    else 1.0 (any output against an empty target is fully wrong).
    """
    if not gt:
        return 0.0 if not pred else 1.0
    return min(1.0, _levenshtein(pred, gt) / len(gt))


# ----- accumulators ----------------------------------------------------------

@dataclass
class _AccPair:
    """Running exact/normalized accuracy over a set of cells."""

    exact_hits: int = 0
    norm_hits: int = 0
    total: int = 0

    def add(self, exact: bool, norm: bool) -> None:
        self.total += 1
        self.exact_hits += int(exact)
        self.norm_hits += int(norm)

    def as_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "exact_acc": self.exact_hits / self.total if self.total else 0.0,
            "norm_acc": self.norm_hits / self.total if self.total else 0.0,
        }


@dataclass
class _ClassAcc:
    """Per-class accuracy, tracked over populated GT cells (the informative
    set — blank/unused service lines would otherwise inflate exact match)."""

    total: int = 0          # all cells in this class (pred ∪ gt keys)
    populated: int = 0      # cells whose GT normalizes non-blank
    exact_hits: int = 0     # exact hits among populated
    norm_hits: int = 0      # normalized hits among populated

    def as_dict(self) -> Dict[str, float]:
        p = self.populated
        return {
            "total": self.total,
            "populated": p,
            "exact_acc": self.exact_hits / p if p else 0.0,
            "norm_acc": self.norm_hits / p if p else 0.0,
        }


# ----- prediction loading for JSON validity ----------------------------------

def _load_pred_validity(pred_path: Path) -> Dict[str, dict]:
    """Map sample_id → {parse_ok, fields} for JSON-validity checks.

    Predictions are one JSON object per line: {sample_id, fields:{...}}. If a
    line also carries the model's original text under "raw", validity parses
    THAT (so we measure the model's real JSON), else the line's own validity
    stands in. Unparseable / sample_id-less lines are skipped (they surface as
    parse failures via the missing-sample denominator).
    """
    out: Dict[str, dict] = {}
    if not pred_path.exists():
        return out
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or "sample_id" not in obj:
            continue
        if "raw" in obj:
            try:
                raw_obj = json.loads(obj["raw"])
                parse_ok = isinstance(raw_obj, dict)
                fields = raw_obj if parse_ok else {}
            except (json.JSONDecodeError, ValueError, TypeError):
                parse_ok, fields = False, {}
        else:
            fields = obj.get("fields", {})
            parse_ok = isinstance(fields, dict)
        out[obj["sample_id"]] = {
            "parse_ok": parse_ok,
            "fields": fields if isinstance(fields, dict) else {},
        }
    return out


# ----- the summary -----------------------------------------------------------

def summarize(
    split_path: Path,
    pred_path: Path,
    batch_dir: Path,
    *,
    model: str = "",
    tier_order: Optional[List[str]] = None,
) -> dict:
    """Roll per-cell results up into the summary dict (see module header).

    Returns a single nested dict; `summary_csv_row` flattens it to one CSV row.
    Accuracy is normalized-match unless a metric says "exact". Reuses the same
    GT flattening and `compare()` as `score()`, so the two never disagree.
    """
    split = _load_split(split_path)
    preds = _load_predictions(pred_path)
    pred_validity = _load_pred_validity(pred_path)

    populated = _AccPair()
    blank = _AccPair()
    by_tier_pop: Dict[str, _AccPair] = defaultdict(_AccPair)
    by_class: Dict[str, _ClassAcc] = {c: _ClassAcc() for c in FIELD_CLASSES}
    cer_sums: Dict[str, float] = {c: 0.0 for c in CER_CLASSES}
    cer_counts: Dict[str, int] = {c: 0 for c in CER_CLASSES}

    n_valid = n_parse_ok = n_keys_ok = 0
    coverage_sum = 0.0
    tiers_seen: List[str] = []

    for row in split:
        sid = row["sample_id"]
        tier = row["tier"]
        if tier not in tiers_seen:
            tiers_seen.append(tier)
        gt_flat = _load_gt_flat(batch_dir, row["json"])
        pred_flat = preds.get(sid, {})

        for key in set(gt_flat) | set(pred_flat):
            gt_v = gt_flat.get(key, "")
            pred_v = pred_flat.get(key, "")
            exact, norm = compare(key, pred_v, gt_v)
            gt_is_blank = normalize(key, gt_v) == ""

            if gt_is_blank:
                blank.add(exact, norm)
            else:
                populated.add(exact, norm)
                by_tier_pop[tier].add(exact, norm)

            cls = classify_field(key)
            if cls is not None:
                acc = by_class[cls]
                acc.total += 1
                if not gt_is_blank:
                    acc.populated += 1
                    acc.exact_hits += int(exact)
                    acc.norm_hits += int(norm)
                    if cls in cer_sums:
                        cer_sums[cls] += char_error_rate(
                            normalize(key, pred_v), normalize(key, gt_v)
                        )
                        cer_counts[cls] += 1

        # ----- JSON validity (per split sample) -----
        expected_keys = set(gt_flat)
        info = pred_validity.get(sid)
        if info is None:
            coverage_sum += 0.0
            continue
        if info["parse_ok"]:
            n_parse_ok += 1
        pred_keys = set(info["fields"])
        covered = expected_keys & pred_keys
        coverage = len(covered) / len(expected_keys) if expected_keys else 1.0
        coverage_sum += coverage
        has_keys = expected_keys.issubset(pred_keys)
        if has_keys:
            n_keys_ok += 1
        if info["parse_ok"] and has_keys:
            n_valid += 1

    n = len(split)
    ordered_tiers = list(tier_order or DEFAULT_TIER_ORDER)
    for t in tiers_seen:  # append any unexpected tiers, stable order
        if t not in ordered_tiers:
            ordered_tiers.append(t)

    cer = {
        c: {
            "populated": cer_counts[c],
            "mean_cer": cer_sums[c] / cer_counts[c] if cer_counts[c] else 0.0,
        }
        for c in CER_CLASSES
    }
    total_cer_n = sum(cer_counts.values())
    cer["overall"] = {
        "populated": total_cer_n,
        "mean_cer": sum(cer_sums.values()) / total_cer_n if total_cer_n else 0.0,
    }

    return {
        "model": model,
        "n_samples": n,
        "missing_predictions": [r["sample_id"] for r in split if r["sample_id"] not in preds],
        "overall": {
            "populated": populated.as_dict(),
            "blank": blank.as_dict(),
        },
        "by_tier_populated": {
            t: by_tier_pop[t].as_dict() for t in ordered_tiers if t in by_tier_pop
        },
        "by_class": {c: by_class[c].as_dict() for c in FIELD_CLASSES},
        "cer": cer,
        "json_validity": {
            "n_samples": n,
            "parse_rate": n_parse_ok / n if n else 0.0,
            "has_expected_keys_rate": n_keys_ok / n if n else 0.0,
            "valid_rate": n_valid / n if n else 0.0,
            "mean_key_coverage": coverage_sum / n if n else 0.0,
        },
        "tier_order": [t for t in ordered_tiers if t in by_tier_pop],
    }


def summary_csv_row(summary: dict) -> Dict[str, object]:
    """Flatten a `summarize()` result into one ordered CSV row (model, tier
    scores, class scores, CER, JSON validity). Tier columns use the canonical
    order so headers are stable across runs/models."""
    row: Dict[str, object] = {
        "model": summary.get("model", ""),
        "n_samples": summary["n_samples"],
        "populated_norm_acc": round(summary["overall"]["populated"]["norm_acc"], 4),
        "blank_norm_acc": round(summary["overall"]["blank"]["norm_acc"], 4),
    }
    by_tier = summary["by_tier_populated"]
    for tier in DEFAULT_TIER_ORDER:
        cell = by_tier.get(tier)
        row[f"tier_{tier}_norm_acc"] = round(cell["norm_acc"], 4) if cell else ""
    for cls in FIELD_CLASSES:
        row[f"class_{cls}_exact"] = round(summary["by_class"][cls]["exact_acc"], 4)
    for cls in CER_CLASSES:
        row[f"cer_{cls}"] = round(summary["cer"][cls]["mean_cer"], 4)
    row["cer_overall"] = round(summary["cer"]["overall"]["mean_cer"], 4)
    row["json_valid_rate"] = round(summary["json_validity"]["valid_rate"], 4)
    row["json_parse_rate"] = round(summary["json_validity"]["parse_rate"], 4)
    return row


def write_summary_csv(summary: dict, csv_path: Path, *, append: bool = True) -> None:
    """Write/append one summary row to a CSV, adding the header if new.

    Appending lets you accumulate a model-comparison table across runs.
    """
    csv_path = Path(csv_path)
    row = summary_csv_row(summary)
    write_header = not (append and csv_path.exists() and csv_path.stat().st_size > 0)
    mode = "a" if append else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header or not append:
            writer.writeheader()
        writer.writerow(row)


def format_summary(summary: dict) -> str:
    """Human-readable rendering of a `summarize()` result."""
    lines: List[str] = []
    model = summary.get("model") or "(unnamed)"
    lines.append(f"=== Summary: {model} ({summary['n_samples']} samples) ===")
    pop = summary["overall"]["populated"]
    blk = summary["overall"]["blank"]
    lines.append(
        f"populated fields : norm={pop['norm_acc']:.3f}  exact={pop['exact_acc']:.3f}  (n={pop['total']})"
    )
    lines.append(
        f"blank fields     : norm={blk['norm_acc']:.3f}  exact={blk['exact_acc']:.3f}  (n={blk['total']})"
    )

    lines.append("")
    lines.append("--- Per-tier (populated fields, normalized) ---")
    lines.append(f"{'tier':<14}{'n':>7}{'norm_acc':>11}")
    for tier in summary["tier_order"]:
        c = summary["by_tier_populated"][tier]
        lines.append(f"{tier:<14}{c['total']:>7}{c['norm_acc']:>11.3f}")

    lines.append("")
    lines.append("--- Per field-class (populated only) ---")
    lines.append(f"{'class':<12}{'pop':>7}{'exact':>9}{'norm':>9}{'CER':>9}")
    for cls in FIELD_CLASSES:
        c = summary["by_class"][cls]
        cer = summary["cer"].get(cls, {}).get("mean_cer")
        cer_s = f"{cer:>9.3f}" if cer is not None else f"{'n/a':>9}"
        lines.append(f"{cls:<12}{c['populated']:>7}{c['exact_acc']:>9.3f}{c['norm_acc']:>9.3f}{cer_s}")
    lines.append(f"{'CER overall':<12}{summary['cer']['overall']['populated']:>7}{'':>18}{summary['cer']['overall']['mean_cer']:>9.3f}")

    lines.append("")
    jv = summary["json_validity"]
    lines.append("--- JSON validity ---")
    lines.append(
        f"parse_rate={jv['parse_rate']:.3f}  has_expected_keys={jv['has_expected_keys_rate']:.3f}  "
        f"valid_rate={jv['valid_rate']:.3f}  mean_key_coverage={jv['mean_key_coverage']:.3f}"
    )
    return "\n".join(lines)
