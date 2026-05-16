"""End-to-end synthetic-sample pipeline.

A "sample" is one image + one ground-truth JSON + (optional) the source PDF.
A "batch" is N samples distributed across one or more difficulty tiers.

Pipeline per sample:
  build_record(seed) → fill_pdf → render_png → apply_tier → write outputs
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image

from hcfa_synth.augment import TIER_NAMES, apply_tier
from hcfa_synth.ground_truth import build_ground_truth
from hcfa_synth.pdf_fill import fill_pdf
from hcfa_synth.records import build_record
from hcfa_synth.render import pdf_to_png


@dataclass
class SampleResult:
    sample_id: str
    tier: str
    seed: int
    image_path: Path
    json_path: Path
    pdf_path: Optional[Path]


def generate_sample(
    sample_id: str,
    tier: str,
    seed: int,
    out_dir: Path,
    *,
    dpi: int = 300,
    keep_pdf: bool = True,
) -> SampleResult:
    """Build one sample and write its artifacts to `out_dir/<tier>/`."""
    tier_dir = out_dir / tier
    tier_dir.mkdir(parents=True, exist_ok=True)

    record = build_record(seed=seed)
    pdf_bytes = fill_pdf(record)
    base_image = pdf_to_png(pdf_bytes, dpi=dpi)
    # Use the same seed for augmentation determinism — each (seed, tier) pair
    # produces a unique sample.
    aug_image = apply_tier(base_image, tier, seed=seed)

    image_path = tier_dir / f"{sample_id}.png"
    json_path = tier_dir / f"{sample_id}.json"
    pdf_path: Optional[Path] = tier_dir / f"{sample_id}.pdf" if keep_pdf else None

    aug_image.save(image_path, "PNG", optimize=True)

    gt = build_ground_truth(record)
    gt["sample"] = {
        "id": sample_id,
        "tier": tier,
        "seed": seed,
        "dpi": dpi,
        "image": image_path.name,
    }
    json_path.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    if pdf_path:
        pdf_path.write_bytes(pdf_bytes)

    return SampleResult(
        sample_id=sample_id,
        tier=tier,
        seed=seed,
        image_path=image_path,
        json_path=json_path,
        pdf_path=pdf_path,
    )


def generate_batch(
    count: int,
    tiers: Iterable[str],
    out_dir: Path,
    *,
    seed_base: int = 0,
    dpi: int = 300,
    keep_pdf: bool = True,
    progress: bool = True,
) -> List[SampleResult]:
    """Generate `count` samples distributed round-robin across `tiers`.

    Each sample's seed is `seed_base + i`, where i is the sample index.
    Manifest is appended to `out_dir/manifest.jsonl`.
    """
    tiers = list(tiers)
    invalid = [t for t in tiers if t not in TIER_NAMES]
    if invalid:
        raise ValueError(f"unknown tiers: {invalid}; valid: {TIER_NAMES}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    results: List[SampleResult] = []
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for i in range(count):
            tier = tiers[i % len(tiers)]
            seed = seed_base + i
            sample_id = f"{i:05d}"
            result = generate_sample(
                sample_id, tier, seed, out_dir, dpi=dpi, keep_pdf=keep_pdf
            )
            manifest.write(json.dumps({
                "sample_id": result.sample_id,
                "tier": result.tier,
                "seed": result.seed,
                "image": str(result.image_path.relative_to(out_dir)),
                "json": str(result.json_path.relative_to(out_dir)),
                "pdf": str(result.pdf_path.relative_to(out_dir)) if result.pdf_path else None,
            }) + "\n")
            results.append(result)
            if progress and (i + 1) % 5 == 0:
                print(f"  generated {i + 1}/{count}")
    return results
