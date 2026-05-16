"""Tests for the end-to-end pipeline."""

import json
from pathlib import Path

import pytest

from hcfa_synth.pipeline import generate_batch, generate_sample


def test_generate_one_sample(tmp_path: Path):
    result = generate_sample("00001", "pristine", seed=1, out_dir=tmp_path, dpi=100)
    assert result.image_path.exists()
    assert result.json_path.exists()
    assert result.pdf_path is not None and result.pdf_path.exists()

    gt = json.loads(result.json_path.read_text())
    assert gt["sample"]["id"] == "00001"
    assert gt["sample"]["tier"] == "pristine"
    assert gt["sample"]["seed"] == 1


def test_generate_sample_without_pdf(tmp_path: Path):
    result = generate_sample("00001", "fax", seed=1, out_dir=tmp_path, dpi=100, keep_pdf=False)
    assert result.image_path.exists()
    assert result.pdf_path is None


def test_generate_batch_round_robins_tiers(tmp_path: Path):
    results = generate_batch(
        count=6,
        tiers=["pristine", "fax"],
        out_dir=tmp_path,
        dpi=100,
        progress=False,
    )
    assert len(results) == 6
    tiers = [r.tier for r in results]
    assert tiers == ["pristine", "fax", "pristine", "fax", "pristine", "fax"]


def test_generate_batch_writes_manifest(tmp_path: Path):
    generate_batch(
        count=3,
        tiers=["pristine"],
        out_dir=tmp_path,
        dpi=100,
        progress=False,
    )
    manifest = tmp_path / "manifest.jsonl"
    assert manifest.exists()
    lines = manifest.read_text().strip().split("\n")
    assert len(lines) == 3
    parsed = [json.loads(l) for l in lines]
    assert {p["sample_id"] for p in parsed} == {"00000", "00001", "00002"}


def test_unknown_tier_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown tiers"):
        generate_batch(count=1, tiers=["not_a_tier"], out_dir=tmp_path)
