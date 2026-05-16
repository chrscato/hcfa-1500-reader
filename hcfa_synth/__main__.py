"""CLI entrypoint: `python -m hcfa_synth <command> [args]`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hcfa_synth.augment import TIER_NAMES
from hcfa_synth.pipeline import generate_batch


def _parse_tiers(arg: str) -> list[str]:
    if arg == "all":
        return list(TIER_NAMES)
    requested = [t.strip() for t in arg.split(",") if t.strip()]
    invalid = [t for t in requested if t not in TIER_NAMES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown tier(s): {invalid}. Valid: 'all' or {TIER_NAMES}"
        )
    return requested


def _cmd_generate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    print(f"Generating {args.count} samples across tiers {args.tiers} -> {out}")
    print(f"  dpi={args.dpi}, seed_base={args.seed}, keep_pdf={not args.no_pdf}")
    start = time.time()
    results = generate_batch(
        count=args.count,
        tiers=args.tiers,
        out_dir=out,
        seed_base=args.seed,
        dpi=args.dpi,
        keep_pdf=not args.no_pdf,
    )
    elapsed = time.time() - start
    per_sample = elapsed / max(len(results), 1)
    print(f"Done. {len(results)} samples in {elapsed:.1f}s ({per_sample:.2f}s/sample)")
    print(f"Manifest: {out / 'manifest.jsonl'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hcfa-synth",
        description="Generate synthetic CMS-1500 (HCFA) claim forms.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate a batch of samples")
    gen.add_argument("--count", type=int, default=20, help="Number of samples (default 20)")
    gen.add_argument(
        "--tiers", type=_parse_tiers, default=_parse_tiers("all"),
        help=f"Comma-separated tier names, or 'all'. Valid: {TIER_NAMES}",
    )
    gen.add_argument("--out", type=str, default="./data", help="Output directory")
    gen.add_argument("--seed", type=int, default=0, help="Starting seed for record generation")
    gen.add_argument("--dpi", type=int, default=300, help="Render DPI (default 300)")
    gen.add_argument("--no-pdf", action="store_true", help="Skip writing source PDFs")
    gen.set_defaults(func=_cmd_generate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
