#!/usr/bin/env python
"""Audit original PanNuke arrays."""

from __future__ import annotations

import argparse
import json

from _bootstrap import PROJECT_ROOT  # noqa: F401

from dfc_sam.data.pannuke_audit import write_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Framework smoke mode only. Omit for the formal full audit.",
    )
    parser.add_argument("--skip-source-hashes", action="store_true")
    parser.add_argument("--skip-duplicate-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_audit(
        args.discovery,
        args.out,
        sample_limit=args.sample_limit,
        hash_source_files=not args.skip_source_hashes,
        check_cross_fold_duplicates=not args.skip_duplicate_check,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
