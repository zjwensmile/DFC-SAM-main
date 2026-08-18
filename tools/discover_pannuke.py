#!/usr/bin/env python
"""Discover the unique official PanNuke fold triplets."""

from __future__ import annotations

import argparse
import json

from _bootstrap import PROJECT_ROOT  # noqa: F401

from dfc_sam.data.pannuke_discovery import write_discovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_discovery(args.raw_root, args.out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
