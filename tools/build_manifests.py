#!/usr/bin/env python
"""Build immutable all-sample and rotating split manifests."""

from __future__ import annotations

import argparse
import json

from _bootstrap import PROJECT_ROOT  # noqa: F401

from dfc_sam.data.manifests import build_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prepared-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build_manifests(args.discovery, args.out, prepared_root=args.prepared_root),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
