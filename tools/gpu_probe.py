#!/usr/bin/env python
"""Record a tiny CUDA allocation on only the explicitly allowed devices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from _bootstrap import PROJECT_ROOT  # noqa: F401

from dfc_sam.utils.hashing import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-devices", default="0,1,2,3")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.expected_devices:
        raise SystemExit(
            f"Refusing probe: CUDA_VISIBLE_DEVICES={visible!r}, expected exactly {args.expected_devices!r}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in the configured Python environment")
    expected_count = len(args.expected_devices.split(","))
    if torch.cuda.device_count() != expected_count:
        raise SystemExit(f"Expected {expected_count} visible GPUs, got {torch.cuda.device_count()}")

    devices = []
    for logical_index in range(torch.cuda.device_count()):
        tensor = torch.ones(1, device=f"cuda:{logical_index}")
        devices.append(
            {
                "logical_index": logical_index,
                "physical_index": int(args.expected_devices.split(",")[logical_index]),
                "name": torch.cuda.get_device_name(logical_index),
                "capability": list(torch.cuda.get_device_capability(logical_index)),
                "probe_allocation_bytes": tensor.nelement() * tensor.element_size(),
            }
        )
        del tensor
    payload = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_visible_devices": visible,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
        "status": "passed",
    }
    atomic_write_json(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
