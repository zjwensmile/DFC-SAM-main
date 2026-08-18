#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python tools/prepare_pannuke.py \
  --raw-root data/pannuke/raw \
  --prepared-root data/pannuke/prepared \
  --manifest-root data/manifests
