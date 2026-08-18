#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEVICES="${CUDA_DEVICES:-0}"
python tools/verify_assets.py
python tools/evaluate_3fold.py \
  --weights-root weights \
  --all-samples data/manifests/base/all_samples.jsonl \
  --manifest-root data/manifests/pannuke_standard_3fold \
  --metrics-repo third_party/PanNuke-metrics \
  --output-root results/threefold_test \
  --devices "$DEVICES" \
  "$@"
