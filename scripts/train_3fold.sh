#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
for SPLIT in 1 2 3; do
  bash scripts/train_split.sh "$SPLIT"
done
