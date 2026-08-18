#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python - <<'PY'
import json
from pathlib import Path
paths=sorted(Path('results/threefold_test').glob('split*/shards/*_progress.json'))
if not paths:
    print('No test progress files yet.')
for path in paths:
    data=json.loads(path.read_text())
    pct=100.0*float(data.get('fraction',0.0))
    eta=data.get('eta_seconds')
    eta_text='unknown' if eta is None else f'{float(eta)/3600:.2f} h'
    print(f"{path.parent.parent.name}/{path.stem}: {data.get('status')} "
          f"{data.get('processed',0)}/{data.get('total',0)} ({pct:.1f}%), ETA {eta_text}")
PY
