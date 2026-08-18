#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^[123]$ ]]; then
  echo "Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_split.sh SPLIT_ID" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SPLIT="$1"
RUN_ROOT="outputs/training/split${SPLIT}"
DETECTOR_DATA="${RUN_ROOT}/detector_data"
DETECTOR_OUT="${RUN_ROOT}/detector"
SAM_H="weights/pretrained/sam_vit_h_4b8939.pth"
RF_INIT="weights/pretrained/rf-detr-xxlarge.pth"
MANIFEST="data/manifests/pannuke_standard_3fold/split_${SPLIT}.json"
ALL_SAMPLES="data/manifests/base/all_samples.jsonl"
METRICS_REPO="third_party/PanNuke-metrics"

python tools/prepare_rfdetr_split1.py \
  --split-id "$SPLIT" \
  --train-manifest "data/manifests/pannuke_standard_3fold/detector/split${SPLIT}_train.txt" \
  --validation-manifest "data/manifests/pannuke_standard_3fold/detector/split${SPLIT}_validation.txt" \
  --output "$DETECTOR_DATA"

RFDETR_SEQUENCE_AUTHORIZED=1 python -m torch.distributed.run --standalone --nproc_per_node=4 \
  tools/train_rfdetr_split1.py \
  --split-id "$SPLIT" --variant 2xlarge --dataset "$DETECTOR_DATA" \
  --weight "$RF_INIT" --output "$DETECTOR_OUT" --resolution 880 \
  --epochs 200 --batch-size 2 --grad-accum-steps 2 --workers 4 --devices auto \
  --gradient-checkpointing --execute

python tools/materialize_training_config.py \
  --template "configs/train/split${SPLIT}_warmup.yaml" \
  --detector "${DETECTOR_OUT}/checkpoint_best_total.pth" --sam-h "$SAM_H" \
  --split-manifest "$MANIFEST" \
  --output "${RUN_ROOT}/warmup.yaml"

python -m torch.distributed.run --nproc_per_node=4 \
  tools/train_bridge.py --config "${RUN_ROOT}/warmup.yaml" \
  --all-samples "$ALL_SAMPLES" --split-manifest "$MANIFEST" \
  --output-root "${RUN_ROOT}/warmup" --device cuda:0 \
  --metrics-repo "$METRICS_REPO" --execute

python tools/materialize_training_config.py \
  --template "configs/train/split${SPLIT}_ugca.yaml" \
  --detector "${DETECTOR_OUT}/checkpoint_best_total.pth" --sam-h "$SAM_H" \
  --warmup "${RUN_ROOT}/warmup/best.pt" --split-manifest "$MANIFEST" \
  --output "${RUN_ROOT}/ugca.yaml"

python -m torch.distributed.run --nproc_per_node=4 \
  tools/train_joint.py --config "${RUN_ROOT}/ugca.yaml" \
  --all-samples "$ALL_SAMPLES" --split-manifest "$MANIFEST" \
  --output-root "${RUN_ROOT}/ugca" --device cuda:0 \
  --metrics-repo "$METRICS_REPO" --execute

echo "Split${SPLIT} training completed. Run validation-only calibration before any test evaluation."
