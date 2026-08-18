# DFC-SAM(RF)-H: RF-DETR-2XL + SAM-H on PanNuke

This repository is the public training and evaluation package for the
three-fold PanNuke main experiment using
RF-DETR-2XL, SAM-H, the Differentiable Feature Bridge (DFB), and UGCA-v3. The
released test path uses each split's validation-frozen class-aware four-view TTA
configuration.

## Results

All values are percentages on the three non-overlapping held-out PanNuke folds.

| Result | bPQ | mPQ | F1det | Macro-F1 |
|---|---:|---:|---:|---:|
| Split1 / Fold3, 2722 images | 70.554 | 52.582 | 83.284 | 67.982 |
| Split2 / Fold1, 2656 images | 70.484 | 53.568 | 83.401 | 70.166 |
| Split3 / Fold2, 2523 images | 69.547 | 51.680 | 83.279 | 69.078 |
| **Three-fold mean** | **70.195** | **52.610** | **83.321** | **69.075** |

| Neoplastic | Epithelial | Inflammatory | Connective/soft | Dead | mPQ_cls |
|---:|---:|---:|---:|---:|---:|
| 60.748 | 59.858 | 45.637 | 45.205 | 18.397 | 45.969 |

The complete 19-tissue expected table is stored in
`expected_results/threefold_summary.json`.

## Important licenses

This release combines components with different terms. Review them before use.

- RF-DETR core code is Apache-2.0, but RF-DETR-2XL and `rfdetr-plus` are under
  Roboflow Platform Model License 1.0. Users must accept and comply with the
  applicable Roboflow Platform Agreement. The full PML text is included at
  `licenses/RFDETR_PLUS_PML-1.0.txt`.
- Segment Anything is Apache-2.0.
- PanNuke is CC BY-NC-SA 4.0 and is intended here for non-commercial research.
- PanNuke is not included. Download it from the official dataset source.

The top-level `LICENSE` is a staging research notice and must be replaced by the
copyright holder's intended final code license before publication.

## Installation

Python 3.10 and a CUDA-capable PyTorch environment are recommended. The formal
experiments used PyTorch 2.5.1, CUDA 11.8 wheels, and four Tesla P100 16GB GPUs.

```bash
conda create -n dfc-sam-rf2xl python=3.10 -y
conda activate dfc-sam-rf2xl

pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.5.1 torchvision==0.20.1
pip install -r requirements.lock.txt
pip install "rfdetr @ git+https://github.com/roboflow/rf-detr.git@c107a748bbf369ad455cd41d85b9c91a5a0ecbad"
pip install --no-deps rfdetr-plus==1.0.2
pip install "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"
pip install -e .

mkdir -p third_party
git clone https://github.com/TissueImageAnalytics/PanNuke-metrics.git third_party/PanNuke-metrics
git -C third_party/PanNuke-metrics checkout c00014d766ca1be142b81bea19d9ef4315cde65a
```


## Dataset

We do not redistribute PanNuke. Download all three official folds, then place
the `images.npy`, `masks.npy`, and `types.npy` files under directories containing
the corresponding `Fold1`, `Fold2`, and `Fold3` markers inside
`data/pannuke/raw/`. Prepare all assets and deterministic manifests with:

```bash
bash scripts/prepare_data.sh
```

The script verifies Fold1=2656, Fold2=2523, Fold3=2722, total=7901. See
`data/README.md` for details.

Both the released checkpoints and the public training entry point use the
original PanNuke fold rotation without creating a new random image-level split:

| Split | Train | Validation | Test |
|---|---|---|---|
| Split1 | Fold1 (2656) | Fold2 (2523) | Fold3 (2722) |
| Split2 | Fold2 (2523) | Fold3 (2722) | Fold1 (2656) |
| Split3 | Fold3 (2722) | Fold1 (2656) | Fold2 (2523) |

## Model weights

The large files are not stored in GitHub. Download the release archive from:

```text
GOOGLE_DRIVE_URL_TO_BE_ADDED
```

Place the files as follows:

```text
weights/
├── split1/rfdetr_dfc_sam_2xl_split1.pt
├── split2/rfdetr_dfc_sam_2xl_split2.pt
├── split3/rfdetr_dfc_sam_2xl_split3.pt
└── pretrained/
    ├── rf-detr-xxlarge.pth
    └── sam_vit_h_4b8939.pth
```

Each file contains the complete RF-DETR-2XL + SAM-H + DFB + UGCA-v3 inference
state for one split. Separate detector or SAM-H checkpoints are not required for
testing. Run `python tools/verify_assets.py` after downloading.

At the current staging milestone, metadata/checksum validation is implemented;
the planned clean-room isolation inference verification has not yet been run.

## Three-fold evaluation

The evaluation command automatically applies the matching checkpoint to the
matching held-out fold:

| Checkpoint | Test data |
|---|---|
| `split1/rfdetr_dfc_sam_2xl_split1.pt` | Fold3, 2722 images |
| `split2/rfdetr_dfc_sam_2xl_split2.pt` | Fold1, 2656 images |
| `split3/rfdetr_dfc_sam_2xl_split3.pt` | Fold2, 2523 images |

The evaluator checks the checkpoint split/test-fold metadata against the
generated PanNuke manifest before starting inference. It loads the frozen
inference and four-view TTA settings embedded in each checkpoint; the files in
`configs/test/` are not used to override a release checkpoint.

```bash
# One GPU, sequential evaluation
CUDA_DEVICES=0 bash scripts/test_3fold.sh

# Four shards per split
CUDA_DEVICES=0,1,2,3 bash scripts/test_3fold.sh

# Resume completed shards
CUDA_DEVICES=0,1,2,3 bash scripts/test_3fold.sh --resume

# Validate all checkpoint/fold mappings without running inference
bash scripts/test_3fold.sh --preflight-only

# In another terminal
watch -n 5 bash scripts/watch.sh
```

Final files are written to `results/threefold_test/summary/`:

- `summary.json`: complete machine-readable result;
- `summary.md`: main three-fold table;
- `threefold_metrics.csv`: bPQ, mPQ, F1det, and Macro-F1;
- `per_tissue.csv`: 19-tissue bPQ/mPQ per split plus mean and population SD;
- `per_class_pq.csv`: five-class PQ per split plus mean and population SD.

After all three splits finish, the terminal also prints the bPQ, mPQ, F1det,
and Macro-F1 result for Split1, Split2, and Split3, followed by their mean.

The evaluator never calibrates on a test fold. It only uses the TTA and
class-aware settings embedded in the corresponding release checkpoint.

## Main-experiment training

The download bundle includes the two initialization checkpoints required for
from-scratch training:

```text
weights/pretrained/
├── rf-detr-xxlarge.pth
└── sam_vit_h_4b8939.pth
```

Verify that both inference and new-training initialization weights are present:

```bash
python tools/verify_assets.py --require-pretrained
```

Run one split or all splits on four visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_split.sh 1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_3fold.sh
```

The public training path contains the main stages used for the released
checkpoints: RF-DETR-2XL
detector training, DFB + SAM-H warmup, UGCA-v3 training, and the validation-only
calibration implementations under `tools/`. Calibration must be frozen before
test evaluation. Rejected exploratory experiments and internal one-shot test
ledger code are intentionally omitted.

The released checkpoints, reported table, generated manifests, and public
training commands all use the same predefined one-train/one-validation/one-test
PanNuke fold rotation shown above. The validation fold is used for checkpoint
selection and inference calibration only; the corresponding test fold remains
sealed until final evaluation. Normal nondeterminism from hardware and external
CUDA kernels may still prevent a newly trained checkpoint from being
bit-for-bit identical.

## Metric definitions

- bPQ/mPQ use the pinned official PanNuke primitives and macro-average over 19
  tissue types.
- F1det pools binary instance TP/FP/FN at IoU > 0.5 within each held-out fold.
- Macro-F1 is the unweighted mean of the five instance-class F1 values.
- `mPQ_cls` is the unweighted mean of the five class PQ values and is not the
  same aggregation as the official 19-tissue mPQ.
- The paper row is the arithmetic mean of the three held-out split metrics.

## Citation

The DFC-SAM paper citation will be added after publication. Please also cite
PanNuke, Segment Anything, and RF-DETR according to their upstream repositories.
