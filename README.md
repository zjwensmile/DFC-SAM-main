# DFC-SAM: Differentiable Feature Coupling of Instance Detectors and SAM

DFC-SAM is a differentiable feature-coupling framework that connects an
instance detector with the Segment Anything Model (SAM) for joint instance
detection, segmentation, and fine-grained classification. Instead of using
human prompts at inference time, the detector supplies continuous predicted
boxes and instance semantics to SAM through a learnable Differentiable Feature
Bridge (DFB). The released PanNuke model combines **RF-DETR-2XL + DFB + SAM-H +
UGCA** and uses **single-view inference**.

The manuscript implements both YOLO26- and RF-DETR-based variants to examine
detector transfer. Its main model and this repository's released PanNuke model
are **DFC-SAM(RF)-H**.

> **Repository scope.** The manuscript reports experiments on a private
> cerebrospinal-fluid (CSF) dataset, PanNuke, and MoNuSAC. This public repository
> currently packages the PanNuke three-fold experiment only. The private CSF
> data cannot be redistributed, and the MoNuSAC experiment is not included in
> this release.

## Method at a glance

```text
image
  ├─ RF-DETR-2XL ─ multi-depth features + matched query semantics + continuous boxes
  │                         │
  │                         ▼
  │                        DFB ── dense instance-conditioned prompts ──► SAM-H ──► masks
  │                                                                       │
  └──────────────── class evidence ◄──────────── UGCA ◄── local morphology ┘
```

- **Differentiable Feature Bridge (DFB)** fuses multi-depth detector features,
  matched-query semantics, and continuous predicted boxes through
  feature-source-aware attention and continuous spatial gating. It produces
  instance-conditioned dense prompts aligned with the SAM latent space, so the
  mask loss can backpropagate through matched positive instances to detector
  features and box regression.
- **Uncertainty-Gated Cross-Attention (UGCA)** uses stop-gradient local
  morphology features from SAM to refine class evidence and candidate quality,
  with stronger correction when classification uncertainty is high. Information
  exchange is bidirectional, while gradient propagation is intentionally
  asymmetric.
- **Quality-aware Weak Supervision (QWS)** is a training-only, data-efficiency
  strategy that selects and continuously weights teacher pseudo masks. Its
  teacher and pseudo-target store are removed at inference. The released
  PanNuke checkpoints use full mask supervision; QWS is retained as an
  auxiliary experimental path rather than the default release setting.

DFC-SAM is **external-prompt-free but internally box-conditioned**: no human or
external prompt is required, but the detector's continuous predicted boxes are
used inside the model. The DFC path does not invoke SAM's original box prompt
encoder, and its sparse prompt embedding is empty. The released evaluator runs
one identity-view forward pass and does not perform box NMS or mask NMS.
Overlapping masks undergo pixel-level competition only when the official
evaluator requires a mutually exclusive label map; complete instances are not
suppressed by that conversion.

## Results reported in the manuscript

The following values summarize the main results described in the manuscript.
Only the PanNuke row is covered by the public data, checkpoints, and evaluation
path in this repository.

| Dataset / evaluation | Main reported results |
|---|---|
| Private CSF test set | AP50 91.6%, DSCmean 90.4%, AJI 84.3% |
| Independent clinical validation | Macro-F1 87.6%, overall MAE 2.3 cells/image |
| PanNuke, official three-fold protocol | bPQ 70.5%, mPQ 52.6%, F1det 83.3%, Macro-F1 69.0% |
| MoNuSAC, corrected official evaluation | aPQ 64.7% |

### Reproducible PanNuke three-fold results

All values below are percentages on the three non-overlapping held-out folds.
The first table reports the exact values stored in
`expected_results/threefold_summary.json`; the rounded manuscript row is the
arithmetic mean of these three split results.

| Result | bPQ | mPQ | F1det | Macro-F1 |
|---|---:|---:|---:|---:|
| Split1 / Fold3, 2722 images | 70.554 | 52.582 | 83.284 | 67.982 |
| Split2 / Fold1, 2656 images | 70.484 | 53.568 | 83.401 | 70.166 |
| Split3 / Fold2, 2523 images | 69.547 | 51.680 | 83.279 | 69.078 |
| **Three-fold mean** | **70.195** | **52.610** | **83.321** | **69.075** |

| Neoplastic | Epithelial | Inflammatory | Connective/soft | Dead | mPQ_cls |
|---:|---:|---:|---:|---:|---:|
| 60.748 | 59.858 | 45.637 | 45.205 | 18.397 | 45.969 |

The complete 19-tissue table is stored in
`expected_results/threefold_summary.json`.

### Model size and computational cost

The full DFC-SAM(RF)-H model is not a lightweight system: the manuscript
reports 768.6M parameters and 2905.6G MACs for single-view inference. Relative
to the same RF-DETR-2XL + SAM-H box-prompt baseline, however, DFB and UGCA add
only 1.2M parameters (+0.16%) and 2.2G MACs (+0.08%), while improving bPQ by
2.8 points, mPQ by 4.0 points, and Macro-F1 by 8.7 points.

## Installation

Python 3.10 and a CUDA-capable PyTorch environment are recommended. The
manuscript experiments used PyTorch 2.5.1, CUDA 11.8 wheels, and four NVIDIA
Tesla V100 GPUs.

```bash
conda create -n dfc-sam-rf2xl python=3.10 -y
conda activate dfc-sam-rf2xl

pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.5.1 torchvision==0.20.1
pip install -r requirements.lock.txt
pip install "rfdetr[train] @ git+https://github.com/roboflow/rf-detr.git@c107a748bbf369ad455cd41d85b9c91a5a0ecbad"
pip install --no-deps rfdetr-plus==1.0.2
pip install "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"
pip install -e .

mkdir -p third_party
git clone https://github.com/TissueImageAnalytics/PanNuke-metrics.git third_party/PanNuke-metrics
git -C third_party/PanNuke-metrics checkout c00014d766ca1be142b81bea19d9ef4315cde65a
```

## Dataset and official split protocol

PanNuke contains 7,901 H&E image patches of size 256 x 256 from 19 tissue
types, with five nucleus classes and approximately 189,744 annotated instances.
PanNuke is not redistributed by this repository. Download all three official
folds, then place `images.npy`, `masks.npy`, and `types.npy` under directories
containing the corresponding `Fold1`, `Fold2`, and `Fold3` markers inside
`data/pannuke/raw/`.

Prepare the assets and deterministic manifests with:

```bash
bash scripts/prepare_data.sh
```

The script verifies Fold1=2656, Fold2=2523, Fold3=2722, total=7901. See
`data/README.md` for the expected directory layout.

The release preserves the official folds and rotates one fold for training,
one for validation, and one for testing. It does not create a new random
image-level split.

| Split | Train | Validation | Test |
|---|---|---|---|
| Split1 | Fold1 (2656) | Fold2 (2523) | Fold3 (2722) |
| Split2 | Fold2 (2523) | Fold3 (2722) | Fold1 (2656) |
| Split3 | Fold3 (2722) | Fold1 (2656) | Fold2 (2523) |

The validation fold is used for checkpoint selection and class-aware
score/quality calibration. These settings are frozen before the test fold is
evaluated; the evaluator never recalibrates on test data.

## Model weights

Large model files are hosted separately. Download the release bundle from
[Google Drive](https://drive.google.com/drive/folders/10TX81FeUtgi7aGWxu7vsqRi43Rg8pawa?usp=sharing).
The scripts do not download weights automatically.

Place the downloaded files as follows:

```text
weights/
├── split1/rfdetr_dfc_sam_2xl_split1.pt
├── split2/rfdetr_dfc_sam_2xl_split2.pt
├── split3/rfdetr_dfc_sam_2xl_split3.pt
└── pretrained/
    ├── rf-detr-xxlarge.pth
    └── sam_vit_h_4b8939.pth
```

Each split file contains the complete RF-DETR-2XL + SAM-H + DFB + UGCA
inference state. Separate detector or SAM-H checkpoints are not required for
testing. Check the downloaded files against the supplied manifest and hashes:

```bash
python tools/verify_assets.py
```

At this release milestone, metadata and checksum verification are implemented;
a clean-room isolation inference run on a newly configured machine has not yet
been completed.

## Three-fold evaluation

The evaluation script automatically pairs every checkpoint with its official
held-out fold:

| Checkpoint | Test data |
|---|---|
| `split1/rfdetr_dfc_sam_2xl_split1.pt` | Fold3, 2722 images |
| `split2/rfdetr_dfc_sam_2xl_split2.pt` | Fold1, 2656 images |
| `split3/rfdetr_dfc_sam_2xl_split3.pt` | Fold2, 2523 images |

Before inference, it verifies the split and test-fold metadata in the
checkpoint against the generated PanNuke manifest. It then runs exactly one
identity-view forward pass and applies the validation-frozen class-aware score
settings stored in that checkpoint. Legacy four-view TTA metadata embedded in
the released checkpoint files is ignored by the current public evaluator.

```bash
# One GPU, sequential evaluation
CUDA_DEVICES=0 bash scripts/test_3fold.sh

# Four shards per split
CUDA_DEVICES=0,1,2,3 bash scripts/test_3fold.sh

# Resume completed shards
CUDA_DEVICES=0,1,2,3 bash scripts/test_3fold.sh --resume

# Validate all checkpoint/fold mappings without running inference
bash scripts/test_3fold.sh --preflight-only

# Optional progress monitor in another terminal
watch -n 5 bash scripts/watch.sh
```

After all three splits finish, the terminal prints the bPQ, mPQ, F1det, and
Macro-F1 result for each split followed by their arithmetic mean. Final files
are written to `results/threefold_test/summary/`:

- `summary.json`: complete machine-readable result;
- `summary.md`: main three-fold table;
- `threefold_metrics.csv`: bPQ, mPQ, F1det, and Macro-F1;
- `per_tissue.csv`: 19-tissue bPQ/mPQ per split plus mean and population SD;
- `per_class_pq.csv`: five-class PQ per split plus mean and population SD.

Therefore, after the data and weights are placed correctly, running
`scripts/test_3fold.sh` produces the corresponding Split1, Split2, and Split3
test results directly; no manual reassignment of checkpoints or folds is
needed.

## Training strategy

The manuscript uses a three-stage optimization strategy:

1. **Stage I — detector initialization.** Train RF-DETR-2XL with true instance
   boxes.
2. **Stage II — bridge and decoder warmup.** First adapt the teacher and build a
   pseudo-target store with three candidate masks per instance when running a
   QWS mixed-supervision experiment; then freeze RF-DETR and warm up DFB
   together with the student SAM mask decoder. The released fully supervised
   PanNuke path does not require the teacher/pseudo-target substep.
3. **Stage III — joint optimization.** Freeze the teacher and pseudo-target
   store, then jointly optimize the detector, DFB, student mask decoder, and
   UGCA. The mask objective remains differentiable along the matched
   positive-instance path.

The teacher image/prompt encoders and the student image encoder remain frozen.
The manuscript schedule uses four V100 GPUs, batch size 2 per GPU (global batch
size 8), 50 epochs at learning rate 1e-4 for Stage I, and 20 epochs each at
learning rate 1e-5 for Stages II and III.

The download bundle also contains the two initialization checkpoints used by
the public training entry points:

```text
weights/pretrained/
├── rf-detr-xxlarge.pth
└── sam_vit_h_4b8939.pth
```

Verify them with:

```bash
python tools/verify_assets.py --require-pretrained
```

The repository includes the detector, DFB/SAM warmup, joint detector/DFB/SAM/
UGCA optimization, calibration, and three-fold orchestration code. The public
training scripts can be invoked with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_split.sh 1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_3fold.sh
```

### Reproducibility status

- **Supported in this release:** official PanNuke fold preparation,
  checkpoint/fold preflight validation, single-view three-fold inference,
  frozen validation calibration, metric aggregation, expected-result files,
  and released inference weights.
- **Training implementation included:** the major model components and staged
  training entry points are present. The public configurations encode the
  manuscript's 50/20/20 epoch schedule, 1e-4/1e-5/1e-5 learning rates, global
  batch size 8 on four GPUs, and Stage-III loss weights
  `lambda_det=0.25`, `lambda_seg=1.0`, and `lambda_ugca=0.25`.
- **Full retraining validation pending:** the configuration and execution path
  have been statically aligned with the manuscript, but a clean end-to-end
  three-fold retraining run has not yet been completed from this public package.
  The released evaluation weights and official split mapping remain the
  currently verified reproducibility path.

Normal nondeterminism from hardware and external CUDA kernels may prevent a
newly trained checkpoint from being bit-for-bit identical even after the
schedule is aligned.

## Metric definitions

- **bPQ / mPQ:** official PanNuke primitives macro-averaged over 19 tissue
  types.
- **F1det:** binary instance TP/FP/FN pooled within each held-out fold at IoU >
  0.5.
- **Macro-F1:** unweighted mean of the five instance-class F1 scores.
- **mPQ_cls:** unweighted mean of the five class PQ values; this is different
  from the official 19-tissue mPQ aggregation.
- **Three-fold paper row:** arithmetic mean of the three held-out split
  metrics.

## Important licenses

This release combines components and data with different terms. Review each
license before use.

- RF-DETR core code is Apache-2.0, while RF-DETR-2XL and `rfdetr-plus` are
  distributed under Roboflow Platform Model License 1.0. Users must accept and
  comply with the applicable Roboflow Platform Agreement. The included license
  text is at `licenses/RFDETR_PLUS_PML-1.0.txt`.
- Segment Anything is Apache-2.0.
- PanNuke is CC BY-NC-SA 4.0 and is used here for non-commercial research.
- PanNuke data are not included and must be obtained from its official source.

The top-level `LICENSE` is currently a staging research notice. The copyright
holder should replace it with the intended final code license before public
release.

## Citation

The DFC-SAM citation will be added after publication. Please also cite PanNuke,
Segment Anything, and RF-DETR according to their original publications and
upstream repositories.
