# Model card: DFC-SAM(RF)-H

## Model

- Detector: RF-DETR-2XL, five PanNuke foreground classes
- Segmenter: SAM-H
- Coupling: RF multi-depth Differentiable Feature Bridge
- Classification refinement: quality-aware UGCA-v3
- Inference: single-view (identity) with split-specific class-aware score settings
- Input: 256x256 RGB PanNuke patches; detector branch is resized to 880x880 and
  SAM branch follows the standard 1024-pixel SAM transform

## Checkpoint mapping

| Checkpoint | Training fold | Validation fold | Held-out test fold |
|---|---|---|---|
| split1 | Fold1 | Fold2 | Fold3 |
| split2 | Fold2 | Fold3 | Fold1 |
| split3 | Fold3 | Fold1 | Fold2 |

Each released checkpoint contains the complete detector, SAM-H, DFB, and UGCA
state. The three files are not an ensemble for one image; each is evaluated on
its corresponding held-out fold to reproduce the three-fold protocol.

The released checkpoints and public training entry point both follow PanNuke's
predefined one-fold train, one-fold validation, and one-fold test rotation. The
validation fold is used for checkpoint selection and frozen inference
calibration. The test fold is not used for either operation.

## Intended use

Academic research and reproducibility on PanNuke-style 256x256 histopathology
patches. The model is not a clinical device and must not be used for diagnosis
or patient-care decisions.

## Known limitations

- Dead-cell PQ is substantially lower than the other four classes.
- Performance varies by tissue; Skin mPQ is the lowest three-fold tissue mean,
  and Uterus has high fold-to-fold variance.
- Results use single-view inference; no geometric TTA or model ensemble is used.
- Results have only been established under the pinned PanNuke protocol and
  dependencies.

## Licensing

RF-DETR-2XL is governed by Roboflow PML-1.0 and a valid applicable Platform
Agreement. PanNuke is CC BY-NC-SA 4.0. Segment Anything is Apache-2.0. Users are
responsible for satisfying all applicable terms before downloading or using the
model artifacts.
