# PanNuke data

The dataset is not redistributed in this repository or in the model-weight
archive. Download the three official PanNuke folds from the dataset authors and
place the unique `images.npy`, `masks.npy`, and `types.npy` triplet for each fold
under `data/pannuke/raw/`. Nested directories are accepted as long as every path
contains an unambiguous `Fold1`, `Fold2`, or `Fold3` marker.

Then run:

```bash
bash scripts/prepare_data.sh
```

The preparation code verifies the official test-fold counts:

- Fold1: 2656 images
- Fold2: 2523 images
- Fold3: 2722 images
- Total: 7901 images

It writes RGB PNGs and five-class YOLO box labels for detector training, while
the segmentation evaluator continues to read the original instance-mask arrays
through generated manifests. Generated data and manifests are ignored by Git.

No additional random image-level partition is created. The generated manifests
use the PanNuke predefined fold rotation:

| Split | Train | Validation | Test |
|---|---|---|---|
| Split1 | Fold1 | Fold2 | Fold3 |
| Split2 | Fold2 | Fold3 | Fold1 |
| Split3 | Fold3 | Fold1 | Fold2 |
