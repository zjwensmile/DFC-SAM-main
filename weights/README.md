# Model weights

The three large checkpoints are intentionally excluded from GitHub. Download the
Google Drive weight archive announced with the release, verify `SHA256SUMS`, and
place the files exactly as follows:

```text
weights/
├── split1/rfdetr_dfc_sam_2xl_split1.pt  # held-out Fold3
├── split2/rfdetr_dfc_sam_2xl_split2.pt  # held-out Fold1
├── split3/rfdetr_dfc_sam_2xl_split3.pt  # held-out Fold2
└── pretrained/
    ├── rf-detr-xxlarge.pth               # detector initialization
    └── sam_vit_h_4b8939.pth              # SAM-H initialization
```

Each inference checkpoint contains the complete RF-DETR-2XL detector, SAM-H,
DFB, and UGCA-v3 state. No separate RF-DETR or SAM-H checkpoint is required for
testing. The two files under `pretrained/` are only used for new training and are
included in the external download bundle for convenience.

RF-DETR-2XL is a Plus model under Roboflow PML-1.0. Downloading and using these
derived checkpoints requires acceptance of and compliance with the applicable
Roboflow Platform Agreement. PanNuke-derived model artifacts are distributed for
non-commercial research under the dataset's CC BY-NC-SA 4.0 terms.
