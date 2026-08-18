# Release checklist

## Completed in this staging bundle

- GitHub code folder separated from multi-GB assets.
- Three path-neutral, inference-only FP32 checkpoints exported.
- RF-DETR-2XL and SAM-H new-training initialization checkpoints staged.
- SHA256 manifests generated and reverified.
- Complete model-state prefixes verified for detector, SAM-H, DFB, and UGCA.
- PanNuke preparation and official fold-rotation manifest generation included.
- Resumable one-/multi-GPU three-fold evaluator included.
- Main, 19-tissue, five-class, and mPQ_cls summary code verified against the 12 formal test shards.
- No machine-local `/data2/...` path remains in text configs or embedded checkpoint configs.

## Required before public GitHub/Google Drive release

- [ ] Copyright holder chooses the final license for project-authored code and replaces the staging `LICENSE`.
- [ ] Confirm that the intended Roboflow Platform Agreement permits redistribution of the fine-tuned 2XL-derived checkpoints.
- [ ] Confirm that the agreement also permits redistribution of `rf-detr-xxlarge.pth`; otherwise replace it with the official gated download instructions.
- [ ] Upload `release_assets/DFC-SAM-RF2XL/` to external storage.
- [ ] Replace `GOOGLE_DRIVE_URL_TO_BE_ADDED` in `README.md`.
- [ ] Run the deferred clean-room isolation inference test.
- [ ] Add the paper citation and final repository URL.
- [ ] Create a fresh clone and run installation/data/weight verification commands exactly as documented.
