# Training initialization weights

Download the external release bundle and place the following files here:

```text
rf-detr-xxlarge.pth
sam_vit_h_4b8939.pth
```

They are required only for new training. The three released final inference
checkpoints under `weights/split*/` are self-contained and do not load these
files during evaluation. Verify all downloads with `weights/SHA256SUMS`.
