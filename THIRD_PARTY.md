# Third-party components

| Component | Pinned revision/version | Terms |
|---|---|---|
| RF-DETR core | `c107a748bbf369ad455cd41d85b9c91a5a0ecbad` | Apache-2.0 |
| `rfdetr-plus` / RF-DETR-2XL | `1.0.2` | PML-1.0 plus applicable Roboflow Platform Agreement |
| Segment Anything | `dca509fe793f601edb92606367a655c15ac00fdf` | Apache-2.0 |
| PanNuke | official three folds | CC BY-NC-SA 4.0 |
| PanNuke metrics | `c00014d766ca1be142b81bea19d9ef4315cde65a` | No upstream license file in the pinned repository; used only for research metric parity |

No third-party source code is copied into this repository. The external weight
bundle contains SAM-H and RF-DETR-2XL initialization binaries for training;
their upstream terms still apply. `rfdetr-plus` must be installed separately
after the user reviews and accepts its license. The full PML-1.0 notice
distributed with version 1.0.2 is reproduced under `licenses/` as required by
that license. Segment Anything's Apache-2.0 text is included at
`licenses/SEGMENT_ANYTHING_APACHE-2.0.txt`.
