"""Isolated RF-DETR-2XL adapter for the DFC-SAM detector contract.

The upstream RF-DETR checkout is never patched.  Three detector feature sources
are observed with temporary hooks: shallow DINO, deep DINO, and the fused P4
projector output.  They deliberately represent depth/fusion diversity rather
than pretending that RF-DETR exposes YOLO's P3/P4/P5 pyramid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from dfc_sam.constants import PANNUKE_CLASSES

from .yolo26_adapter import DetectorOutput


class _DeterministicPositionEmbedding(nn.Module):
    """Exact sine positions for the integration's unpadded square RF feature grid.

    CUDA cumsum has no strict-deterministic implementation on the target GPU. Every
    RF input in this integration is the same fixed 880x880 shape, so its nested
    feature mask is all-false and cumsum(ones) is exactly a one-based arange.
    This instance-local replacement avoids relaxing determinism globally or
    changing the vendored RF-DETR source.
    """

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.num_pos_feats = int(original.num_pos_feats)
        self.temperature = float(original.temperature)
        self.normalize = bool(original.normalize)
        self.scale = float(original.scale)

    def forward(self, tensor_list: Any, align_dim_orders: bool = True) -> Tensor:
        values = tensor_list.tensors
        mask = tensor_list.mask
        if mask is None or bool(mask.any()):
            raise RuntimeError("RF-DETR DFC-SAM integration requires an unpadded fixed-size feature grid")
        batch, height, width = mask.shape
        y_embed = torch.arange(1, height + 1, device=values.device, dtype=torch.float32)
        y_embed = y_embed.view(1, height, 1).expand(batch, height, width)
        x_embed = torch.arange(1, width + 1, device=values.device, dtype=torch.float32)
        x_embed = x_embed.view(1, 1, width).expand(batch, height, width)
        if self.normalize:
            y_embed = y_embed / (float(height) + 1.0e-6) * self.scale
            x_embed = x_embed / (float(width) + 1.0e-6) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=values.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        position = torch.cat((pos_y, pos_x), dim=3)
        if align_dim_orders:
            return position.permute(1, 2, 0, 3)
        return position.permute(0, 3, 1, 2)


def _cxcywh_to_xyxy(boxes: Tensor, image_hw: tuple[int, int]) -> Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    result = torch.cat((center - size / 2, center + size / 2), dim=-1)
    height, width = image_hw
    return result * result.new_tensor((width, height, width, height))


class RFDETR2XLAdapter(nn.Module):
    """Expose a five-class RF-DETR-2XL through ``DetectorOutput``."""

    class_probability_mode = "sigmoid"
    feature_source_names = ("dino_shallow", "dino_deep", "projected_p4")

    def __init__(self, detector: Any) -> None:
        super().__init__()
        self.wrapper = detector
        context = getattr(detector, "model", None)
        model = getattr(context, "model", None)
        if not isinstance(model, nn.Module):
            raise TypeError("RF-DETR wrapper does not expose model.model as a torch module")
        self.model = model
        self._context = context
        self._resolution = int(detector.model_config.resolution)
        self.register_buffer(
            "pixel_mean",
            torch.tensor(detector.means, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(detector.stds, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self._validate_architecture()
        if len(self.model.backbone) < 2:
            # Lightweight fake models used by unit tests do not need positions.
            self._deterministic_position_installed = False
        else:
            self.model.backbone[1] = _DeterministicPositionEmbedding(self.model.backbone[1])
            self._deterministic_position_installed = True

    @property
    def semantic_dim(self) -> int:
        return int(self.model.class_embed.in_features)

    @property
    def pyramid_dims(self) -> tuple[int, int, int]:
        backbone = self.model.backbone[0]
        channels = tuple(int(value) for value in backbone.encoder._out_feature_channels)
        if not channels:
            raise RuntimeError("RF-DETR encoder did not declare output feature channels")
        projected = int(self.semantic_dim)
        return channels[0], channels[-1], projected

    @property
    def resolution(self) -> int:
        return self._resolution

    def _validate_architecture(self) -> None:
        if self._resolution != 880:
            raise ValueError(f"RF-DETR-2XL integration is frozen at 880px, got {self._resolution}")
        if int(self.model.num_queries) != 300:
            raise ValueError(f"Expected 300 RF-DETR queries, got {self.model.num_queries}")
        if int(self.model.class_embed.out_features) != len(PANNUKE_CLASSES) + 1:
            raise ValueError(
                "RF-DETR checkpoint must contain five foreground classes plus no-object"
            )
        class_names = getattr(self._context, "class_names", None)
        names = tuple(str(name).lower() for name in (class_names or ()))
        if names != PANNUKE_CLASSES:
            raise ValueError(f"RF-DETR checkpoint class order mismatch: {names!r}")
        backbone = self.model.backbone[0]
        if not hasattr(backbone, "encoder") or not hasattr(backbone, "projector"):
            raise TypeError("RF-DETR backbone does not expose encoder/projector hooks")
        if tuple(str(value) for value in backbone.projector_scale) != ("P4",):
            raise ValueError("The pinned RF-DETR-2XL integration expects one projected P4 feature")
        if self.semantic_dim != 512:
            raise ValueError(f"Expected RF-DETR-2XL semantic width 512, got {self.semantic_dim}")

    def train(self, mode: bool = True) -> RFDETR2XLAdapter:
        """Keep RF-DETR's one-group query layout while allowing gradients.

        The upstream module expands its query bank into training groups when
        ``model.training`` is true. DFC-SAM must retain the one-to-one 300-query
        identity used by the Bridge and Hungarian matcher, so Stage III enables
        parameter gradients but keeps the wrapped RF forward in evaluation
        layout. Bridge/SAM/UGCA modes are managed independently by the stage
        policy.
        """
        super().train(mode)
        self.model.eval()
        return self

    def forward(
        self,
        images: Tensor,
        *,
        coupled_grad: bool,
        decode_boxes_grad: bool = True,
    ) -> DetectorOutput:
        del decode_boxes_grad  # RF boxes are directly decoded normalized cxcywh values.
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RF-DETR RGB batch [B,3,H,W], got {tuple(images.shape)}")
        image_hw = (int(images.shape[-2]), int(images.shape[-1]))
        if image_hw != (self._resolution, self._resolution):
            raise ValueError(
                f"RF-DETR-2XL requires {self._resolution}x{self._resolution} input, got {image_hw}"
            )
        if self.model.training:
            raise RuntimeError(
                "RF-DETR must remain in eval mode inside DFC-SAM; its training mode expands "
                "300 inference queries into grouped training queries"
            )

        captured: dict[str, Any] = {}
        backbone = self.model.backbone[0]

        def capture(name: str):
            def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                captured[name] = inputs[0] if name == "decoder" else output

            return hook

        handles = (
            backbone.encoder.register_forward_hook(capture("encoder")),
            backbone.projector.register_forward_hook(capture("projector")),
            self.model.class_embed.register_forward_hook(capture("decoder")),
        )
        normalized = (images - self.pixel_mean) / self.pixel_std
        try:
            if coupled_grad:
                raw = self.model(normalized)
            else:
                with torch.no_grad():
                    raw = self.model(normalized)
        finally:
            for handle in handles:
                handle.remove()

        encoder = captured.get("encoder")
        projector = captured.get("projector")
        decoder = captured.get("decoder")
        if not isinstance(encoder, list | tuple) or len(encoder) < 2:
            raise RuntimeError("Could not capture RF-DETR shallow/deep DINO features")
        if not isinstance(projector, list | tuple) or len(projector) != 1:
            raise RuntimeError("Could not capture RF-DETR projected P4 feature")
        if not isinstance(decoder, Tensor) or decoder.ndim != 4:
            raise RuntimeError("Could not capture RF-DETR decoder query semantics")

        logits_with_background = raw["pred_logits"]
        boxes_cxcywh = raw["pred_boxes"]
        logits = logits_with_background[..., : len(PANNUKE_CLASSES)]
        semantics = decoder[-1]
        if logits.shape[:2] != semantics.shape[:2]:
            raise RuntimeError("RF-DETR semantic queries are not aligned with final predictions")
        query_indices = torch.arange(logits.shape[1], device=images.device, dtype=torch.long)
        # Preserve auxiliary/encoder predictions for the native Stage-III
        # detection criterion while DFB consumes only the aligned final layer.
        native = dict(raw)
        return DetectorOutput(
            p3=encoder[0],
            p4=encoder[-1],
            p5=projector[0],
            raw_one2many=native,
            raw_one2one=native,
            decoded_boxes_xyxy=_cxcywh_to_xyxy(boxes_cxcywh, image_hw),
            base_logits=logits,
            semantic_features=semantics,
            query_indices=query_indices,
        )


def load_rfdetr_2xlarge_adapter(checkpoint: str | Path) -> RFDETR2XLAdapter:
    """Load the Stage-I RF-DETR checkpoint without importing it on YOLO-only runs."""
    try:
        from rfdetr import from_checkpoint
    except ImportError as exc:  # pragma: no cover - depends on the isolated RF venv
        raise ImportError(
            "RF-DETR integration requires third_party/rf-detr/.venv and "
            "third_party/rf-detr/src on PYTHONPATH"
        ) from exc

    resolved = Path(checkpoint).expanduser().resolve()
    detector = from_checkpoint(
        resolved,
        num_classes=len(PANNUKE_CLASSES),
        accept_platform_model_license=True,
    )
    detector.model.model.eval()
    return RFDETR2XLAdapter(detector)
