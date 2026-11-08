"""Shared model definition for the cat/dog detector.

We fine-tune torchvision's Faster R-CNN (MobileNetV3-Large FPN backbone),
pre-trained on COCO, to detect exactly two classes: cat and dog.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from python_pets.config import CLASSES, DEFAULT_WEIGHTS


def build_model(num_classes: int = len(CLASSES), pretrained: bool = True):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_detector(weights_path: Path = DEFAULT_WEIGHTS, device: str | None = None):
    """Return (model, device, mode).

    mode == "trained"  -> our fine-tuned cat/dog weights
    mode == "coco"     -> stock COCO model, filtered to cat/dog (fallback)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if weights_path.exists():
        model = build_model(pretrained=False)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        mode = "trained"
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
        mode = "coco"
    model.to(device).eval()
    return model, device, mode
