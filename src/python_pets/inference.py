"""Run the detector on one image and format the results."""
from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms import functional as F

from python_pets.config import CLASSES, COCO_TO_OURS

MAX_SIDE = 1024  # downscale huge photos for speed


@torch.inference_mode()
def detect(model, device: str, mode: str, img: Image.Image, threshold: float) -> list[dict]:
    scale = min(1.0, MAX_SIDE / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    out = model([F.to_tensor(img).to(device)])[0]

    detections = []
    for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
        label, score = int(label), float(score)
        if mode == "coco":
            if label not in COCO_TO_OURS:
                continue
            label = COCO_TO_OURS[label]
        if score < threshold:
            continue
        x1, y1, x2, y2 = (float(v) / scale for v in box)
        detections.append({"label": CLASSES[label], "score": round(score, 4),
                           "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]})
    return detections


def summarize(detections: list[dict]) -> tuple[str, dict[str, int]]:
    """Return (verdict, counts) where verdict is "cat", "dog", "both" or "none"."""
    counts = {c: sum(d["label"] == c for d in detections) for c in CLASSES[1:]}
    if not detections:
        verdict = "none"
    elif counts["cat"] and counts["dog"]:
        verdict = "both"
    else:
        verdict = detections[0]["label"]
    return verdict, counts
