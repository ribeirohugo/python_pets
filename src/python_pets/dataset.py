"""PyTorch dataset over Pascal-VOC image/annotation pairs."""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from python_pets.config import CLASSES

LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def find_pairs(images_dir: Path, annotations_dir: Path) -> list[tuple[Path, Path]]:
    """Match every Pascal-VOC xml annotation with its image."""
    images = {p.stem: p for p in images_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    pairs = []
    for xml in annotations_dir.rglob("*.xml"):
        img = images.get(xml.stem)
        if img is not None:
            pairs.append((img, xml))
    return sorted(pairs)


def parse_voc(xml_path: Path) -> tuple[list[list[float]], list[int]]:
    tree = ET.parse(xml_path)
    boxes, labels = [], []
    for obj in tree.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        if name not in LABEL_TO_ID or name == "background":
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(LABEL_TO_ID[name])
    return boxes, labels


class CatDogDataset(Dataset):
    def __init__(self, pairs, train: bool):
        self.items = []
        for img, xml in pairs:
            boxes, labels = parse_voc(xml)
            if boxes:
                self.items.append((img, boxes, labels))
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, boxes, labels = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.train:
            if random.random() < 0.5:  # horizontal flip
                img = F.hflip(img)
                w = img.width
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            img = F.adjust_brightness(img, random.uniform(0.75, 1.25))
            img = F.adjust_contrast(img, random.uniform(0.75, 1.25))
            img = F.adjust_saturation(img, random.uniform(0.75, 1.25))

        return F.to_tensor(img), {"boxes": boxes, "labels": labels}


def collate(batch):
    return tuple(zip(*batch))
