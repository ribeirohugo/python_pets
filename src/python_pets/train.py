"""Fetch the dataset at DATASET_URL (see .env) into datasets/ and fine-tune the detector.

Usage:
    python -m python_pets.train                 # default: 10 epochs
    python -m python_pets.train --epochs 5 --batch-size 4
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from python_pets.config import DEFAULT_WEIGHTS
from python_pets.dataset import CatDogDataset, collate, find_pairs
from python_pets.download import resolve_dataset
from python_pets.model import build_model


@torch.no_grad()
def evaluate(model, loader, device, score_thresh=0.5):
    """Image-level accuracy of the top detection + mean IoU against the GT box."""
    model.eval()
    correct = total = 0
    ious = []
    for images, targets in loader:
        outputs = model([i.to(device) for i in images])
        for out, tgt in zip(outputs, targets):
            total += 1
            keep = out["scores"] >= score_thresh
            if keep.sum() == 0:
                ious.append(0.0)
                continue
            top_box = out["boxes"][keep][0:1].cpu()
            top_label = out["labels"][keep][0].item()
            iou = box_iou(top_box, tgt["boxes"])[0]
            best = iou.argmax().item()
            ious.append(iou[best].item())
            correct += int(top_label == tgt["labels"][best].item())
    return correct / max(total, 1), sum(ious) / max(len(ious), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--data", type=str, default=None,
                    help="Folder containing the images and .xml annotations (default: datasets/, fetched from DATASET_URL if empty)")
    ap.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--resume", action="store_true",
                    help="Continue an interrupted run from weights/last_checkpoint.pt")
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    images_dir, annotations_dir = resolve_dataset(args.data)
    print(f"Images at: {images_dir} | Annotations at: {annotations_dir}")
    pairs = find_pairs(images_dir, annotations_dir)
    if not pairs:
        raise SystemExit("No image/annotation pairs found - check the dataset folder.")
    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_split)
    train_ds = CatDogDataset(pairs[n_val:], train=True)
    val_ds = CatDogDataset(pairs[:n_val], train=False)
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate, persistent_workers=args.workers > 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = build_model().to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out.with_name("last_checkpoint.pt")
    best_acc, start_epoch = -1.0, 1

    if args.resume:
        if not ckpt_path.exists():
            raise SystemExit(f"No checkpoint to resume from ({ckpt_path}). Run without --resume.")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        best_acc, start_epoch = ckpt["best_acc"], ckpt["epoch"] + 1
        args.epochs = ckpt["total_epochs"]  # LR schedule was built for this many epochs
        print(f"Resuming from epoch {start_epoch} (best val accuracy so far: {best_acc:.3f})")
        if start_epoch > args.epochs:
            raise SystemExit("That run already finished all its epochs.")

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            t0, running = time.time(), 0.0
            for i, (images, targets) in enumerate(train_dl, 1):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.autocast(device_type=device, enabled=device == "cuda"):
                    loss = sum(model(images, targets).values())
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scale_before = scaler.get_scale()
                scaler.step(optim)
                scaler.update()
                # If the scaler skipped optim.step() (inf/NaN grads -> scale lowered),
                # don't advance the LR schedule either; avoids PyTorch's ordering warning.
                if scaler.get_scale() >= scale_before:
                    sched.step()
                running += loss.item()
                if i % 50 == 0:
                    print(f"  epoch {epoch} step {i}/{len(train_dl)} loss {running / i:.4f}")

            acc, miou = evaluate(model, val_dl, device)
            print(f"Epoch {epoch}/{args.epochs}  loss {running / len(train_dl):.4f}  "
                  f"val_acc {acc:.3f}  val_mIoU {miou:.3f}  ({time.time() - t0:.0f}s)")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), args.out)
                print(f"  saved best model -> {args.out}")
            torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch, "total_epochs": args.epochs, "best_acc": best_acc}, ckpt_path)
    except KeyboardInterrupt:
        print(f"\nStopped by Ctrl+C during epoch {epoch} (that epoch's progress is discarded).")
        if args.out.exists():
            print(f"Best model so far is saved at {args.out} (val accuracy {best_acc:.3f}).")
        if ckpt_path.exists():
            print("Continue later with:  python -m python_pets.train --resume")
        else:
            print("No epoch finished yet, so there is nothing to resume - start again with: python -m python_pets.train")
        return

    print(f"Done. Best val accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    main()
