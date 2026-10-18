"""5-day sprint F2: rotation-prediction self-supervision for shadow direction.

Pretext: predict 0/90/180/270 rotation (4-way). Compared with flip pretext and
ImageNet init. Uses the same 68/16/21 base-name split as the main method.

Usage:
    python train_direction_rotation.py --epochs 1      # smoke
    python train_direction_rotation.py --epochs 30     # full
"""
import argparse
import random
import time
from pathlib import Path

import env_setup  # noqa: F401

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from train_direction_net import split_by_base, IMG_SIZE, MEAN, STD

BASE = Path(__file__).resolve().parent
AISD = BASE / "AISD" / "AISD"
OUT = str(BASE / "checkpoints" / "direction_resnet18_rot_basis.pth")
LOG_DIR = BASE / "logs" / "sprint20261018"
device = "cuda" if torch.cuda.is_available() else "cpu"


class RotationDataset(torch.utils.data.Dataset):
    """Each image yields 4 samples: rotated by 0/90/180/270 degrees."""

    def __init__(self, files, train=True):
        self.files = files
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return 4 * len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx // 4]
        k = idx % 4
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if k == 1:
            img = img.transpose(Image.ROTATE_90)
        elif k == 2:
            img = img.transpose(Image.ROTATE_180)
        elif k == 3:
            img = img.transpose(Image.ROTATE_270)
        return self.tf(img), torch.tensor(k, dtype=torch.long)


def build_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 4)
    return model


@torch.no_grad()
def evaluate_acc(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15,
                    help="default 15; 4x samples/epoch vs flip 2x, so ~same optimizer steps as flip 30")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="", help="checkpoint suffix")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"direction_rotation_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    files = sorted(str(p) for p in (AISD / "Train412" / "shadow").glob("*.tif"))
    train_pool, test_files, _, n_te = split_by_base(files, 0.2, seed=0)
    tr_files, val_files, n_tr, n_va = split_by_base(train_pool, 0.2, seed=1)
    log("# rotation-prediction self-supervised pretraining (4-way)")
    log(f"# time={time.strftime('%Y-%m-%d %H:%M:%S')} device={device}")
    log(f"# split: train={len(tr_files)} ({n_tr} bases) val={len(val_files)} "
        f"({n_va} bases) test={len(test_files)} ({n_te} bases, NOT touched)")
    log(f"# epochs={args.epochs} batch={args.batch_size} lr={args.lr} seed={args.seed}")
    log(f"# steps/epoch={len(tr_files)*4//args.batch_size} "
        f"(flip: {len(tr_files)*2//args.batch_size}; matched by 15 vs 30 epochs)")

    train_loader = torch.utils.data.DataLoader(
        RotationDataset(tr_files, train=True), batch_size=args.batch_size,
        shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        RotationDataset(val_files, train=False), batch_size=args.batch_size,
        shuffle=False, num_workers=0)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = 0.0
    suffix = f"_{args.tag}" if args.tag else ""
    if args.seed != 0:
        suffix += f"_s{args.seed}"
    out_path = str(BASE / "checkpoints" / f"direction_resnet18_rot_basis{suffix}.pth")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        acc = evaluate_acc(model, val_loader)
        log(f"Epoch {epoch:3d}/{args.epochs} loss={np.mean(losses):.4f} "
            f"val_acc={acc:.4f} ({time.time()-t0:.0f}s)")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), out_path)
    log(f"# best val_acc={best:.4f}")
    log(f"# checkpoint -> {out_path}")
    log(f"# log -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
