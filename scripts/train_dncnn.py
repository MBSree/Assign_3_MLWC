"""Train DnCNN on top of a frozen pre-trained SRCNN (paper's two-stage procedure)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse, numpy as np, torch, torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from utils.config import (DEVICE, SEED, BATCH_SIZE, LR, MAX_EPOCHS_DN,
                          N_TRAIN, N_VAL, N_SC, N_SYM, CKPT_DIR, RESULTS_DIR)
from utils.channel import make_dataset
from utils.dataset import make_loaders
from models.channelnet import ChannelNet


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def run_epoch(model, loader, optim, criterion, train=True):
    """Returns true-scale MSE per complex element of the full ChannelNet output."""
    model.train(train)
    model.srcnn.eval()                      # SRCNN is frozen (stage-2 training)
    tot_se, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y, s in loader:
            x, y, s = x.to(DEVICE), y.to(DEVICE), s.to(DEVICE)
            if train:
                optim.zero_grad()
            pred = model(x, stage="full")
            loss = criterion(pred, y)
            if train:
                loss.backward(); optim.step()
            tot_se += (((pred - y) * s) ** 2).sum().item()
            n += x.size(0)
    return tot_se / (n * N_SC * N_SYM)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="VehA", choices=["VehA", "SUI5"])
    p.add_argument("--snr", type=int, default=12)
    p.add_argument("--pilots", type=int, default=48)
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS_DN)
    a = p.parse_args()

    set_seed(SEED)
    tag = f"{a.model}_p{a.pilots}_snr{a.snr}"
    sr_path = CKPT_DIR / f"SRCNN_{tag}.pt"
    if not sr_path.exists():
        raise FileNotFoundError(f"Train SRCNN first: {sr_path}")
    print(f"Device: {DEVICE} | DnCNN | {tag}")

    Ht, _, Hi, _ = make_dataset(a.model, a.snr, a.pilots, N_TRAIN, seed=SEED)
    Hv, _, Hiv, _ = make_dataset(a.model, a.snr, a.pilots, N_VAL, seed=SEED + 1)
    train_loader, val_loader = make_loaders(Hi, Ht, Hiv, Hv, BATCH_SIZE)

    net = ChannelNet().to(DEVICE)
    net.srcnn.load_state_dict(torch.load(sr_path, map_location=DEVICE)["model"])
    for q in net.srcnn.parameters():
        q.requires_grad = False
    optim = Adam(net.dncnn.parameters(), lr=LR)
    crit = nn.MSELoss()
    ckpt = CKPT_DIR / f"ChannelNet_{tag}.pt"

    best, tr_h, va_h = 1e9, [], []
    for ep in range(1, a.epochs + 1):
        tr = run_epoch(net, train_loader, optim, crit, True)
        va = run_epoch(net, val_loader, optim, crit, False)
        tr_h.append(tr); va_h.append(va)
        if va < best:
            best = va
            torch.save({"model": net.state_dict(), "val": va, "args": vars(a)}, ckpt)
        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep:3d}/{a.epochs}  train MSE={tr:.5f}  val MSE={va:.5f}")

    np.savez(RESULTS_DIR / f"hist_DnCNN_{tag}.npz", train=tr_h, val=va_h)
    plt.figure(figsize=(6, 4))
    plt.semilogy(tr_h, label="train"); plt.semilogy(va_h, label="validation")
    plt.xlabel("Epoch"); plt.ylabel("MSE (true scale)"); plt.grid(alpha=0.3); plt.legend()
    plt.title(f"IR stage (DnCNN) - {a.model}, {a.pilots} pilots, {a.snr} dB")
    plt.savefig(RESULTS_DIR / f"loss_DnCNN_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Best val MSE={best:.5f} -> {ckpt}")


if __name__ == "__main__":
    main()