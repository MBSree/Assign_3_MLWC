"""
Task 1 – generate ground-truth channel images and pilot patterns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from utils.config import RESULTS_DIR, SEED
from utils.channel import generate_H, pilot_mask, make_dataset

np.random.seed(SEED)
RESULTS_DIR.mkdir(exist_ok=True)


def save_channel(H, title, fname):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for a, part, name in zip(ax, (np.real, np.imag), ("Real", "Imag")):
        im = a.imshow(part(H), aspect="auto", origin="lower", cmap="viridis")
        a.set_title(name); a.set_xlabel("OFDM symbol"); a.set_ylabel("Subcarrier")
        plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved", fname)


def main():
    for model in ["VehA", "SUI5"]:
        H, _ = generate_H(model, n=1)
        save_channel(H[0], f"Ground-truth – {model}", f"fig2_gt_{model}.png")

    for npil in [16, 24, 36, 48]:
        m = pilot_mask(npil)
        plt.figure(figsize=(5, 3.5))
        plt.imshow(m.T, aspect="auto", origin="lower", cmap="gray_r")
        plt.xlabel("Subcarrier"); plt.ylabel("OFDM symbol")
        plt.title(f"Lattice pilots – {npil} (actual={m.sum()})")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"pilots_{npil}.png", dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved pilots_{npil}.png  (count={m.sum()})")

    # example LS vs perfect
    Ht, Hls, Hi, _ = make_dataset("VehA", 12, 48, 1, seed=SEED)
    save_channel(Ht[0],  "Perfect (VehA)",          "ex_perfect.png")
    save_channel(Hls[0], "LS at pilots only",       "ex_ls_sparse.png")
    save_channel(Hi[0],  "LS + 2-D interpolation",  "ex_ls_interp.png")


if __name__ == "__main__":
    main()
