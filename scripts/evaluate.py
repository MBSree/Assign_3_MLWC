"""MSE vs SNR (Figs 3/4) and MSE vs number of pilots (Fig 5)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.config import DEVICE, SEED, CKPT_DIR, RESULTS_DIR, N_TEST, N_TRAIN
from utils.channel import make_dataset, pilot_mask
from utils.baselines import (mse, estimate_correlation, mmse_estimate,
                             reduced_rank_mmse)
from utils.dataset import complex_to_real_imag
from models.channelnet import ChannelNet


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def load_net(model, npil, snr):
    p = CKPT_DIR / f"ChannelNet_{model}_p{npil}_snr{snr}.pt"
    if not p.exists():
        return None
    net = ChannelNet().to(DEVICE)
    net.load_state_dict(torch.load(p, map_location=DEVICE)["model"])
    net.eval()
    return net


@torch.no_grad()
def predict(net, H_interp, bs=100):
    """Run ChannelNet and return a complex estimate in the original scale."""
    x = complex_to_real_imag(H_interp)
    scale = np.std(x, axis=(1, 2, 3), keepdims=True) + 1e-8
    xn = torch.from_numpy(x / scale)
    out = []
    for i in range(0, len(xn), bs):
        out.append(net(xn[i:i + bs].to(DEVICE), stage="full").cpu().numpy())
    pred = np.concatenate(out) * scale
    return pred[:, 0] + 1j * pred[:, 1]


def sweep_snr(model, snr_list, npil):
    keys = ["LS", "MMSE", "RedRank", "EstMMSE", "CN12", "CN22"]
    res = {k: [] for k in keys}
    nets = {s: load_net(model, npil, s) for s in (12, 22)}
    for s, n in nets.items():
        print(f"ChannelNet @ {s} dB: {'loaded' if n is not None else 'NOT FOUND'}")

    print("Estimating correlation from perfect channels (ideal MMSE / reduced-rank) ...")
    H_ref, _, _, _ = make_dataset(model, 30, npil, min(N_TRAIN, 2000), seed=SEED)
    R = estimate_correlation(H_ref)
    mask = pilot_mask(npil)

    for snr in tqdm(snr_list, desc=model):
        Ht, Hls, Hi, _ = make_dataset(model, snr, npil, N_TEST, seed=SEED + 1000 + int(snr))
        # noisy-only training set for the *estimated* MMSE (no perfect-channel knowledge)
        _, _, Hi_tr, _ = make_dataset(model, snr, npil, 1000, seed=SEED + 5000 + int(snr))
        res["LS"].append(mse(Hi, Ht))
        res["MMSE"].append(mse(mmse_estimate(Hls, mask, R, snr), Ht))
        res["RedRank"].append(mse(reduced_rank_mmse(Hls, mask, R, snr), Ht))
        res["EstMMSE"].append(mse(mmse_estimate(Hls, mask, estimate_correlation(Hi_tr), snr), Ht))
        for s, k in ((12, "CN12"), (22, "CN22")):
            res[k].append(mse(predict(nets[s], Hi), Ht) if nets[s] is not None else np.nan)
    return res


def plot_snr(snr, res, model, npil):
    plt.figure(figsize=(7.5, 5))
    plt.semilogy(snr, res["LS"], "o-", label="LS + interp")
    plt.semilogy(snr, res["MMSE"], "s-", label="Ideal MMSE (R from perfect channels)")
    plt.semilogy(snr, res["EstMMSE"], "v-", label="Estimated MMSE (R from noisy LS)")
    plt.semilogy(snr, res["RedRank"], "x-", label="Reduced-rank MMSE")
    if not np.all(np.isnan(res["CN12"])):
        plt.semilogy(snr, res["CN12"], "^-", label="ChannelNet - Deep Low-SNR (12 dB)")
    if not np.all(np.isnan(res["CN22"])):
        plt.semilogy(snr, res["CN22"], "d-", label="ChannelNet - Deep High-SNR (22 dB)")
    plt.xlabel("SNR (dB)"); plt.ylabel("MSE")
    plt.title(f"MSE vs SNR - {model}, {npil} pilots")
    plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8)
    f = RESULTS_DIR / f"MSE_SNR_{model}_p{npil}.png"
    plt.savefig(f, dpi=130, bbox_inches="tight"); plt.close()
    print("Saved", f)


def pilot_sweep(model, pilot_list, snr_fixed=20, net_snr=22):
    """Fig. 5 style. Uses a network trained for each pilot count if its checkpoint
    exists, otherwise falls back to the 48-pilot network (and prints a warning)."""
    res = {"LS": [], "MMSE": [], "EstMMSE": [], "CN": []}
    fallback = load_net(model, 48, net_snr)
    for npil in pilot_list:
        net = load_net(model, npil, net_snr)
        if net is None:
            net = fallback
            print(f"[{model}] no {npil}-pilot network -> using 48-pilot network (mismatched)")
        mask = pilot_mask(npil)
        H_ref, _, _, _ = make_dataset(model, 30, npil, 1500, seed=SEED)
        R = estimate_correlation(H_ref)
        Ht, Hls, Hi, _ = make_dataset(model, snr_fixed, npil, N_TEST, seed=SEED + 2000)
        _, _, Hi_tr, _ = make_dataset(model, snr_fixed, npil, 1000, seed=SEED + 6000)
        res["LS"].append(mse(Hi, Ht))
        res["MMSE"].append(mse(mmse_estimate(Hls, mask, R, snr_fixed), Ht))
        res["EstMMSE"].append(mse(mmse_estimate(Hls, mask, estimate_correlation(Hi_tr), snr_fixed), Ht))
        res["CN"].append(mse(predict(net, Hi), Ht) if net is not None else np.nan)

    plt.figure(figsize=(7, 5))
    plt.semilogy(pilot_list, res["LS"], "o-", label="LS + interp")
    plt.semilogy(pilot_list, res["MMSE"], "s-", label="Ideal MMSE")
    plt.semilogy(pilot_list, res["EstMMSE"], "v-", label="Estimated MMSE")
    if not np.all(np.isnan(res["CN"])):
        plt.semilogy(pilot_list, res["CN"], "d-", label=f"ChannelNet ({net_snr} dB)")
    plt.xlabel("Number of pilots"); plt.ylabel("MSE")
    plt.title(f"MSE vs #pilots - {model}, SNR={snr_fixed} dB")
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    f = RESULTS_DIR / f"MSE_pilots_{model}_snr{snr_fixed}.png"
    plt.savefig(f, dpi=130, bbox_inches="tight"); plt.close()
    print("Saved", f)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=["VehA", "SUI5", "both"])
    ap.add_argument("--pilots", type=int, default=48)
    a = ap.parse_args()
    set_seed(SEED)

    snr_list = list(range(0, 31, 5))
    models = ["VehA", "SUI5"] if a.model == "both" else [a.model]
    for m in models:
        print(f"\n=== {m} ===")
        res = sweep_snr(m, snr_list, a.pilots)
        plot_snr(snr_list, res, m, a.pilots)
        np.savez(RESULTS_DIR / f"mse_{m}_p{a.pilots}.npz", snr=snr_list, **res)
        pr = pilot_sweep(m, [16, 24, 36, 48], snr_fixed=20)
        np.savez(RESULTS_DIR / f"mse_pilots_{m}.npz", pilots=[16, 24, 36, 48], **pr)
    print("\nDone. Plots are in results/")


if __name__ == "__main__":
    main()