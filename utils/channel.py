"""
Channel generation, lattice pilots and 2-D interpolation.
All functions are fully vectorised where possible.
"""

import numpy as np
from scipy.interpolate import griddata, NearestNDInterpolator
from utils.config import (
    T_TOTAL,
    N_SC, N_SYM, FC, UE_SPEED, SC_SPACING,
    VEHA_DELAYS_NS, VEHA_POWERS_DB,
    SUI5_DELAYS_NS, SUI5_POWERS_DB, SEED
)

rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------
# Power delay profiles
# ------------------------------------------------------------------
def get_pdp(model: str):
    if model.upper() == "VEHA":
        delays = np.asarray(VEHA_DELAYS_NS, dtype=np.float64) * 1e-9
        p_db   = np.asarray(VEHA_POWERS_DB, dtype=np.float64)
    elif model.upper() == "SUI5":
        delays = np.asarray(SUI5_DELAYS_NS, dtype=np.float64) * 1e-9
        p_db   = np.asarray(SUI5_POWERS_DB, dtype=np.float64)
    else:
        raise ValueError(model)
    p = 10 ** (p_db / 10.0)
    p /= p.sum()
    return delays, p


# ------------------------------------------------------------------
# Simple Jakes-like fading (sum of sinusoids)
# ------------------------------------------------------------------
def _fading(n_real, n_taps, n_sym, fd, fs):
    n_osc = 12
    theta = rng.uniform(0, 2*np.pi, size=(n_real, n_taps, n_osc))
    phi   = rng.uniform(0, 2*np.pi, size=(n_real, n_taps, n_osc))
    alpha = np.pi * (np.arange(1, n_osc+1) / (n_osc+1))
    f_d   = fd * np.cos(alpha)                     # (n_osc,)
    t     = np.arange(n_sym) / fs                  # (n_sym,)

    # phase: (n_real, n_taps, n_osc, n_sym)
    phase = 2*np.pi * f_d[None,None,:,None] * t[None,None,None,:] + phi[...,None] + theta[...,None]
    h = (np.cos(phase) + 1j*np.sin(phase)).sum(axis=2) / np.sqrt(n_osc)
    return h   # (n_real, n_taps, n_sym)


def generate_H(model: str, snr_db=None, n=1):
    """
    Returns
        H_true  : (n, N_SC, N_SYM) complex, unit average power
        H_noisy : same shape (only if snr_db is given)
    """
    delays, powers = get_pdp(model)
    n_taps = len(delays)
    fd = (UE_SPEED/3.6) * FC / 3e8
    fs = 1.0 / T_TOTAL                     # one sample per OFDM symbol

    h_taps = _fading(n, n_taps, N_SYM, fd, fs)
    h_taps *= np.sqrt(powers)[None, :, None]

    freqs = (np.arange(N_SC) - N_SC//2) * SC_SPACING
    exp_term = np.exp(-1j * 2*np.pi * freqs[:,None] * delays[None,:])  # (N_SC, n_taps)

    H = np.einsum("rtl,st->rsl", h_taps, exp_term)   # (n, N_SC, N_SYM)
    H /= np.sqrt(np.mean(np.abs(H)**2) + 1e-12)

    if snr_db is None:
        return H, None

    sigma2 = 10 ** (-snr_db / 10.0)
    noise = np.sqrt(sigma2/2) * (rng.standard_normal(H.shape) + 1j*rng.standard_normal(H.shape))
    return H, H + noise


# ------------------------------------------------------------------
# Lattice pilot masks – regular patterns, exact counts
# ------------------------------------------------------------------
def pilot_mask(n_pilots: int):
    """
    Regular lattice pilot patterns with exact counts.
    Designed to be as uniform as possible across time and frequency.
    """
    mask = np.zeros((N_SC, N_SYM), dtype=bool)

    if n_pilots == 48:
        # Dense lattice: 4 time positions, staggered in frequency
        for sc in range(0, N_SC, 6):
            mask[sc, [0, 7]] = True
        for sc in range(3, N_SC, 6):
            mask[sc, [4, 11]] = True
        # remaining to reach exactly 48
        for sc in range(1, N_SC, 6):
            for sym in (2, 9):
                if mask.sum() < 48:
                    mask[sc, sym] = True
    elif n_pilots == 36:
        for sc in range(0, N_SC, 6):
            mask[sc, [0, 7]] = True
        for sc in range(3, N_SC, 6):
            mask[sc, 4] = True
    elif n_pilots == 24:
        for sc in range(0, N_SC, 9):
            mask[sc, [0, 7]] = True
        for sc in range(4, N_SC, 9):
            mask[sc, 4] = True
    elif n_pilots == 16:
        # Clean diamond / rectangular lattice: 8 + 8
        # Suggested clean pattern: every 9th subcarrier on two symbols
        mask[0:N_SC:9, 0] = True
        mask[4:N_SC:9, 7] = True
        # If still short, add a third time position cleanly
        if mask.sum() < 16:
            mask[2:N_SC:9, 4] = True
    else:
        raise ValueError(f"Unsupported pilot count {n_pilots}")

    # Exact count
    idx = np.argwhere(mask)
    if len(idx) > n_pilots:
        mask[tuple(idx[n_pilots:].T)] = False
    elif len(idx) < n_pilots:
        free = np.argwhere(~mask)
        # add remaining in a deterministic order
        for r, c in free:
            if mask.sum() >= n_pilots:
                break
            mask[r, c] = True

    assert mask.sum() == n_pilots, f"Got {mask.sum()} pilots, expected {n_pilots}"
    return mask


# ------------------------------------------------------------------
# LS at pilots + improved 2-D interpolation (no zero-fill)
# ------------------------------------------------------------------
def ls_at_pilots(H_noisy, mask):
    H_ls = np.zeros_like(H_noisy)
    H_ls[:, mask] = H_noisy[:, mask]
    return H_ls


def interpolate_channel(H_ls, method="linear"):
    """
    2-D interpolation of the sparse LS grid.
    Uses linear interpolation inside the convex hull and nearest-neighbour
    extrapolation outside so that edge symbols / subcarriers are not zeroed.
    """
    n, n_sc, n_sym = H_ls.shape
    H_out = np.zeros_like(H_ls)

    # fixed grid
    sc_g, sym_g = np.meshgrid(np.arange(n_sc), np.arange(n_sym), indexing="ij")
    pts_full = np.column_stack([sc_g.ravel(), sym_g.ravel()])

    for i in range(n):
        pos = np.argwhere(np.abs(H_ls[i]) > 1e-12)
        if len(pos) < 3:
            continue
        for real_imag, extract in enumerate((np.real, np.imag)):
            vals = extract(H_ls[i])[pos[:,0], pos[:,1]]
            # linear inside hull
            interp = griddata(pos, vals, pts_full, method=method, fill_value=np.nan)
            # nearest for the NaNs (outside hull)
            nan_mask = np.isnan(interp)
            if nan_mask.any():
                nn = NearestNDInterpolator(pos, vals)
                interp[nan_mask] = nn(pts_full[nan_mask])
            if real_imag == 0:
                H_out[i].real = interp.reshape(n_sc, n_sym)
            else:
                H_out[i].imag = interp.reshape(n_sc, n_sym)
    return H_out


def make_dataset(model, snr_db, n_pilots, n_samples, seed=None):
    """Convenience wrapper used by training / evaluation scripts."""
    if seed is not None:
        global rng
        rng = np.random.default_rng(seed)
    H_true, H_noisy = generate_H(model, snr_db, n_samples)
    mask = pilot_mask(n_pilots)
    H_ls   = ls_at_pilots(H_noisy, mask)
    H_interp = interpolate_channel(H_ls)
    return H_true, H_ls, H_interp, mask
