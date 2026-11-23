"""
Conventional baselines for Task 2.

- LS + 2-D interpolation (done in channel.py)
- Ideal MMSE      : pilot-only Wiener filter, R = sample covariance of PERFECT channels
- Reduced-rank MMSE (ALMMSE-style): same filter, R truncated to its strongest eigenmodes
- Estimated MMSE  : same filter, R learned only from NOISY LS+interp channels
                    (no perfect-channel knowledge)
"""

import numpy as np


def mse(H_hat, H_true):
    """Normalised MSE between estimated and true channel grids."""
    return np.mean(np.abs(H_hat - H_true) ** 2) / (np.mean(np.abs(H_true) ** 2) + 1e-12)


def estimate_correlation(H_train, max_samples=5000):
    """R = E[h h^H] from samples (n, N_SC, N_SYM) -> (N, N), N = N_SC*N_SYM.
    Note the convention: X.T @ conj(X) gives E[h h^H]."""
    n = min(H_train.shape[0], max_samples)
    X = H_train[:n].reshape(n, -1)
    return (X.T @ X.conj()) / n


def mmse_estimate(H_ls, mask, R, snr_db):
    """Pilot-only linear MMSE:  h_hat = A y_p,  A = R_dp (R_pp + sigma^2 I)^-1"""
    n_sc, n_sym = mask.shape
    N = n_sc * n_sym
    n = H_ls.shape[0]
    idx = np.flatnonzero(mask.ravel())
    P = len(idx)
    sigma2 = 10 ** (-snr_db / 10.0)

    R_pp = R[np.ix_(idx, idx)]
    R_dp = R[:, idx]
    try:
        M = R_pp + sigma2 * np.eye(P)
        A = np.linalg.solve(M.T, R_dp.T).T          # == R_dp @ inv(M)
    except np.linalg.LinAlgError:
        A = np.zeros((N, P), dtype=np.complex128)

    Yp = H_ls.reshape(n, N)[:, idx]
    return (Yp @ A.T).reshape(n, n_sc, n_sym)


def reduced_rank_mmse(H_ls, mask, R, snr_db, rank=4):
    """Reduced-rank (approximate) MMSE: keep only the `rank` strongest
    eigenmodes of R, then apply the pilot-only Wiener filter.
    The simulated channel is low-rank (about 6-13 significant eigenvalues),
    so a small fixed rank gives a visibly different curve from full MMSE."""
    try:
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)[-rank:]
        R_rr = (eigvecs[:, idx] * eigvals[idx]) @ eigvecs[:, idx].conj().T
    except np.linalg.LinAlgError:
        R_rr = R
    return mmse_estimate(H_ls, mask, R_rr, snr_db)


def estimated_mmse(H_ls, mask, H_noisy_interp_train, snr_db):
    """'Estimated MMSE': R learned only from noisy LS+interp training channels."""
    return mmse_estimate(H_ls, mask, estimate_correlation(H_noisy_interp_train), snr_db)


def approx_mmse(H_ls, mask, H_train_for_R=None, snr_db=12.0, reduced_rank=False, rank=4):
    """Convenience wrapper (kept for backward compatibility)."""
    N = H_ls.shape[1] * H_ls.shape[2]
    if H_train_for_R is not None and H_train_for_R.shape[0] >= 100:
        R = estimate_correlation(H_train_for_R)
    else:
        R = np.eye(N, dtype=np.complex128)
    if reduced_rank:
        return reduced_rank_mmse(H_ls, mask, R, snr_db, rank=rank)
    return mmse_estimate(H_ls, mask, R, snr_db)