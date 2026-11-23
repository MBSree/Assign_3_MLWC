"""
Dataset utilities with consistent, invertible normalisation.

Important design choice (fixes the bug you found):
- We compute a single scale factor from the *input* image only.
- Both input and target are divided by that same scale.
- After the network predicts, we multiply by the same scale before computing MSE.
  Consequently a perfect network yields MSE ≈ 0.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def complex_to_real_imag(H):
    """(N, 72, 14) complex → (N, 2, 72, 14) float32  (channel 0 = real, 1 = imag)"""
    return np.stack([np.real(H), np.imag(H)], axis=1).astype(np.float32)


class ChannelDataset(Dataset):
    def __init__(self, H_input, H_target):
        """
        H_input, H_target : complex arrays of shape (N, 72, 14)
        """
        x = complex_to_real_imag(H_input)   # (N, 2, 72, 14)
        y = complex_to_real_imag(H_target)

        # per-sample scale computed from the INPUT only (receiver does not know target)
        # shape (N, 1, 1, 1)
        self.scale = np.std(x, axis=(1,2,3), keepdims=True) + 1e-8
        self.x = torch.from_numpy(x / self.scale)
        self.y = torch.from_numpy(y / self.scale)
        self.scale = torch.from_numpy(self.scale.astype(np.float32))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.scale[idx]


def make_loaders(H_in_tr, H_tg_tr, H_in_va, H_tg_va, batch_size=64):
    tr = ChannelDataset(H_in_tr, H_tg_tr)
    va = ChannelDataset(H_in_va, H_tg_va)
    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def denormalize(pred, scale):
    """pred and scale are tensors; returns the prediction in the original scale."""
    return pred * scale
