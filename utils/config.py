"""
Configuration for ChannelNet (COM 837 Assignment 3)
"""

from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
CKPT_DIR    = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
for d in (DATA_DIR, CKPT_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# OFDM grid (paper / LTE-like)
# ------------------------------------------------------------------
N_SC  = 72
N_SYM = 14

# LTE numerology
SC_SPACING = 15e3          # 15 kHz
T_SYM      = 1.0 / SC_SPACING   # useful symbol duration ≈ 66.7 µs
# Approximate cyclic prefix (normal CP for 15 kHz is ~4.7 µs)
T_CP       = 4.7e-6
T_TOTAL    = T_SYM + T_CP  # total OFDM symbol time used for Doppler sampling

FC       = 2.1e9
UE_SPEED = 50              # km/h

# ------------------------------------------------------------------
# Pilots
# ------------------------------------------------------------------
PILOT_COUNTS   = [16, 24, 36, 48]
DEFAULT_PILOTS = 48

# ------------------------------------------------------------------
# Channel models
# ------------------------------------------------------------------
VEHA_DELAYS_NS = [0, 310, 710, 1090, 1730, 2510]
VEHA_POWERS_DB = [0.0, -1.0, -9.0, -10.0, -15.0, -20.0]

SUI5_DELAYS_NS = [0, 4000, 10000]
SUI5_POWERS_DB = [0.0, -5.0, -10.0]

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
SNR_TRAIN = [12, 22]
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS_SR = 40
MAX_EPOCHS_DN = 30

# Reduced size for CPU.  Change these to 32000/4000/4000 if you have a GPU.
N_TRAIN = 2000
N_VAL   = 400
N_TEST  = 500

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
