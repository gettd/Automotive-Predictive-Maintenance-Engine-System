#!/usr/bin/env python3
"""
Synthetic time-series generator from tabular snapshots using:
1) Gaussian copula for static (same-timestep) feature relationships,
2) Feature-specific AR(1) small-step changes for temporal coherence,
3) Optional failure event with precursor (drift) + noise, and condition lock.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from path_params import *

# path
SEED_CSV = SEED_DATASET 
OUTPUT_DIR = SYNTHETIC_OUTPUT_DIR     
N_SEQUENCES = 40                        # how many sequences to generate
SEQ_LEN_RANGE = (6000, 12000)           # random length in seconds for each sequence (min, max)
SAMPLING_RATE_HZ = 1                    # 1 Hz -> 1 row per second
RANDOM_SEED = 42

FEATURE_COLS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]

TARGET_COL = "Engine Condition"         # will be generated here

# Temporal small-step dynamics (feature-specific) 
# AR(1): delta_t = alpha * delta_{t-1} + sigma * epsilon
ALPHA = {              # temporal persistence by feature (0..1)
    "Engine rpm":       0.6,
    "Lub oil pressure": 0.4,
    "Fuel pressure":    0.4,
    "Coolant pressure": 0.5,
    "lub oil temp":     0.85,
    "Coolant temp":     0.9,
}
SIGMA = {              # base step size (feature-specific) ~ units per second
    "Engine rpm":       15.0,
    "Lub oil pressure": 0.02,
    "Fuel pressure":    0.02,
    "Coolant pressure": 0.015,
    "lub oil temp":     0.03,
    "Coolant temp":     0.04,
}

# Feature ranges (domain knowledge)
FEATURE_RANGES = {
    "Engine rpm": (800, 7000),              # idle to redline
    "Lub oil pressure": (25, 65),           # PSI
    "Fuel pressure": (30, 60),              # PSI
    "Coolant pressure": (12, 16),           # PSI
    "lub oil temp": (90, 105),              # °C
    "Coolant temp": (90, 105),              # °C
}

# delta magnitudes
DELTA_SCALE = {
    "Engine rpm": 50,          # fast-changing
    "Lub oil pressure": 0.5,   # slow
    "Fuel pressure": 0.5,
    "Coolant pressure": 0.05,
    "lub oil temp": 0.05,
    "Coolant temp": 0.05,
}

# Hard clipping to keep values realistic
CLIP_EXPAND_FRAC = 0.10   # allow +/-10% outside observed range

# fail event
P_WILL_OVERHEAT = 0.6
OVERHEAT_START_FRAC = (0.05, 0.95)      # uniform fraction of the sequence
PRECURSOR_LEN_RANGE = (60, 1800)        # seconds of early warning signs (1–30 min)
OVERHEAT_DURATION_RANGE = (20, 120)     # time to reach full failure (seconds)
OVERHEAT_COOLANT_TEMP = (114, 124)      # plateau range (°C) once overheated

# Magnitudes applied during precursor + failure 
PRECURSOR = {
    "Coolant temp":     {"drift": (0.01, 0.08)},
    "lub oil temp":     {"drift": (0.005, 0.03)},
    "Coolant pressure": {"mult":  (0.0, 0.03)},
    "Lub oil pressure": {"mult":  (-0.04, -0.01)},
    "Fuel pressure":    {"mult":  (-0.02, -0.005)},
    "Engine rpm":       {"jitter_sigma": (10, 60)},
}

FAILURE = {
    "Coolant temp":     {"plateau": OVERHEAT_COOLANT_TEMP},
    "lub oil temp":     {"drift": (0.01, 0.05)},
    "Coolant pressure": {"mult":  (0.02, 0.05)},
    "Lub oil pressure": {"mult":  (-0.10, -0.03)},
    "Fuel pressure":    {"mult":  (-0.06, -0.02)},
    "Engine rpm":       {"jitter_sigma": (20, 80), "mean_shift_perc": (-0.15, -0.02)},
}

# bounded update
def bounded_update(prev_val, feature, failure_drift=False):
    """Update feature with small delta, respecting ranges."""
    low, high = FEATURE_RANGES[feature]
    scale = DELTA_SCALE[feature]

    delta = np.random.normal(0, scale)

    if failure_drift:
        if feature == "Lub oil pressure":
            delta -= abs(np.random.normal(0.1, 0.05))  # trending downward
        elif feature == "Coolant temp":
            delta += abs(np.random.normal(0.1, 0.05))  # trending upward
        elif feature == "lub oil temp":
            delta += abs(np.random.normal(0.08, 0.03))
        elif feature == "Fuel pressure":
            delta -= abs(np.random.normal(0.05, 0.02))
        elif feature == "Coolant pressure":
            delta += abs(np.random.normal(0.02, 0.01))

    new_val = prev_val + delta
    return np.clip(new_val, low, high)

#copula fit/sample
def _empirical_cdf_vals(x):
    return np.sort(x.astype(float))

def _inverse_ecdf(sorted_vals, u):
    if u <= 0: return sorted_vals[0]
    if u >= 1: return sorted_vals[-1]
    p = u * (len(sorted_vals) - 1)
    i = int(np.floor(p))
    frac = p - i
    if i >= len(sorted_vals) - 1:
        return sorted_vals[-1]
    return sorted_vals[i] * (1 - frac) + sorted_vals[i + 1] * frac

def fit_gaussian_copula(df, feature_cols):
    X = df[feature_cols].to_numpy(dtype=float)
    n, d = X.shape
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    U = ranks / (n + 1.0)
    Z = norm.ppf(U)
    Sigma = np.corrcoef(Z, rowvar=False)

    inv_marginals = {}
    for j, col in enumerate(feature_cols):
        inv_marginals[col] = _empirical_cdf_vals(df[col].values)

    mins = df[feature_cols].min()
    maxs = df[feature_cols].max()

    return {"Sigma": Sigma,
            "inv_marginals": inv_marginals,
            "mins": mins.to_dict(),
            "maxs": maxs.to_dict()}

def sample_from_copula(model, feature_cols, n_samples=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    d = len(feature_cols)
    Sigma = np.array(model["Sigma"])
    Z = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma, size=n_samples)
    U = norm.cdf(Z)

    X_out = np.zeros_like(U)
    for j, col in enumerate(feature_cols):
        inv_vals = model["inv_marginals"][col]
        for i in range(n_samples):
            X_out[i, j] = _inverse_ecdf(inv_vals, U[i, j])
    return X_out

#temporal simulation
def simulate_sequence_from_seed(seed_row, model, seq_len, rng):
    feature_cols = list(model["mins"].keys())
    X = np.zeros((seq_len, len(feature_cols)), dtype=float)
    X[0, :] = seed_row

    deltas = np.zeros(len(feature_cols), dtype=float)
    lower = np.array([model["mins"][c] for c in feature_cols], dtype=float)
    upper = np.array([model["maxs"][c] for c in feature_cols], dtype=float)
    span = upper - lower
    lower_clip = lower - CLIP_EXPAND_FRAC * span
    upper_clip = upper + CLIP_EXPAND_FRAC * span

    will_overheat = rng.random() < P_WILL_OVERHEAT
    overheat_start = None
    precursor_start = None
    precursor_end = None
    overheat_duration = None

    if will_overheat:
        overheat_start = rng.integers(int(seq_len * OVERHEAT_START_FRAC[0]),
                                      max(int(seq_len * OVERHEAT_START_FRAC[1]), int(seq_len * OVERHEAT_START_FRAC[0]) + 1))
        precursor_len = int(rng.integers(PRECURSOR_LEN_RANGE[0], PRECURSOR_LEN_RANGE[1]))
        precursor_start = max(0, overheat_start - precursor_len)
        precursor_end = overheat_start
        overheat_duration = int(rng.integers(OVERHEAT_DURATION_RANGE[0], OVERHEAT_DURATION_RANGE[1]))

    def urange(a, b):
        return rng.uniform(a, b)

    prec_cfg, fail_cfg = {}, {}
    for k, v in PRECURSOR.items():
        if "drift" in v:
            prec_cfg[(k, "drift")] = urange(*v["drift"])
        if "mult" in v:
            prec_cfg[(k, "mult")] = urange(*v["mult"])
        if "jitter_sigma" in v:
            prec_cfg[(k, "jitter_sigma")] = urange(*v["jitter_sigma"])
    for k, v in FAILURE.items():
        if "drift" in v:
            fail_cfg[(k, "drift")] = urange(*v["drift"])
        if "mult" in v:
            fail_cfg[(k, "mult")] = urange(*v["mult"])
        if "jitter_sigma" in v:
            fail_cfg[(k, "jitter_sigma")] = urange(*v["jitter_sigma"])
        if "mean_shift_perc" in v:
            fail_cfg[(k, "mean_shift_perc")] = urange(*v["mean_shift_perc"])
        if "plateau" in v:
            fail_cfg[(k, "plateau")] = (min(v["plateau"]), max(v["plateau"]))

    col_idx = {c: i for i, c in enumerate(feature_cols)}
    engine_condition = np.zeros(seq_len, dtype=float)

    for t in range(1, seq_len):
        x_prev = X[t - 1, :].copy()
        x_new = x_prev.copy()

        for j, col in enumerate(feature_cols):
            a = ALPHA.get(col, 0.7)
            s = SIGMA.get(col, 0.05)
            deltas[j] = a * deltas[j] + s * rng.normal()
            x_new[j] = x_prev[j] + deltas[j]

        if will_overheat and (precursor_start is not None) and (precursor_start <= t < precursor_end):
            ramp_pos = (t - precursor_start + 1) / max(1, (precursor_end - precursor_start))
            shape = 1.6
            w = ramp_pos ** shape

            if ("Coolant temp", "drift") in prec_cfg:
                X_c = col_idx["Coolant temp"]
                x_new[X_c] += prec_cfg[("Coolant temp", "drift")] * (1.0 + 1.5 * w) + rng.normal(0, 0.05)
            if ("lub oil temp", "drift") in prec_cfg:
                X_o = col_idx["lub oil temp"]
                x_new[X_o] += prec_cfg[("lub oil temp", "drift")] * (1.0 + 1.2 * w) + rng.normal(0, 0.03)
            if ("Coolant pressure", "mult") in prec_cfg:
                X_cp = col_idx["Coolant pressure"]
                x_new[X_cp] *= (1.0 + prec_cfg[("Coolant pressure", "mult")] * w) + rng.normal(0, 0.002)
            if ("Lub oil pressure", "mult") in prec_cfg:
                X_lp = col_idx["Lub oil pressure"]
                x_new[X_lp] *= (1.0 + prec_cfg[("Lub oil pressure", "mult")] * w) + rng.normal(0, 0.002)
            if ("Fuel pressure", "mult") in prec_cfg:
                X_fp = col_idx["Fuel pressure"]
                x_new[X_fp] *= (1.0 + prec_cfg[("Fuel pressure", "mult")] * w) + rng.normal(0, 0.002)
            if ("Engine rpm", "jitter_sigma") in prec_cfg:
                X_r = col_idx["Engine rpm"]
                x_new[X_r] += rng.normal(0, prec_cfg[("Engine rpm", "jitter_sigma")] * (0.5 + w))
            prec_len = max(1, (precursor_end - precursor_start))
            ramp_pos = (t - precursor_start + 1) / prec_len  # in (0,1]
            deg_shape = 1.6
            deg_frac = float(min(1.0, ramp_pos ** deg_shape))
            engine_condition[t] = deg_frac
            


        if will_overheat and (overheat_start is not None) and (t >= overheat_start):
            engine_condition[t] = 1
            prog = min(1.0, (t - overheat_start) / overheat_duration) if overheat_duration else 0.0

            if ("Coolant temp", "plateau") in fail_cfg:
                X_c = col_idx["Coolant temp"]
                lo, hi = fail_cfg[("Coolant temp", "plateau")]
                x_new[X_c] = rng.uniform(lo, hi)
            if ("lub oil temp", "drift") in fail_cfg:
                X_o = col_idx["lub oil temp"]
                x_new[X_o] += fail_cfg[("lub oil temp", "drift")] * (1.0 + 2.0 * prog) + rng.normal(0, 0.05)
            if ("Coolant pressure", "mult") in fail_cfg:
                X_cp = col_idx["Coolant pressure"]
                x_new[X_cp] *= (1.0 + fail_cfg[("Coolant pressure", "mult")] * (0.5 + 1.5 * prog)) + rng.normal(0, 0.003)
            if ("Lub oil pressure", "mult") in fail_cfg:
                X_lp = col_idx["Lub oil pressure"]
                x_new[X_lp] *= (1.0 + fail_cfg[("Lub oil pressure", "mult")] * (0.5 + 1.5 * prog)) + rng.normal(0, 0.003)
            if ("Fuel pressure", "mult") in fail_cfg:
                X_fp = col_idx["Fuel pressure"]
                x_new[X_fp] *= (1.0 + fail_cfg[("Fuel pressure", "mult")] * (0.5 + 1.5 * prog)) + rng.normal(0, 0.003)
            if ("Engine rpm", "jitter_sigma") in fail_cfg:
                X_r = col_idx["Engine rpm"]
                jitter = rng.normal(0, fail_cfg[("Engine rpm", "jitter_sigma")] * (0.7 + 1.3 * prog))
                mean_shift_perc = fail_cfg.get(("Engine rpm", "mean_shift_perc"), -0.08)
                shift = x_prev[X_r] * (mean_shift_perc if isinstance(mean_shift_perc, float) else rng.uniform(*FAILURE["Engine rpm"]["mean_shift_perc"]))
                x_new[X_r] = max(0.0, x_prev[X_r] + shift * prog + jitter)

        if t > 0:
            engine_condition[t] = max(engine_condition[t], engine_condition[t - 1])

        x_new = np.array([
            np.clip(val, FEATURE_RANGES[col][0], FEATURE_RANGES[col][1])
            for val, col in zip(x_new, feature_cols)
        ])

        X[t, :] = x_new

    return X, engine_condition


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.read_csv(SEED_CSV)
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column in seed data: {col}")

    copula_model = fit_gaussian_copula(df, FEATURE_COLS)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(1, N_SEQUENCES + 1):
        seq_len = int(rng.integers(SEQ_LEN_RANGE[0], SEQ_LEN_RANGE[1]))
        seed_row = sample_from_copula(copula_model, FEATURE_COLS, n_samples=1, rng=rng)[0]
        seed_row = np.array([np.clip(val, FEATURE_RANGES[col][0], FEATURE_RANGES[col][1])
                     for val, col in zip(seed_row, FEATURE_COLS)])

        

        X, y = simulate_sequence_from_seed(seed_row, copula_model, seq_len, rng)

        t = np.arange(seq_len, dtype=float) / float(SAMPLING_RATE_HZ)
        out_df = pd.DataFrame(X, columns=FEATURE_COLS)
        out_df.insert(0, "Time", t)
        out_df[TARGET_COL] = y

        out_path = out_dir / f"synthetic_timeseries_{k}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} (len={seq_len})")

    print("Done.")

if __name__ == "__main__":
    main()
