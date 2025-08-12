import os
import numpy as np
import pandas as pd
from pathlib import Path
from path_params import *
import math

# config
N_SEQUENCES = 40
SEQUENCE_LENGTH = 10000  # number of rows (timesteps in sec) per sequence
SAMPLING_RATE_HZ = 1     # 1Hz = 1 row = 1 second
RANDOM_SEED = 42
NOISE_LEVEL = 0.01

# simulate early warning signs (more realistic parameters)
def add_precursor_ramp_enhanced(df, start_idx, end_idx):
    ramp_len = end_idx - start_idx
    for i in range(start_idx, end_idx):
        if i < 0 or i >= len(df):
            continue
        ramp_progress = (i - start_idx + 1) / ramp_len

        # RPM jitter: realistic idle variability
        base_rpm = df.at[i, "Engine rpm"]
        rpm_jitter = np.random.normal(0, 30 * np.sqrt(ramp_progress))  # σ ≈ 30 rpm, increases with ramp
        df.at[i, "Engine rpm"] = base_rpm + rpm_jitter

        # Lub oil temp: gradual nonlinear rise (viscosity drop trend)
        df.at[i, "lub oil temp"] += np.random.normal(0, 0.3) + 1.0 + 3.5 * (ramp_progress ** 1.5)

        # Coolant temp: exponential warm-up trend (Haury & Volkering, 2011)
        A = np.random.uniform(6.0, 18.0)  # °C rise before overheat
        tau = np.random.uniform(30, 200)  # warm-up time constant
        t_rel = (i - start_idx + 1)
        coolant_increase = A * (1 - math.exp(-t_rel / tau))
        df.at[i, "Coolant temp"] += coolant_increase + np.random.normal(0, 0.5)

        # Coolant pressure: small steady increase
        df.at[i, "Coolant pressure"] *= 1 + 0.015 * ramp_progress + np.random.normal(0, 0.003)

        # Lub oil pressure: viscosity drop 5–15%
        drop_factor = np.random.uniform(0.05, 0.15)
        df.at[i, "Lub oil pressure"] *= 1 - drop_factor * ramp_progress + np.random.normal(0, 0.003)

        # Fuel pressure: slight drop 2–5%
        fuel_drop_factor = np.random.uniform(0.02, 0.05)
        df.at[i, "Fuel pressure"] *= 1 - fuel_drop_factor * ramp_progress + np.random.normal(0, 0.002)


# force coolant to overheat and change engine condition to 1 (failed), also degrade sensors
def inject_overheat_enhanced(df, start_idx, duration):
    for i in range(start_idx, len(df)):
        progress = min(1.0, (i - start_idx) / duration)

        # RPM drop with variability
        base_rpm = df.at[i, "Engine rpm"]
        rpm_shift = np.random.normal(-40 * progress, 30)
        df.at[i, "Engine rpm"] = max(0, base_rpm + rpm_shift)

        # Oil pressure drop to ~80% of nominal
        df.at[i, "Lub oil pressure"] *= 0.85 - 0.05 * progress + np.random.normal(0, 0.003)

        # Coolant temp: fixed overheated range (114–124°C from overheating studies)
        df.at[i, "Coolant temp"] = np.random.uniform(114, 124)

        # Coolant pressure: small increase from boiling/expansion
        df.at[i, "Coolant pressure"] *= 1.05 + 0.03 * progress + np.random.normal(0, 0.003)

        # Oil temp: further increase
        df.at[i, "lub oil temp"] += 1.0 + 2.5 * progress + np.random.normal(0, 0.3)

        # Fuel pressure drop
        df.at[i, "Fuel pressure"] *= 0.90 - 0.03 * progress + np.random.normal(0, 0.002)

        df.at[i, "Engine Condition"] = 1


# add noise to all features except engine condition and time
def add_feature_noise(df, ref_df, noise_frac=0.01):
    for col in df.columns:
        if col not in ["Time", "Engine Condition"]:
            sigma = ref_df[col].std() * noise_frac
            df[col] += np.random.normal(0, sigma, size=len(df))


# generate n sequences that overheat randomly
def generate_sequences(source_df, n_sequences, sampling_rate, noise_level, seed=None):
    np.random.seed(seed)
    sequences = []

    feature_cols = [
        "Engine rpm", "Lub oil pressure", "Fuel pressure",
        "Coolant pressure", "lub oil temp", "Coolant temp"
    ]

    for seq_idx in range(n_sequences):
        seq_len = np.random.randint(5000, 12000)
        df = source_df.sample(n=seq_len, replace=True).reset_index(drop=True)
        df.insert(0, "Time", np.arange(seq_len) / sampling_rate)
        df["Engine Condition"] = 0

        will_overheat = np.random.choice([True, False], p=[0.6, 0.4])

        if will_overheat:
            overheat_start = np.random.randint(int(seq_len * 0.1), int(seq_len * 0.95))
            overheat_duration = np.random.randint(10, 60)
            ramp_len = np.random.randint(60, 1200)

            precursor_start = max(0, overheat_start - ramp_len)
            precursor_end = overheat_start

            add_precursor_ramp_enhanced(df, precursor_start, precursor_end)
            inject_overheat_enhanced(df, start_idx=overheat_start, duration=overheat_duration)

        add_feature_noise(df, source_df, noise_frac=noise_level)
        ordered_cols = ["Time"] + feature_cols + ["Engine Condition"]
        sequences.append(df[ordered_cols])

    return sequences


def main():
    input_path = Path(SEED_DATASET)
    output_dir = Path(SYNTHETIC_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading input data: {input_path}")
    source_df = pd.read_csv(input_path)

    print(f"Generating {N_SEQUENCES} synthetic time-series datasets...")
    datasets = generate_sequences(
        source_df=source_df,
        n_sequences=N_SEQUENCES,
        sampling_rate=SAMPLING_RATE_HZ,
        noise_level=NOISE_LEVEL,
        seed=RANDOM_SEED,
    )

    for i, df in enumerate(datasets, 1):
        out_file = output_dir / f"synthetic_timeseries_{i}.csv"
        df.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

    print("All datasets generated successfully!")


if __name__ == "__main__":
    main()
