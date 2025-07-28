import os
import numpy as np
import pandas as pd
from pathlib import Path
from path_params import *

#config
N_SEQUENCES = 40                      
SEQUENCE_LENGTH = 10000 # number of rows (timesteps) per sequence
SAMPLING_RATE_HZ = 1    # 1Hz = 1 row = 1 second
RANDOM_SEED = 42
NOISE_LEVEL = 0.01

#simulate early warning signs
def add_precursor_ramp_enhanced(df, start_idx, ramp_len=20):
    for i in range(start_idx - ramp_len, start_idx):
        if i < 0:
            continue
        ramp_pos = (i - (start_idx - ramp_len) + 1) / ramp_len  # 0–1

        base_rpm = df.at[i, "Engine rpm"]
        rpm_jitter = np.random.normal(0, 50 * ramp_pos)

        df.at[i, "Engine rpm"] = base_rpm + rpm_jitter
        df.at[i, "lub oil temp"] += 2.5 * ramp_pos
        df.at[i, "Coolant temp"] = 90 + 20 * ramp_pos + np.random.uniform(-0.5, 0.5)
        df.at[i, "Coolant pressure"] *= 1 + 0.03 * ramp_pos
        df.at[i, "Lub oil pressure"] *= 1 - 0.10 * ramp_pos      
        df.at[i, "Fuel pressure"] *= 1 - 0.05 * ramp_pos

#force coolant to overheat and change engine condition to 1 (failed), also degrade sensors
def inject_overheat_enhanced(df, start_idx, duration):
    end_idx = min(start_idx + duration, len(df))

    for i in range(start_idx, len(df)):
        progress = min(1.0, (i - start_idx) / duration)
        
        base_rpm = df.at[i, "Engine rpm"]
        rpm_shift = np.random.normal(-50 * progress, 60)

        df.at[i, "Engine rpm"] = max(0, base_rpm + rpm_shift)
        df.at[i, "Lub oil pressure"] *= 0.85 - 0.05 * progress
        df.at[i, "Coolant temp"] = np.random.uniform(112, 122)
        df.at[i, "Coolant pressure"] *= 1.04 + 0.04 * progress
        df.at[i, "lub oil temp"] += 1.0 + 2.5 * progress
        df.at[i, "Fuel pressure"] *= 0.92 - 0.03 * progress

        df.at[i, "Engine Condition"] = 1

#add noise to all features except engine condition and time
def add_feature_noise(df, ref_df, noise_frac=0.01):
    for col in df.columns:
        if col not in ["Time", "Engine Condition"]:
            sigma = ref_df[col].std() * noise_frac
            df[col] += np.random.normal(0, sigma, size=len(df))

#generate n sequences that overheat randomly (when overheat, there will be warning signs beforehand)
def generate_sequences(source_df, n_sequences, seq_len, sampling_rate, noise_level, seed=None):
    np.random.seed(seed)
    sequences = []

    feature_cols = [
        "Engine rpm", "Lub oil pressure", "Fuel pressure",
        "Coolant pressure", "lub oil temp", "Coolant temp"
    ]

    for seq_idx in range(n_sequences):
        df = source_df.sample(n=seq_len, replace=True).reset_index(drop=True)
        df.insert(0, "Time", np.arange(seq_len) / sampling_rate)
        df["Engine Condition"] = 0

        will_overheat = np.random.choice([True, False], p=[0.6, 0.4])

        if will_overheat:
            overheat_start = np.random.randint(int(seq_len * 0.05), int(seq_len * 0.95))
            overheat_duration = np.random.randint(10, 60)
            ramp_len = np.random.randint(10, 50)
            precursor_start = max(0, overheat_start - ramp_len)

            add_precursor_ramp_enhanced(df, precursor_start, ramp_len)
            inject_overheat_enhanced(df, start_idx=overheat_start, duration=overheat_duration)

        else:
            pass 

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
        seq_len=SEQUENCE_LENGTH,
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
