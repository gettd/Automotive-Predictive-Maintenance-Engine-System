import os
import pandas as pd
import matplotlib.pyplot as plt
from path_params import *

data_dir = SYNTHETIC_OUTPUT_DIR
time_column = "Time"

sensor_columns = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp", "Engine Condition"
]

os.makedirs(PLOT_DIR, exist_ok=True)
subfolders = ["train", "test"]

# find min max of each col across all files
global_min = {col: float('inf') for col in sensor_columns}
global_max = {col: float('-inf') for col in sensor_columns}

for subfolder in subfolders:
    input_subdir = os.path.join(data_dir, subfolder)
    csv_files = [f for f in os.listdir(input_subdir) if f.endswith(".csv")]

    for file_name in csv_files:
        file_path = os.path.join(input_subdir, file_name)
        df = pd.read_csv(file_path)

        for col in sensor_columns:
            if col in df.columns:
                col_min, col_max = df[col].min(), df[col].max()
                global_min[col] = min(global_min[col], col_min)
                global_max[col] = max(global_max[col], col_max)

#add padding for better visualization
y_limits = {}
for col in sensor_columns:
    if global_min[col] < float('inf'):
        pad = (global_max[col] - global_min[col]) * 0.05
        y_limits[col] = (global_min[col] - pad, global_max[col] + pad)
    else:
        y_limits[col] = (0, 1) #default

print("Global y-axis limits per sensor:")
for col, (ymin, ymax) in y_limits.items():
    print(f"  {col}: {ymin:.2f} to {ymax:.2f}")

#plot
for subfolder in subfolders:
    input_subdir = os.path.join(data_dir, subfolder)
    output_subdir = os.path.join(PLOT_DIR, subfolder)
    os.makedirs(output_subdir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_subdir) if f.endswith(".csv")]

    for file_name in csv_files:
        file_path = os.path.join(input_subdir, file_name)
        df = pd.read_csv(file_path)

        print(f"Plotting file: {file_name} in {subfolder}")

        num_plots = len(sensor_columns)
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, 2.5 * num_plots), sharex=True)

        for i, col in enumerate(sensor_columns):
            if col in df.columns:
                axs[i].plot(df[time_column], df[col], label=col, color='tab:blue')
                axs[i].set_ylabel(col)
                axs[i].grid(True)
                axs[i].legend(loc='upper right')
                axs[i].set_ylim(y_limits[col])  #set limit
            else:
                axs[i].text(0.5, 0.5, f"{col} not found", ha='center', va='center', transform=axs[i].transAxes)
                axs[i].set_ylabel(col)

        axs[-1].set_xlabel("Time")
        last_time = df[time_column].iloc[-1]
        fig.suptitle(f"Sensor Data for {file_name} ({subfolder})\nFinal Time: {last_time:.0f}", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_file = os.path.join(output_subdir, file_name.replace(".csv", ".png"))
        plt.savefig(output_file)
        plt.close(fig)

        print(f"Saved plot to: {output_file}")

print("All plots generated and saved with consistent per-sensor y-axis scaling.")
