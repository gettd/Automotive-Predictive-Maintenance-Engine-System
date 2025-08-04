import os
import pandas as pd
import matplotlib.pyplot as plt
from path_params import *

data_dir = SYNTHETIC_OUTPUT_DIR
time_column = "Time"

sensor_columns = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp" , "Engine Condition"
]

os.makedirs(PLOT_DIR, exist_ok=True)

csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

for file_name in csv_files:
    file_path = os.path.join(data_dir, file_name)
    df = pd.read_csv(file_path)

    print(f"Plotting file: {file_name}")

    num_plots = len(sensor_columns)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 2.5 * num_plots), sharex=True)

    for i, col in enumerate(sensor_columns):
        if col in df.columns:
            axs[i].plot(df[time_column], df[col], label=col, color='tab:blue')
            axs[i].set_ylabel(col)
            axs[i].grid(True)
            axs[i].legend(loc='upper right')
        else:
            axs[i].text(0.5, 0.5, f"{col} not found", ha='center', va='center', transform=axs[i].transAxes)
            axs[i].set_ylabel(col)

    axs[-1].set_xlabel("Time")
    last_time = df[time_column].iloc[-1]
    fig.suptitle(
        f"Sensor Data for {file_name}\nFinal Time: {last_time:.0f}",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_file = os.path.join(PLOT_DIR, file_name.replace(".csv", ".png"))
    plt.savefig(output_file)
    plt.close(fig)

    print(f"Saved plot to: {output_file}")

print("All plots generated and saved.")