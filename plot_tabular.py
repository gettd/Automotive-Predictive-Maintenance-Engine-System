import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from path_params import *

df = pd.read_csv(SEED_DATASET)

sensor_columns = [
    "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp" , "Engine Condition"
]

os.makedirs(PLOT_DIR, exist_ok=True)

correlation_matrix = df.corr()

engine_condition_corr = correlation_matrix["Engine Condition"].drop("Engine Condition").sort_values(ascending=False)

print("Correlation with Engine Condition:")
print(engine_condition_corr)


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[["Engine Condition"]].sort_values(by="Engine Condition", ascending=False), 
            annot=True, cmap="coolwarm", vmin=-1, vmax=1)

plt.title("Correlation of Sensor Readings with Engine Condition")
output_file = os.path.join(PLOT_DIR, "engine_data.png")
plt.savefig(output_file)
