import os
import pandas as pd
import numpy as np
from path_params import *

input_dir = SYNTHETIC_OUTPUT_TEST_DIR
#input_dir = SYNTHETIC_OUTPUT_TRAIN_DIR

keep_columns = ["Time", "Engine rpm", "Lub oil pressure", "Coolant temp", "Engine Condition"]

fill_value = 0

for file_name in os.listdir(input_dir):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_dir, file_name)

        # Read CSV
        df = pd.read_csv(file_path)

        # Identify columns to blank out
        cols_to_replace = [col for col in df.columns if col not in keep_columns]

        # Replace their values
        df[cols_to_replace] = fill_value

        # Overwrite file with updated data
        df.to_csv(file_path, index=False)

        print(f"Processed and saved: {file_name}  (replaced {len(cols_to_replace)} columns)")

print("All CSV files updated successfully.")
