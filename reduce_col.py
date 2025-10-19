import os
import pandas as pd
from path_params import *

#input_dir = SYNTHETIC_OUTPUT_TEST_DIR
#input_dir = SYNTHETIC_OUTPUT_TRAIN_DIR

keep_columns = ["Time", "Engine rpm", "Coolant temp", "Engine Condition"]
fill_value = 0


def remove_unwanted_columns(input_dir, keep_columns):
    """
    Loop through all CSV files in the input_dir and remove columns not in keep_columns.
    """
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)

            # Read CSV
            df = pd.read_csv(file_path)

            # Keep only desired columns (ignore missing ones safely)
            df = df[[col for col in keep_columns if col in df.columns]]

            # Overwrite file with the reduced data
            df.to_csv(file_path, index=False)

            print(f"Processed and saved: {file_name}  (kept {len(df.columns)} columns)")

    print("All CSV files updated successfully (unwanted columns removed).")


# --- OLD LOGIC (commented out for optional use) ---
# def replace_unwanted_columns_with_value(input_dir, keep_columns, fill_value=0):
#     """
#     Loop through all CSV files and replace values in columns not in keep_columns with fill_value.
#     """
#     for file_name in os.listdir(input_dir):
#         if file_name.endswith(".csv"):
#             file_path = os.path.join(input_dir, file_name)
#
#             # Read CSV
#             df = pd.read_csv(file_path)
#
#             # Identify columns to replace
#             cols_to_replace = [col for col in df.columns if col not in keep_columns]
#
#             # Replace their values
#             df[cols_to_replace] = fill_value
#
#             # Overwrite file with updated data
#             df.to_csv(file_path, index=False)
#
#             print(f"Processed and saved: {file_name}  (replaced {len(cols_to_replace)} columns)")
#
#     print("All CSV files updated successfully (values replaced).")


# ---- RUN THE FUNCTION YOU WANT ----
remove_unwanted_columns(input_dir, keep_columns)
# replace_unwanted_columns_with_value(input_dir, keep_columns, fill_value)
