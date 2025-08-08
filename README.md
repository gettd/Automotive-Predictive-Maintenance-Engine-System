# Automotive Predictive Maintenance Engine System

This repository contains the codebase and data pipeline for **transforming tabular engine health data into synthetic time series datasets**, and training/testing a machine learning model to perform predictive maintenance on vehicle engines.

This project is part of the Computer Engineering (EGCI) Bachelor Thesis at **Mahidol University**. The full paper, including background, motivations, methodology, and results, can be found [in this repository](https://github.com/gettd/Thesis-Paper).

---

## Project Summary

The goal is to simulate realistic sequential vehicle engine data using a publicly available tabular dataset and use that data to **train a time-series model** capable of early fault detection (specifically overheating and lubrication failures).

The public dataset used comes from [Kaggle](https://www.kaggle.com/datasets/parvmodi/automotive-vehicles-engine-health-dataset/data), and was featured in the published IEEE paper [#10912235](https://ieeexplore.ieee.org/document/10912235).

Since the original dataset is not time series-based, this project performs:

1. **Transformation of tabular data into sequential time series format**
2. **Simulation of failure cases with early warning signs**
3. **Noise injection to simulate real-world sensor readings**
4. **Model training using LSTM**
5. **Evaluation on synthetic test sequences**

---

## Repository Structure

```
├── datasets
    ├── data/              # Original dataset (CSV from Kaggle)
    ├── plot/              # All plots and visualizations
        ├── result/           # Visualization when testing
        ├── test/
        ├── train/
        └── engine_data.png   #heatmap showing correlation coefficient of factors to engine condition
    └── synthetic_data/    # N generated synthetic time series datasets
        ├── test/
        └── train/
├── model
    ├── lstm_model.keras       # Trained Keras model (output)
    └── scaler.pkl             # Fitted scaler used for test-time normalization
├── generate_synthetic_timeseries.py  # Core generator script
├── path_params.py         # Centralized file paths
├── plot_tabular.py        # Plot Pearson correlation for original dataset
├── plot_timeseries.py     # Plot all generated synthetic sequences
├── train_model.py         # Train LSTM model using synthetic data
└── test_model.py          # Evaluate model on K random datasets
```

---

## How to Run the Project

Assuming you have the original dataset`engine_data.csv` ready in `datasets/data`:

### Step 1: Generate Synthetic Time Series
```bash
python generate_synthetic_timeseries.py
```
> Configure the number of generated datasets inside the script (by default, **N** = 40). More data leads to better results but increases training time.

> Allocate the datasets to training and testing directory as you prefer.

### Step 2 (Optional): Visualize the Generated Data
```bash
python plot_timeseries.py
```

### Step 3: Train the Model
```bash
python train_model.py
```
> This will save the trained model (`lstm_model.keras`) and the scaler (`scaler.pkl`).

### Step 4: Test the Model
```bash
python test_model.py
```
> You can configure the script to:
> - Pick **K** random datasets from the synthetic data folder
> - Or test on a specific dataset file

---

## Dependencies

You'll need Python 3.x and the following packages (can be installed via pip):

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow joblib
```

### Python Built-in Modules Used
- `os`
- `pathlib`
- `random`

---

## Example Output

Each synthetic dataset contains 10,000 rows of 1Hz sensor data with simulated failure or healthy patterns. Here's a sample of the output schema:

| Time | Engine rpm | Lub oil pressure | Fuel pressure | Coolant temp | ... | Engine Condition |
|------|------------|------------------|----------------|---------------|-----|------------------|
| 0    | 1600       | 4.3              | 2.1            | 90.1          | ... | 0                |
| ...  | ...        | ...              | ...            | ...           | ... | 1 (if failed)    |

Plots and model test results are saved under `datasets/plot/`.

---

## License

This project is released under a **custom fair-use license**:

> You are free to use, modify, and build upon this work for research or educational purposes. However, **citation of this repository and/or the original thesis paper is required** if the methodology or dataset generation pipeline is used in derivative works. Commercial use or direct copying without attribution is prohibited.

---

## Related Paper (and full reference of other papers)

📄 [Thesis Full Paper Repository](https://github.com/gettd/Thesis-Paper)
[slide](https://docs.google.com/presentation/d/1ua4ZmV-eydtFXtYIBoyijZYJOtkEgYoP/edit?usp=sharing&ouid=117551709161989122124&rtpof=true&sd=true)
