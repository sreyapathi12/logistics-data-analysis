# Week 2: Data Collection, Cleaning and Preprocessing for Logistics Analysis

Week 2 task for the Logistics Data Analyst Internship at YuvaIntern.

## Overview
A reusable preprocessing pipeline for logistics order and shipping data. The reference dataset is the public **DataCo Smart Supply Chain** dataset. For the demonstration, its column structure and typical problems are reproduced with **synthetic data** (5,100 records with deliberately injected defects), so every step can be run and checked.

## Problems handled
| Problem | Technique |
|---|---|
| Duplicate rows | `drop_duplicates()` |
| Inconsistent labels and wrong types | Column and text standardisation, `to_datetime(errors="coerce")` |
| Impossible values (negative quantity, negative days, coordinates at 0,0) | Rule checks converted to missing values |
| Missing values | Drop near-empty columns; group medians; business-logic rebuild; "Unknown" category; missing flags |
| Outliers | IQR rule on log scale with z-score cross-check; capping (winsorising) and flag column |
| Different scales | Z-score, Min-Max, log transform, one-hot encoding via `ColumnTransformer` |
| Data leakage | Time-ordered split; scalers fitted on training data only |
| Silent errors | Automatic validation checks |

## Files
- `week2_preprocessing_pipeline.py`: the full pipeline
- `requirements.txt`: required libraries
- `figures/`: missing values, outlier treatment, scaling charts

## How to run
```bash
pip install -r requirements.txt
python week2_preprocessing_pipeline.py
```
The script prints a data profile, validation results and a scaler comparison, and saves `clean_orders.csv`, `results.json` and the charts.

## Using the real dataset
Replace `make_raw_data()` in `run_pipeline()` with `load_real_dataco("path/to/file.csv")`. The file uses non-UTF-8 encoding, and column names may need small changes in `standardise()`. The real-file loader has not been tested on the full dataset.

## Note
All results shown here come from synthetic data and illustrate the method, not real business findings.