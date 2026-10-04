"""
Week 2 - Data Collection, Cleaning and Preprocessing for Logistics Analysis
Reference dataset : DataCo Smart Supply Chain (public). Its structure and typical
                    defects are reproduced here with SYNTHETIC data so the pipeline
                    can be demonstrated and tested end to end.
Run               : python week2_preprocessing_pipeline.py
Needs             : pandas, numpy, scikit-learn, matplotlib

To use the real file, replace make_raw_data() with load_real_dataco() and adjust
the column names if they differ in your download.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, OneHotEncoder,
                                   RobustScaler, StandardScaler)

OUT = Path(__file__).parent
VALID_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]


# --- SNIPPET: synthetic ---
def make_raw_data(n=5000, seed=7):
    """Synthetic orders that imitate DataCo-style columns, with deliberate defects."""
    rng = np.random.default_rng(seed)
    regions = {"North": (28.6, 77.2), "South": (13.0, 77.6), "East": (22.6, 88.4),
               "West": (19.1, 72.9), "Central": (23.3, 77.4)}
    region = rng.choice(list(regions), n)
    mode = rng.choice(VALID_MODES, n, p=[0.6, 0.2, 0.15, 0.05])
    scheduled = pd.Series(mode).map({"Standard Class": 4, "Second Class": 2,
                                     "First Class": 1, "Same Day": 0}).to_numpy()
    real = np.clip(scheduled + rng.integers(-1, 3, n), 0, None).astype(float)
    order_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")

    df = pd.DataFrame({
        "Order Id": np.arange(1, n + 1),
        "order date (DateOrders)": order_date.strftime("%Y-%m-%d"),
        "shipping date (DateOrders)": (order_date + pd.to_timedelta(real, unit="D")).strftime("%Y-%m-%d"),
        "Shipping Mode": mode,
        "Days for shipping (real)": real,
        "Days for shipment (scheduled)": scheduled.astype(float),
        "Order Region": region,
        "Sales": rng.lognormal(5, 0.6, n).round(2),
        "Order Item Quantity": rng.integers(1, 6, n).astype(float),
        "Order Item Discount Rate": rng.uniform(0, 0.25, n).round(2),
        "Latitude": [regions[r][0] + rng.normal(0, 0.8) for r in region],
        "Longitude": [regions[r][1] + rng.normal(0, 0.8) for r in region],
        "Order Zipcode": rng.integers(100000, 999999, n).astype(float),
        "Product Description": np.nan,
    })

    pick = lambda frac: rng.random(n) < frac
    # inconsistent text labels
    messy = pick(0.15)
    df.loc[messy, "Shipping Mode"] = df.loc[messy, "Shipping Mode"].str.upper() + " "
    # missing values
    for col, frac in [("Days for shipping (real)", 0.03), ("Sales", 0.04), ("Latitude", 0.02),
                      ("Longitude", 0.02), ("Order Region", 0.015), ("Order Item Quantity", 0.01)]:
        df.loc[pick(frac), col] = np.nan
    df.loc[pick(0.80), "Order Zipcode"] = np.nan
    # impossible values
    df.loc[pick(0.005), "Order Item Quantity"] = -1
    df.loc[pick(0.004), "Days for shipping (real)"] = -1
    bad_geo = pick(0.005)
    df.loc[bad_geo, ["Latitude", "Longitude"]] = 0.0
    df.loc[pick(0.005), "shipping date (DateOrders)"] = "N/A"
    # outliers (data-entry errors: extra zero)
    df.loc[pick(0.01), "Sales"] *= 15
    # duplicates
    df = pd.concat([df, df.sample(int(0.02 * n), random_state=1)], ignore_index=True)
    return df.sample(frac=1, random_state=3).reset_index(drop=True)
# --- END ---


# --- SNIPPET: load ---
def load_real_dataco(path):
    """Load the public DataCo CSV (column names may differ slightly by download)."""
    df = pd.read_csv(path, encoding="latin-1")      # file is not UTF-8 encoded
    print(df.shape)
    return df
# --- END ---


# --- SNIPPET: profile ---
def profile(df, title="Data profile"):
    """First look at the data: types, missing values, duplicates, cardinality."""
    out = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(1),
        "unique": df.nunique(),
    })
    print(f"\n== {title} ==  rows={len(df)}  cols={df.shape[1]}  "
          f"duplicate rows={df.duplicated().sum()}")
    print(out.sort_values("missing_%", ascending=False).to_string())
    return out
# --- END ---


# --- SNIPPET: standardise ---
def standardise(df):
    """Tidy column names, text labels and data types."""
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_"))
    df = df.rename(columns={
        "order_date_dateorders": "order_date",
        "shipping_date_dateorders": "shipping_date",
        "days_for_shipping_real": "days_real",
        "days_for_shipment_scheduled": "days_scheduled",
        "order_item_quantity": "quantity",
        "order_item_discount_rate": "discount_rate",
    })
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].str.strip()                           # remove stray spaces
    df["shipping_mode"] = df["shipping_mode"].str.title()   # 'STANDARD CLASS ' -> 'Standard Class'
    df["order_region"] = df["order_region"].str.title()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["shipping_date"] = pd.to_datetime(df["shipping_date"], errors="coerce")  # 'N/A' -> NaT
    return df
# --- END ---


# --- SNIPPET: duplicates_invalid ---
def remove_duplicates_and_invalid(df, log):
    """Drop repeated rows and turn impossible values into missing values."""
    before = len(df)
    df = df.drop_duplicates().copy()
    log["duplicates_removed"] = before - len(df)

    rules = {
        "quantity_<=0": df["quantity"] <= 0,
        "days_real_<0": df["days_real"] < 0,
        "coords_(0,0)": (df["latitude"] == 0) & (df["longitude"] == 0),
        "shipped_before_ordered": df["shipping_date"] < df["order_date"],
    }
    log["invalid_values"] = {k: int(v.sum()) for k, v in rules.items()}

    df.loc[rules["quantity_<=0"], "quantity"] = np.nan
    df.loc[rules["days_real_<0"], "days_real"] = np.nan
    df.loc[rules["coords_(0,0)"], ["latitude", "longitude"]] = np.nan
    df.loc[rules["shipped_before_ordered"], "shipping_date"] = pd.NaT
    return df
# --- END ---


# --- SNIPPET: missing ---
def handle_missing(df, log, drop_threshold=0.60):
    """Choose a treatment per column instead of one rule for everything."""
    df = df.copy()
    miss_pct = df.isna().mean()
    log["missing_before_imputation"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}

    # 1) Columns that are mostly empty carry no usable signal
    drop_cols = miss_pct[miss_pct > drop_threshold].index.tolist()
    df = df.drop(columns=drop_cols)
    log["columns_dropped"] = drop_cols

    # 2) Keep a record of what was imputed
    for c in ["sales", "days_real", "latitude"]:
        df[f"{c}_was_missing"] = df[c].isna().astype(int)

    # 3) Rows with no order date cannot be placed in time
    df = df.dropna(subset=["order_date"])

    # 4) Categorical: mode for shipping mode, explicit 'Unknown' for region
    df["shipping_mode"] = df["shipping_mode"].fillna(df["shipping_mode"].mode()[0])
    df["order_region"] = df["order_region"].fillna("Unknown")

    # 5) Numeric: group medians keep structure (robust to outliers)
    by_mode = df.groupby("shipping_mode")["days_real"].transform("median")
    df["days_real"] = df["days_real"].fillna(by_mode).round()
    for c in ["latitude", "longitude"]:
        df[c] = df[c].fillna(df.groupby("order_region")[c].transform("median"))
    for c in ["sales", "quantity"]:
        df[c] = df[c].fillna(df[c].median())
    df["quantity"] = df["quantity"].round().astype(int)

    # 6) Rebuild shipping date from business logic: order date + days taken
    rebuilt = df["order_date"] + pd.to_timedelta(df["days_real"], unit="D")
    df["shipping_date"] = df["shipping_date"].fillna(rebuilt)
    return df
# --- END ---


# --- SNIPPET: outliers ---
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def treat_outliers(df, log, col="sales"):
    """Flag and cap outliers. Sales is right-skewed, so bounds are set on the log scale."""
    df = df.copy()
    logged = np.log1p(df[col])
    lo, hi = iqr_bounds(logged)
    is_out = (logged < lo) | (logged > hi)

    z = (logged - logged.mean()) / logged.std()           # cross-check with z-score
    log["outliers"] = {"iqr_log_flagged": int(is_out.sum()),
                       "zscore_gt_3": int((z.abs() > 3).sum()),
                       "raw_max_before": round(float(df[col].max()), 2)}

    df[f"{col}_outlier"] = is_out.astype(int)             # keep a flag for later analysis
    df[col] = np.expm1(logged.clip(lo, hi))               # winsorise, then back to rupees
    log["outliers"]["raw_max_after"] = round(float(df[col].max()), 2)
    return df
# --- END ---


# --- SNIPPET: features ---
def add_features(df):
    """Derive business-meaningful fields from cleaned columns."""
    df = df.copy()
    df["shipping_delay"] = df["days_real"] - df["days_scheduled"]
    df["is_late"] = (df["shipping_delay"] > 0).astype(int)
    df["order_month"] = df["order_date"].dt.month
    df["order_dow"] = df["order_date"].dt.dayofweek
    return df
# --- END ---


# --- SNIPPET: scalers ---
def compare_scalers(df, col="sales"):
    """See how each scaler changes the same column before choosing one."""
    x = df[[col]]
    result = {
        "raw": x[col],
        "min-max": pd.Series(MinMaxScaler().fit_transform(x)[:, 0]),
        "z-score": pd.Series(StandardScaler().fit_transform(x)[:, 0]),
        "robust": pd.Series(RobustScaler().fit_transform(x)[:, 0]),
        "log1p": np.log1p(x[col]),
    }
    return pd.DataFrame({k: v.describe()[["min", "max", "mean", "std"]]
                         for k, v in result.items()}).round(2)
# --- END ---


# --- SNIPPET: preprocessor ---
NUM_STD = ["days_real", "days_scheduled", "shipping_delay", "latitude", "longitude"]
NUM_MINMAX = ["quantity", "discount_rate"]
NUM_LOG = ["sales"]
CATEGORICAL = ["shipping_mode", "order_region"]
MODEL_COLS = NUM_STD + NUM_MINMAX + NUM_LOG + CATEGORICAL


def build_preprocessor():
    """One reusable object that applies encoding and scaling the same way every time."""
    log_minmax = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                           ("scale", MinMaxScaler())])
    return ColumnTransformer(
        [("std", StandardScaler(), NUM_STD),
         ("minmax", MinMaxScaler(), NUM_MINMAX),
         ("log_minmax", log_minmax, NUM_LOG),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        sparse_threshold=0)                                # always return a dense array


def split_and_transform(df, train_frac=0.8):
    """Time-ordered split; the scalers learn ONLY from the training period."""
    df = df.sort_values("order_date")
    cut = int(len(df) * train_frac)
    train, test = df.iloc[:cut], df.iloc[cut:]

    prep = build_preprocessor()
    X_train = prep.fit_transform(train[MODEL_COLS])        # fit on training data only
    X_test = prep.transform(test[MODEL_COLS])              # reuse the same parameters
    names = prep.get_feature_names_out()
    return (pd.DataFrame(X_train, columns=names, index=train.index),
            pd.DataFrame(X_test, columns=names, index=test.index), prep)
# --- END ---


# --- SNIPPET: validate ---
def validate(df):
    """Automatic checks so a bad cleaning step fails loudly instead of silently."""
    checks = {
        "no duplicate rows": not df.duplicated().any(),
        "no missing values in model columns": bool(df[MODEL_COLS].notna().all().all()),
        "quantity > 0": bool((df["quantity"] > 0).all()),
        "days_real >= 0": bool((df["days_real"] >= 0).all()),
        "shipping_date >= order_date": bool((df["shipping_date"] >= df["order_date"]).all()),
        "shipping modes are valid": bool(df["shipping_mode"].isin(VALID_MODES).all()),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert all(checks.values()), "Validation failed"
    return checks
# --- END ---


def make_figures(raw, capped, scaled_train, folder):
    folder.mkdir(exist_ok=True)
    # Fig 1: missing values in the raw data
    miss = (raw.isna().mean() * 100).sort_values()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(miss.index, miss.values, color="#2E75B6")
    ax.axvline(60, color="#C0392B", ls="--", lw=1)
    ax.text(61, 0.2, "60% drop threshold", color="#C0392B", fontsize=8)
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Missing values in raw data")
    fig.tight_layout(); fig.savefig(folder / "fig1_missing_values.png", dpi=200); plt.close(fig)

    # Fig 2: sales before / after outlier treatment
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    axes[0].boxplot(raw["Sales"].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#F5B7B1"))
    axes[0].set_title("Sales before"); axes[0].set_xticks([])
    axes[1].boxplot(capped["sales"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#AED6F1"))
    axes[1].set_title("Sales after (capped)"); axes[1].set_xticks([])
    for a in axes: a.set_ylabel("Sales")
    fig.tight_layout(); fig.savefig(folder / "fig2_outliers_sales.png", dpi=200); plt.close(fig)

    # Fig 3: shape of sales under different transformations
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3))
    axes[0].hist(capped["sales"], bins=40, color="#7FB3D5"); axes[0].set_title("Capped sales")
    axes[1].hist(np.log1p(capped["sales"]), bins=40, color="#76D7C4"); axes[1].set_title("After log1p")
    axes[2].hist(scaled_train["log_minmax__sales"], bins=40, color="#F7DC6F")
    axes[2].set_title("log1p + min-max")
    for a in axes: a.tick_params(labelsize=7); a.title.set_fontsize(9)
    fig.tight_layout(); fig.savefig(folder / "fig3_scaling_sales.png", dpi=200); plt.close(fig)


# --- SNIPPET: main ---
def run_pipeline():
    log = {}
    raw = make_raw_data()                      # swap for load_real_dataco("DataCoSupplyChainDataset.csv")
    profile(raw, "RAW")

    df = standardise(raw)
    log["rows_raw"] = len(raw)
    df = remove_duplicates_and_invalid(df, log)
    df = handle_missing(df, log)
    df = treat_outliers(df, log, "sales")
    capped_only = df.copy()
    df = add_features(df)

    print("\nValidation:")
    log["validation"] = validate(df)
    log["rows_clean"] = len(df)

    print("\nScaler comparison for 'sales':\n", compare_scalers(df))
    X_train, X_test, prep = split_and_transform(df)
    print(f"\nModel-ready matrices: train={X_train.shape}, test={X_test.shape}")

    df.to_csv(OUT / "clean_orders.csv", index=False)
    return raw, capped_only, df, X_train, X_test, log
# --- END ---


if __name__ == "__main__":
    raw, capped, clean, X_train, X_test, log = run_pipeline()
    profile(clean, "CLEAN")
    make_figures(raw, capped, X_train, OUT / "figures")

    extra = {
        "missing_pct_raw": (raw.isna().mean() * 100).round(1).to_dict(),
        "scalers": compare_scalers(clean).to_dict(),
        "train_shape": list(X_train.shape), "test_shape": list(X_test.shape),
        "mean_days_real_raw": round(float(raw["Days for shipping (real)"].mean()), 2),
        "mean_days_real_clean": round(float(clean["days_real"].mean()), 2),
        "mean_sales_raw": round(float(raw["Sales"].mean()), 2),
        "mean_sales_clean": round(float(clean["sales"].mean()), 2),
        "late_rate_clean": round(float(clean["is_late"].mean() * 100), 1),
        "shipping_mode_labels_raw": int(raw["Shipping Mode"].nunique()),
        "shipping_mode_labels_clean": int(clean["shipping_mode"].nunique()),
    }
    (OUT / "results.json").write_text(json.dumps({**log, **extra}, indent=2, default=str))
    print("\nSaved clean_orders.csv, results.json and figures/")