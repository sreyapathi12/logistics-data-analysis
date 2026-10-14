"""
Week 3: Advanced Data Analysis and Visualization in Logistics

Project structure:
- data/logistics_hypothetical_dataset.csv
- outputs/ (charts saved here)

Install libraries with: pip install -r requirements.txt
Run from the project root: python logistics_analysis.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "logistics_hypothetical_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n--- Dataset preview ---")
print(df.head())
print("\n--- Dataset information ---")
print(df.info())
print("\n--- Missing values ---")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

numeric_cols = [
    "Distance_km", "Shipment_Weight_kg",
    "Delivery_Time_days", "Transport_Cost_INR"
]
print("\n--- Descriptive statistics ---")
print(df[numeric_cols].describe().round(2))

print("\n--- Mean, median and mode ---")
for col in numeric_cols:
    mode_values = df[col].mode()
    mode_value = mode_values.iloc[0] if not mode_values.empty else "N/A"
    print(
        f"{col}: mean={df[col].mean():.2f}, "
        f"median={df[col].median():.2f}, mode={mode_value}"
    )

print("\n--- Correlation matrix ---")
print(df[numeric_cols].corr().round(2))

print("\n--- Results by transport mode ---")
mode_summary = df.groupby("Transport_Mode").agg(
    shipments=("Shipment_ID", "count"),
    average_delivery_days=("Delivery_Time_days", "mean"),
    average_cost_INR=("Transport_Cost_INR", "mean"),
    on_time_rate=("On_Time", lambda x: (x == "Yes").mean() * 100),
).round(2)
print(mode_summary)

# 1. Histogram: delivery-time distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Delivery_Time_days"], bins=18, edgecolor="black")
plt.title("Distribution of Delivery Times")
plt.xlabel("Delivery time (days)")
plt.ylabel("Number of shipments")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "chart_delivery_distribution.png", dpi=160)
plt.show()
plt.close()

# 2. Bar chart: shipment count by transport mode
plt.figure(figsize=(8, 5))
df["Transport_Mode"].value_counts().reindex(["Road", "Rail", "Air"]).plot(
    kind="bar", edgecolor="black"
)
plt.title("Shipment Count by Transport Mode")
plt.xlabel("Transport mode")
plt.ylabel("Number of shipments")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "chart_transport_mode_counts.png", dpi=160)
plt.show()
plt.close()

# 3. Box plot: delivery-time comparison
plt.figure(figsize=(8, 5))
df.boxplot(column="Delivery_Time_days", by="Transport_Mode", grid=False)
plt.suptitle("")
plt.title("Delivery Time by Transport Mode")
plt.xlabel("Transport mode")
plt.ylabel("Delivery time (days)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "chart_delivery_by_mode.png", dpi=160)
plt.show()
plt.close()

# 4. Scatter plot: distance versus cost
plt.figure(figsize=(8, 5))
plt.scatter(df["Distance_km"], df["Transport_Cost_INR"], alpha=0.65)
plt.title("Distance vs Transportation Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Transportation cost (INR)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "chart_distance_cost.png", dpi=160)
plt.show()
plt.close()

# 5. Correlation matrix
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 5))
plt.imshow(corr, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(label="Correlation coefficient")
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=30, ha="right")
plt.yticks(range(len(numeric_cols)), numeric_cols)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "chart_correlation_matrix.png", dpi=160)
plt.show()
plt.close()

# 6. On-time rate by transport mode
on_time_rates = (
    df.groupby("Transport_Mode")["On_Time"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reindex(["Road", "Rail", "Air"])
)
plt.figure(figsize=(8, 5))
on_time_rates.plot(kind="bar", edgecolor="black")
plt.title("On-Time Delivery Rate by Transport Mode")
plt.xlabel("Transport mode")
plt.ylabel("On-time deliveries (%)")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "chart_ontime_mode.png", dpi=160)
plt.show()
plt.close()

print(f"\nCharts saved in: {OUTPUT_DIR}")
