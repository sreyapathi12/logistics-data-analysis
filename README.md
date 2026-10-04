# Logistics Data Analysis: Last-Mile Delivery and Inventory Optimisation

Week 1 task (Strategic Planning and Data Exploration in Logistics) for the Logistics Data Analyst Internship at YuvaIntern.

## Scenario
A hypothetical Hyderabad-based FMCG distributor ("HydLogix Distribution") with one warehouse, two fulfilment hubs and about 3,000 deliveries per day. It faces late deliveries in outer zones, high cost per delivery, and a mix of stockouts and excess stock.

## What the script does
`logistics_pipeline.py` illustrates the planned analysis workflow:

| Step | Function | Technique |
|---|---|---|
| Data | `make_synthetic_data()` | Synthetic demand and delivery data (placeholder for real data) |
| Cleaning | `clean_orders()` | Deduplication, coordinate validation, IQR outlier capping, feature engineering |
| KPIs | `compute_kpis()` | On-time delivery, cost per order, cost per km, inventory turnover |
| Exploration | `explore()` | Summary statistics, weekly demand, late rate by distance band, correlations |
| Forecasting | `forecast_demand()` | Gradient boosting with lag/calendar features, compared with a seasonal-naive baseline |
| Zones | `build_delivery_zones()` | K-Means clustering, k chosen by silhouette score |
| Inventory | `inventory_policy()` | ABC analysis, safety stock, reorder point, EOQ |
| Routing | `solve_vrp()` | Capacitated vehicle routing with Google OR-Tools |

## How to run
```bash
pip install -r requirements.txt
python logistics_pipeline.py
```

## Important note
All data in this repository is **synthetic**. Outputs show how the methods work, not real business results. In the full project, `make_synthetic_data()` would be replaced with real company data or public datasets (Olist, DataCo Smart Supply Chain, M5 Forecasting).

## Report
The full strategic planning report (KPIs, research, roadmap) is submitted separately as a Word document.