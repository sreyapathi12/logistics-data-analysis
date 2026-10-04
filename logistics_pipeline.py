"""
Week 1 - Strategic Planning & Data Exploration in Logistics
Scenario : Last-mile delivery and inventory analytics for a hypothetical
           Hyderabad-based FMCG distributor (HydLogix Distribution).
Run      : python logistics_pipeline.py
Needs    : pandas, numpy, scikit-learn, ortools

NOTE: The data below is SYNTHETIC. In the real project, replace
make_synthetic_data() with pd.read_csv() on the company / public datasets.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, silhouette_score

DEPOT = (17.4435, 78.3772)  # warehouse location (lat, lon), Hyderabad


# --- SNIPPET: synthetic ---
def make_synthetic_data(n_days=365, n_skus=10, n_orders=3000, seed=42):
    """Create demand history and delivery orders that mimic real patterns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")

    demand_rows = []
    for sku in range(n_skus):
        base = rng.integers(40, 150)
        season = 1 + 0.2 * np.sin(2 * np.pi * dates.dayofyear / 365)
        weekend = 1 + 0.15 * (dates.dayofweek >= 5)
        promo = rng.random(n_days) < 0.08
        units = base * season * weekend * (1 + 0.4 * promo)
        units = np.maximum(0, units + rng.normal(0, base * 0.08, n_days))
        demand_rows.append(pd.DataFrame({
            "date": dates, "sku": f"SKU{sku:02d}",
            "units": units.round(), "promo": promo.astype(int),
            "unit_cost": rng.uniform(20, 200),
        }))
    demand = pd.concat(demand_rows, ignore_index=True)

    lat = rng.normal(17.42, 0.05, n_orders)
    lon = rng.normal(78.42, 0.06, n_orders)
    km = np.hypot(lat - DEPOT[0], lon - DEPOT[1]) * 111 * 1.3
    ordered = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, n_days * 24, n_orders), unit="h")
    promised = ordered + pd.Timedelta(hours=24)
    delivered = ordered + pd.to_timedelta(
        4 + km * 0.7 + rng.gamma(2, 4, n_orders), unit="h")
    orders = pd.DataFrame({
        "order_id": np.arange(n_orders), "order_time": ordered,
        "promised_time": promised, "delivered_time": delivered,
        "lat": lat, "lon": lon, "distance_km": km,
        "delivery_cost": 30 + 9 * km + rng.normal(0, 15, n_orders),
        "demand_units": rng.integers(1, 12, n_orders),
    })
    return demand, orders
# --- END ---


# --- SNIPPET: clean ---
def clean_orders(orders):
    """Standard cleaning steps for raw delivery data."""
    df = orders.drop_duplicates("order_id").copy()
    for col in ["order_time", "promised_time", "delivered_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["order_time", "delivered_time", "lat", "lon"])

    # Remove implausible coordinates (outside the Hyderabad metro area)
    df = df[df["lat"].between(17.1, 17.8) & df["lon"].between(78.1, 78.8)]

    # Cap extreme costs using the IQR rule
    q1, q3 = df["delivery_cost"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["delivery_cost"] = df["delivery_cost"].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # Engineered features
    df["delay_hrs"] = (df["delivered_time"] - df["promised_time"]).dt.total_seconds() / 3600
    df["is_late"] = (df["delay_hrs"] > 0).astype(int)
    df["order_dow"] = df["order_time"].dt.dayofweek
    return df
# --- END ---


# --- SNIPPET: kpis ---
def compute_kpis(orders, demand, avg_inventory_value):
    """Return the headline KPIs used throughout the project."""
    otd = 1 - orders["is_late"].mean()                       # On-time delivery rate
    cost_per_order = orders["delivery_cost"].mean()          # Rs per order
    cost_per_km = orders["delivery_cost"].sum() / orders["distance_km"].sum()
    cogs = (demand["units"] * demand["unit_cost"]).sum()
    inventory_turnover = cogs / avg_inventory_value          # times per year
    return {
        "on_time_delivery_%": round(float(otd) * 100, 1),
        "avg_cost_per_order_Rs": round(float(cost_per_order), 1),
        "cost_per_km_Rs": round(float(cost_per_km), 2),
        "inventory_turnover_x": round(float(inventory_turnover), 1),
    }
# --- END ---


# --- SNIPPET: eda ---
def explore(orders, demand):
    """Quick exploratory analysis - trends, delay drivers, correlations."""
    print(orders[["distance_km", "delivery_cost", "delay_hrs"]].describe().round(2))

    weekly = demand.set_index("date").groupby(pd.Grouper(freq="W"))["units"].sum()
    print("\nWeekly demand (last 4 weeks):\n", weekly.tail(4))

    orders["dist_band"] = pd.cut(orders["distance_km"], [0, 5, 10, 15, 100],
                                 labels=["0-5", "5-10", "10-15", "15+"])
    print("\nLate-delivery rate by distance band:\n",
          orders.groupby("dist_band", observed=True)["is_late"].mean().round(3))

    print("\nCorrelations:\n",
          orders[["distance_km", "delivery_cost", "delay_hrs", "demand_units"]].corr().round(2))
# --- END ---


# --- SNIPPET: forecast ---
def forecast_demand(demand, sku="SKU00", horizon=28):
    """Gradient boosting demand forecast with lag and calendar features."""
    d = demand[demand["sku"] == sku].sort_values("date").copy()
    d["lag_1"] = d["units"].shift(1)
    d["lag_7"] = d["units"].shift(7)
    d["roll_mean_7"] = d["units"].shift(1).rolling(7).mean()
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d = d.dropna()

    features = ["lag_1", "lag_7", "roll_mean_7", "dow", "month", "promo"]
    train, test = d.iloc[:-horizon], d.iloc[-horizon:]   # time-based split, no shuffling

    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3)
    model.fit(train[features], train["units"])
    pred = model.predict(test[features])

    mape = mean_absolute_percentage_error(test["units"], pred)
    naive = mean_absolute_percentage_error(test["units"], test["lag_7"])  # baseline
    print(f"{sku}: model MAPE = {mape:.1%} | seasonal-naive MAPE = {naive:.1%}")
    return model, mape
# --- END ---


# --- SNIPPET: cluster ---
def build_delivery_zones(orders, k_range=range(3, 8)):
    """Group delivery points into zones with K-Means; pick k by silhouette score."""
    coords = orders[["lat", "lon"]].to_numpy()
    best_k, best_score = None, -1
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(coords)
        score = silhouette_score(coords, labels)
        if score > best_score:
            best_k, best_score = k, score

    orders["zone"] = KMeans(n_clusters=best_k, n_init=10,
                            random_state=0).fit_predict(coords)
    print(f"Chosen zones: {best_k} (silhouette = {best_score:.2f})")
    print(orders.groupby("zone").agg(orders=("order_id", "count"),
                                     late_rate=("is_late", "mean"),
                                     avg_cost=("delivery_cost", "mean")).round(2))
    return orders
# --- END ---


# --- SNIPPET: inventory ---
def inventory_policy(demand, lead_time_days=3, service_z=1.65,
                     order_cost=500, holding_rate=0.25):
    """ABC classification + safety stock, reorder point and EOQ per SKU."""
    g = demand.groupby("sku").agg(mean_d=("units", "mean"), std_d=("units", "std"),
                                  unit_cost=("unit_cost", "first"))
    g["annual_value"] = g["mean_d"] * 365 * g["unit_cost"]

    g = g.sort_values("annual_value", ascending=False)
    share = g["annual_value"].cumsum() / g["annual_value"].sum()
    g["abc_class"] = np.where(share <= 0.8, "A", np.where(share <= 0.95, "B", "C"))

    g["safety_stock"] = service_z * g["std_d"] * np.sqrt(lead_time_days)   # z * sigma * sqrt(L)
    g["reorder_point"] = g["mean_d"] * lead_time_days + g["safety_stock"]
    g["eoq"] = np.sqrt(2 * g["mean_d"] * 365 * order_cost /
                       (holding_rate * g["unit_cost"]))
    return g.round(1)
# --- END ---


# --- SNIPPET: routing ---
def haversine_matrix(points):
    """Pairwise distances in metres between (lat, lon) points."""
    p = np.radians(np.array(points))
    dlat = p[:, None, 0] - p[None, :, 0]
    dlon = p[:, None, 1] - p[None, :, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(p[:, None, 0]) * np.cos(p[None, :, 0]) * np.sin(dlon / 2) ** 2
    return (2 * 6371000 * np.arcsin(np.sqrt(a)) * 1.3).astype(int)  # x1.3 road factor


def solve_vrp(depot, stops, demands, n_vehicles=3, capacity=40):
    """Capacitated vehicle routing with Google OR-Tools."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    dist = haversine_matrix([depot] + stops)
    manager = pywrapcp.RoutingIndexManager(len(dist), n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dist[manager.IndexToNode(i)][manager.IndexToNode(j)])

    transit = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    dem = [0] + list(demands)
    demand_cb = routing.RegisterUnaryTransitCallback(lambda i: dem[manager.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(demand_cb, 0, [capacity] * n_vehicles, True, "Cap")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 5

    sol = routing.SolveWithParameters(params)
    routes, total_m = [], 0
    for v in range(n_vehicles):
        idx, route, leg = routing.Start(v), [], 0
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            nxt = sol.Value(routing.NextVar(idx))
            leg += routing.GetArcCostForVehicle(idx, nxt, v)
            idx = nxt
        routes.append(route + [0])
        total_m += leg
    return routes, total_m / 1000  # route node lists, total km
# --- END ---


# --- SNIPPET: main ---
if __name__ == "__main__":
    demand, raw_orders = make_synthetic_data()      # swap for pd.read_csv(...) later
    orders = clean_orders(raw_orders)

    print(compute_kpis(orders, demand, avg_inventory_value=4_000_000))
    explore(orders, demand)
    forecast_demand(demand, "SKU00")
    orders = build_delivery_zones(orders)
    print(inventory_policy(demand).head())

    # Route one zone's first 15 orders from the depot
    sample = orders[orders["zone"] == orders["zone"].iloc[0]].head(15)
    stops = list(zip(sample["lat"], sample["lon"]))
    routes, km = solve_vrp(DEPOT, stops, sample["demand_units"].tolist())
    print(f"Optimised total route distance: {km:.1f} km across {len(routes)} vehicles")
# --- END ---