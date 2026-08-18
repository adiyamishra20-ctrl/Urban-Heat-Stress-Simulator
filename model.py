from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FEATURES = [
    "ndvi", "tree_canopy_pct", "impervious_pct", "building_density_pct",
    "road_density_km_per_km2", "population_density", "albedo",
    "mean_surface_temp_c", "wind_speed_mps", "elevation_m"
]

def load_data(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "neighborhood_heat_data.csv"
    return pd.read_csv(path)

def train_model(df):
    X = df[FEATURES]
    y = df["heat_stress_index"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=700, max_depth=14, min_samples_leaf=1,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }
    return model, metrics, X_test, y_test, pred

def add_predictions(df, model):
    out = df.copy()
    out["predicted_heat_stress"] = model.predict(out[FEATURES])
    return out

def apply_scenario(df, scenario):
    out = df.copy()
    if scenario == "baseline":
        return out
    if scenario == "cool_roofs":
        out["albedo"] = (out["albedo"] + 0.15).clip(0.15, 0.60)
        out["mean_surface_temp_c"] = out["mean_surface_temp_c"] - 1.5
    elif scenario == "urban_greening":
        out["tree_canopy_pct"] = (out["tree_canopy_pct"] + 25).clip(0, 90)
        out["ndvi"] = (out["ndvi"] + 0.18).clip(0, 0.95)
        out["mean_surface_temp_c"] = out["mean_surface_temp_c"] - 1.8
    elif scenario == "combined":
        # Aggressive portfolio scenario: large-scale cool surfaces + greening
        # + partial depaving + modeled local cooling.
        out["albedo"] = (out["albedo"] + 0.12).clip(0.15, 0.60)
        out["tree_canopy_pct"] = (out["tree_canopy_pct"] + 18).clip(0, 90)
        out["ndvi"] = (out["ndvi"] + 0.13).clip(0, 0.95)
        out["impervious_pct"] = (out["impervious_pct"] - 8).clip(5, 95)
        out["mean_surface_temp_c"] = out["mean_surface_temp_c"] - 2.2
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return out

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "cool_roofs": "Cool roofs",
    "urban_greening": "Urban greening",
    "combined": "Combined strategy"
}
