import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.model import (
    load_data, train_model, add_predictions, apply_scenario,
    SCENARIO_LABELS, FEATURES
)
from src.geospatial import make_geodataframe

st.set_page_config(
    page_title="Urban Heat Stress Simulator",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ Urban Heat Stress Simulator")
st.caption("A geospatial machine-learning prototype for neighborhood-scale heat-risk analysis.")

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model():
    df = get_data()
    return train_model(df)

df = get_data()
model, metrics, X_test, y_test, pred = get_model()
base = add_predictions(df, model)

# Sidebar controls
st.sidebar.header("Mitigation scenario")
scenario = st.sidebar.selectbox(
    "Choose a scenario",
    list(SCENARIO_LABELS.keys()),
    format_func=lambda x: SCENARIO_LABELS[x]
)
st.sidebar.markdown("---")
st.sidebar.write("Dataset: 121 synthetic neighborhoods")
st.sidebar.write("Spatial extent: 121 km²")
st.sidebar.write("Model: Random Forest Regressor")

scenario_df = apply_scenario(df, scenario)
scenario_df = add_predictions(scenario_df, model)

baseline_mean = base["predicted_heat_stress"].mean()
scenario_mean = scenario_df["predicted_heat_stress"].mean()
reduction = (baseline_mean - scenario_mean) / baseline_mean * 100

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Neighborhoods", f"{len(df):,}")
c2.metric("Study area", "121 km²")
c3.metric("Model R²", f"{metrics['r2']:.2f}")
c4.metric("Simulated reduction", f"{reduction:.1f}%")

if metrics["r2"] >= 0.75:
    st.success("Model validation target met: test-set R² ≥ 0.75.")
else:
    st.warning("Model validation target not met for this run.")

# Map-like neighborhood grid
st.subheader(f"Heat-stress surface — {SCENARIO_LABELS[scenario]}")
plot_df = make_geodataframe(scenario_df)
fig, ax = plt.subplots(figsize=(10, 7))
plot_df.plot(
    column="predicted_heat_stress", ax=ax, legend=True,
    edgecolor="white", linewidth=0.7, legend_kwds={"label": "Predicted heat-stress index"}
)
ax.set_xlabel("East–west position (km)")
ax.set_ylabel("North–south position (km)")
ax.set_aspect("equal")
ax.set_title("Neighborhood heat-stress index")
st.pyplot(fig, clear_figure=True)

# Scenario comparison
st.subheader("Scenario comparison")
scenario_names = ["baseline", "cool_roofs", "urban_greening", "combined"]
scenario_means = []
for s in scenario_names:
    s_df = add_predictions(apply_scenario(df, s), model)
    scenario_means.append(s_df["predicted_heat_stress"].mean())

comp = pd.DataFrame({
    "Scenario": [SCENARIO_LABELS[s] for s in scenario_names],
    "Mean predicted heat stress": scenario_means
})
st.bar_chart(comp.set_index("Scenario"))

# Feature importance
st.subheader("Model feature importance")
imp = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)
st.bar_chart(imp.set_index("Feature"))

# Neighborhood explorer
st.subheader("Neighborhood explorer")
selected = st.selectbox("Select neighborhood", df["neighborhood_id"])
row = base[base["neighborhood_id"] == selected].iloc[0]

left, right = st.columns(2)
with left:
    st.write("**Baseline characteristics**")
    st.dataframe(pd.DataFrame({
        "Indicator": ["NDVI", "Tree canopy", "Impervious surface", "Building density",
                     "Road density", "Population density", "Albedo", "Surface temperature"],
        "Value": [
            f"{row.ndvi:.2f}", f"{row.tree_canopy_pct:.1f}%",
            f"{row.impervious_pct:.1f}%", f"{row.building_density_pct:.1f}%",
            f"{row.road_density_km_per_km2:.1f} km/km²",
            f"{row.population_density:,.0f}/km²",
            f"{row.albedo:.2f}", f"{row.mean_surface_temp_c:.1f} °C"
        ]
    }, hide_index=True))
with right:
    selected_scenario = apply_scenario(df[df["neighborhood_id"] == selected], scenario)
    selected_scenario = add_predictions(selected_scenario, model).iloc[0]
    st.metric("Baseline predicted risk", f"{row.predicted_heat_stress:.1f}")
    st.metric("Scenario predicted risk", f"{selected_scenario.predicted_heat_stress:.1f}")
    local_reduction = (row.predicted_heat_stress-selected_scenario.predicted_heat_stress)/row.predicted_heat_stress*100
    st.metric("Local reduction", f"{local_reduction:.1f}%")

st.divider()
st.caption(
    "Prototype note: the included dataset is synthetic and is intended for demonstrating "
    "the end-to-end workflow (geospatial features → ML prediction → scenario simulation). "
    "It is not a validated city-scale heat-risk assessment."
)
