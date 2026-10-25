import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


RESULTS_FILE = Path(
    "data/mumbai_scenario_results.csv"
)

GEOJSON_FILE = Path(
    "data/mumbai_wards_cleaned.geojson"
)

MODEL_FILE = Path(
    "data/mumbai_heat_risk_model.pkl"
)

WEATHER_FILE = Path(
    "data/mumbai_weather_2025.csv"
)

st.set_page_config(
    page_title="Mumbai Urban Heat Stress Simulator",
    page_icon="🌡️",
    layout="wide",
)


@st.cache_data
def load_results():
    """Load Mumbai scenario simulation results."""

    if not RESULTS_FILE.exists():
        st.error(
            "Scenario results not found. Run "
            "`python src/scenario_simulator.py` first."
        )
        st.stop()

    return pd.read_csv(RESULTS_FILE)


@st.cache_data
def load_geojson():
    """Load Mumbai ward polygon boundaries."""

    if not GEOJSON_FILE.exists():
        st.error(
            "Mumbai ward GeoJSON not found."
        )
        st.stop()

    with open(
        GEOJSON_FILE,
        "r",
        encoding="utf-8",
    ) as geojson_file:
        return json.load(geojson_file)


@st.cache_resource
def load_model():
    """Load the trained Mumbai heat-risk model."""

    if not MODEL_FILE.exists():
        st.error(
            "Mumbai model not found. Run "
            "`python src/model.py` first."
        )
        st.stop()

    with open(
        MODEL_FILE,
        "rb",
    ) as model_file:
        return pickle.load(model_file)

@st.cache_data
def load_weather():
    """Load real Mumbai weather data."""

    if not WEATHER_FILE.exists():
        st.error(
            "Mumbai weather data not found. Run "
            "`python src/weather_data.py` first."
        )
        st.stop()

    return pd.read_csv(
        WEATHER_FILE,
        parse_dates=["date"],
    )


def classify_risk(score):
    """Assign a color category to a heat-risk score."""

    if score < 40:
        return "Low"
    elif score < 60:
        return "Moderate"
    elif score < 80:
        return "High"

    return "Very High"


results = load_results()
weather = load_weather()
mumbai_geojson = load_geojson()
model = load_model()


st.title("🌡️ Mumbai Urban Heat Stress Simulator")

st.caption(
    "Interactive heat-risk and mitigation analysis "
    "across 227 Mumbai electoral wards"
)


scenario_names = [
    "Baseline",
    "Urban Greening",
    "Cool Roofs and Reflective Surfaces",
    "Blue-Green Cooling Corridors",
]


reduction_columns = {
    "Urban Greening": (
        "Urban Greening Reduction (%)"
    ),
    "Cool Roofs and Reflective Surfaces": (
        "Cool Roofs Reduction (%)"
    ),
    "Blue-Green Cooling Corridors": (
        "Blue-Green Corridors Reduction (%)"
    ),
}


st.sidebar.header("Simulation Controls")


selected_scenario = st.sidebar.selectbox(
    "Select mitigation scenario",
    scenario_names,
)


risk_filter = st.sidebar.multiselect(
    "Show risk categories",
    [
        "Low",
        "Moderate",
        "High",
        "Very High",
    ],
    default=[
        "Low",
        "Moderate",
        "High",
        "Very High",
    ],
)


sort_order = st.sidebar.radio(
    "Sort neighborhoods by",
    [
        "Highest risk first",
        "Lowest risk first",
    ],
)


total_neighborhoods = len(results)
total_area = results["area_km2"].sum()
baseline_average = results["Baseline"].mean()


if selected_scenario == "Baseline":
    scenario_average = baseline_average
    average_reduction = 0.0
    selected_values = results["Baseline"]
else:
    scenario_average = results[
        selected_scenario
    ].mean()

    average_reduction = (
        (baseline_average - scenario_average)
        / baseline_average
        * 100
    )

    selected_values = results[
        selected_scenario
    ]


results["Selected Risk"] = selected_values

results["Risk Category"] = results[
    "Selected Risk"
].apply(classify_risk)


st.subheader("Project Overview")


metric_1, metric_2, metric_3, metric_4 = (
    st.columns(4)
)


with metric_1:
    st.metric(
        "Electoral Wards",
        f"{total_neighborhoods}",
    )


with metric_2:
    st.metric(
        "Study Area",
        f"{total_area:.2f} km²",
    )


with metric_3:
    st.metric(
        "Average Heat Risk",
        f"{scenario_average:.2f}",
    )


with metric_4:
    st.metric(
        "Risk Reduction",
        f"{average_reduction:.2f}%",
    )

st.subheader("Real Mumbai Weather Context")

weather_col1, weather_col2, weather_col3, weather_col4 = (
    st.columns(4)
)

with weather_col1:
    st.metric(
        "Average Temperature",
        f"{weather['temperature_c'].mean():.1f} °C",
    )

with weather_col2:
    st.metric(
        "Maximum Temperature",
        f"{weather['temperature_c'].max():.1f} °C",
    )

with weather_col3:
    st.metric(
        "Average Humidity",
        f"{weather['relative_humidity_pct'].mean():.1f}%",
    )

with weather_col4:
    st.metric(
        "Total Rainfall",
        f"{weather['precipitation_mm'].sum():.1f} mm",
    )

weather_chart = px.line(
    weather,
    x="date",
    y=[
        "temperature_c",
        "relative_humidity_pct",
    ],
    title="Mumbai Daily Temperature and Humidity",
)

weather_chart.update_layout(
    xaxis_title="Date",
    yaxis_title="Value",
    legend_title="Weather Variable",
)

st.plotly_chart(
    weather_chart,
    use_container_width=True,
)

st.subheader("Scenario Comparison")


scenario_averages = pd.DataFrame(
    {
        "Scenario": [
            "Baseline",
            "Urban Greening",
            "Cool Roofs",
            "Blue-Green Corridors",
        ],
        "Average Heat Risk": [
            results["Baseline"].mean(),
            results["Urban Greening"].mean(),
            results[
                "Cool Roofs and Reflective Surfaces"
            ].mean(),
            results[
                "Blue-Green Cooling Corridors"
            ].mean(),
        ],
    }
)


scenario_chart = px.bar(
    scenario_averages,
    x="Scenario",
    y="Average Heat Risk",
    color="Average Heat Risk",
    color_continuous_scale=[
        "#2ca25f",
        "#fec44f",
        "#de2d26",
    ],
    text_auto=".2f",
)


scenario_chart.update_layout(
    coloraxis_showscale=False,
    xaxis_title="",
    yaxis_title="Average Heat-Risk Score",
)


st.plotly_chart(
    scenario_chart,
    use_container_width=True,
)


chart_col1, chart_col2 = st.columns(2)


with chart_col1:
    st.subheader("Risk Distribution")

    distribution_chart = px.histogram(
        results,
        x="Selected Risk",
        color="Risk Category",
        nbins=15,
        color_discrete_map={
            "Low": "#2ca25f",
            "Moderate": "#fec44f",
            "High": "#fd8d3c",
            "Very High": "#de2d26",
        },
    )

    distribution_chart.update_layout(
        xaxis_title="Heat-Risk Score",
        yaxis_title="Number of Wards",
        legend_title="Risk Category",
    )

    st.plotly_chart(
        distribution_chart,
        use_container_width=True,
    )


with chart_col2:
    st.subheader("Ward Risk Ranking")

    ranking_data = results[
        [
            "neighborhood_id",
            "Selected Risk",
            "Risk Category",
        ]
    ].sort_values(
        "Selected Risk",
        ascending=(
            sort_order == "Lowest risk first"
        ),
    ).head(15)

    ranking_chart = px.bar(
        ranking_data,
        x="Selected Risk",
        y="neighborhood_id",
        color="Risk Category",
        orientation="h",
        color_discrete_map={
            "Low": "#2ca25f",
            "Moderate": "#fec44f",
            "High": "#fd8d3c",
            "Very High": "#de2d26",
        },
    )

    ranking_chart.update_layout(
        yaxis_title="Ward",
        xaxis_title="Heat-Risk Score",
    )

    st.plotly_chart(
        ranking_chart,
        use_container_width=True,
    )


filtered_results = results[
    results["Risk Category"].isin(risk_filter)
].copy()


if sort_order == "Highest risk first":
    filtered_results = filtered_results.sort_values(
        "Selected Risk",
        ascending=False,
    )
else:
    filtered_results = filtered_results.sort_values(
        "Selected Risk",
        ascending=True,
    )


st.subheader("Mumbai Ward Heat-Risk Map")


map_data = filtered_results.copy()


if len(map_data) > 0:
    polygon_map = px.choropleth_map(
        map_data,
        geojson=mumbai_geojson,
        locations="neighborhood_id",
        featureidkey=(
            "properties.neighborhood_id"
        ),
        color="Selected Risk",
        hover_name="neighborhood_id",
        hover_data={
            "Selected Risk": ":.2f",
            "area_km2": ":.2f",
            "neighborhood_id": False,
        },
        color_continuous_scale=[
            "#2ca25f",
            "#fec44f",
            "#fd8d3c",
            "#de2d26",
        ],
        range_color=[
            float(map_data["Selected Risk"].min()),
            float(map_data["Selected Risk"].max()),
        ],
        map_style="open-street-map",
        zoom=9.5,
        center={
            "lat": 19.08,
            "lon": 72.88,
        },
        opacity=0.70,
        height=650,
    )

    polygon_map.update_layout(
        margin={
            "r": 0,
            "t": 0,
            "l": 0,
            "b": 0,
        },
        coloraxis_colorbar_title="Heat Risk",
    )

    st.plotly_chart(
        polygon_map,
        use_container_width=True,
    )
else:
    st.warning(
        "No wards match the selected risk filters."
    )


st.subheader("Filtered Ward Data")


table_columns = [
    "neighborhood_id",
    "area_km2",
    "Baseline",
    "Selected Risk",
    "Risk Category",
]


if selected_scenario != "Baseline":
    table_columns.append(
        reduction_columns[selected_scenario]
    )


st.dataframe(
    filtered_results[table_columns],
    use_container_width=True,
    hide_index=True,
)


st.subheader("Download Results")


download_data = filtered_results[
    table_columns
].to_csv(index=False)


st.download_button(
    label="Download filtered ward results",
    data=download_data,
    file_name="mumbai_heat_risk_results.csv",
    mime="text/csv",
)


st.subheader("Model Feature Importance")


feature_columns = [
    "vegetation_pct",
    "building_density_pct",
    "impervious_surface_pct",
    "population_density",
    "humidity_pct",
    "average_temperature_c",
]


feature_importance = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Importance": model.feature_importances_,
    }
).sort_values(
    "Importance",
    ascending=True,
)


importance_chart = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale=[
        "#9ecae1",
        "#3182bd",
        "#08519c",
    ],
)


importance_chart.update_layout(
    xaxis_title="Relative Importance",
    yaxis_title="",
    coloraxis_showscale=False,
)


st.plotly_chart(
    importance_chart,
    use_container_width=True,
)


st.subheader("Interpretation")


if selected_scenario == "Baseline":
    st.info(
        "Baseline represents current modeled "
        "conditions without additional mitigation."
    )
else:
    st.success(
        f"{selected_scenario} produces an average "
        f"simulated heat-risk reduction of "
        f"{average_reduction:.2f}%."
    )


st.caption(
    "Note: This simulator uses synthetic heat-related "
    "features applied to real Mumbai electoral ward "
    "boundaries. Results represent modeled scenarios, "
    "not measured real-world outcomes."
)