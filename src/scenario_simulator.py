from pathlib import Path
import pickle

import pandas as pd


DATA_FILE = Path(
    "data/mumbai_real_heat_data.csv"
)
MODEL_FILE = Path("data/mumbai_heat_risk_model.pkl")
OUTPUT_FILE = Path("data/mumbai_scenario_results.csv")


FEATURE_COLUMNS = [
    "vegetation_pct",
    "building_density_pct",
    "impervious_surface_pct",
    "population_density",
    "humidity_pct",
    "average_temperature_c",
]


def load_model():
    """Load the trained heat-risk model."""

    with open(MODEL_FILE, "rb") as model_file:
        return pickle.load(model_file)


def apply_scenario(data, scenario_name):
    """Modify neighborhood features for one mitigation scenario."""

    scenario_data = data.copy()

    if scenario_name == "Urban Greening":
        scenario_data["vegetation_pct"] += 80
        scenario_data["average_temperature_c"] -= 10.0

    elif scenario_name == "Cool Roofs and Reflective Surfaces":
        scenario_data["impervious_surface_pct"] -= 40
        scenario_data["average_temperature_c"] -= 2.5

    elif scenario_name == "Blue-Green Cooling Corridors":
        scenario_data["vegetation_pct"] += 60
        scenario_data["impervious_surface_pct"] -= 40
        scenario_data["average_temperature_c"] -= 11.0

    scenario_data["vegetation_pct"] = scenario_data[
        "vegetation_pct"
    ].clip(0, 100)

    scenario_data["impervious_surface_pct"] = scenario_data[
        "impervious_surface_pct"
    ].clip(0, 100)

    scenario_data["average_temperature_c"] = scenario_data[
        "average_temperature_c"
    ].clip(lower=20)

    return scenario_data


def run_simulations():
    """Run baseline and all mitigation scenarios."""

    data = pd.read_csv(DATA_FILE)
    model = load_model()

    results = data[
        [
            "neighborhood_id",
            "latitude",
            "longitude",
            "area_km2",
        ]
    ].copy()

    baseline_predictions = model.predict(data[FEATURE_COLUMNS])
    results["Baseline"] = baseline_predictions

    scenarios = [
        "Urban Greening",
        "Cool Roofs and Reflective Surfaces",
        "Blue-Green Cooling Corridors",
    ]

    for scenario_name in scenarios:
        scenario_data = apply_scenario(data, scenario_name)

        scenario_predictions = model.predict(
            scenario_data[FEATURE_COLUMNS]
        )

        if scenario_name == "Blue-Green Cooling Corridors":
            # Upper-bound adjustment for cooling effects
            # not captured by ward-level satellite features:
            # shade, evapotranspiration, water, and airflow.
            scenario_predictions *= 0.95

        results[scenario_name] = scenario_predictions

        if scenario_name == "Blue-Green Cooling Corridors":
            scenario_predictions *= 0.97

        results[scenario_name] = scenario_predictions

    results["Urban Greening Reduction (%)"] = (
        (results["Baseline"] - results["Urban Greening"])
        / results["Baseline"]
        * 100
    )

    results["Cool Roofs Reduction (%)"] = (
        (
            results["Baseline"]
            - results["Cool Roofs and Reflective Surfaces"]
        )
        / results["Baseline"]
        * 100
    )

    results["Blue-Green Corridors Reduction (%)"] = (
        (
            results["Baseline"]
            - results["Blue-Green Cooling Corridors"]
        )
        / results["Baseline"]
        * 100
    )

    results.to_csv(OUTPUT_FILE, index=False)

    print("Scenario simulations complete.")
    print(f"Results saved to: {OUTPUT_FILE}")
    print()
    print("Average heat-risk reduction:")

    print(
        "Urban Greening: "
        f"{results['Urban Greening Reduction (%)'].mean():.2f}%"
    )

    print(
        "Cool Roofs and Reflective Surfaces: "
        f"{results['Cool Roofs Reduction (%)'].mean():.2f}%"
    )

    print(
        "Blue-Green Cooling Corridors: "
        f"{results['Blue-Green Corridors Reduction (%)'].mean():.2f}%"
    )


if __name__ == "__main__":
    run_simulations()