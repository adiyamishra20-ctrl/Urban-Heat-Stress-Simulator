from pathlib import Path

import pandas as pd


BASE_FILE = Path(
    "data/mumbai_heat_data.csv"
)

SATELLITE_FILE = Path(
    "data/mumbai_satellite_features.csv"
)

OUTPUT_FILE = Path(
    "data/mumbai_real_heat_data.csv"
)


def build_real_feature_table():
    """Merge real satellite features into Mumbai model data."""

    base_data = pd.read_csv(BASE_FILE)
    satellite_data = pd.read_csv(
        SATELLITE_FILE
    )

    data = base_data.merge(
        satellite_data,
        on="neighborhood_id",
        how="inner",
    )

    # Convert NDVI from approximately -1 to 1
    # into a percentage-like vegetation feature.
    data["vegetation_pct"] = (
        (data["ndvi"] + 1)
        / 2
        * 100
    )

    # Replace synthetic temperature with
    # real satellite-derived surface temperature.
    data["average_temperature_c"] = (
        data["land_surface_temperature_c"]
    )

    # Recalculate the target using real satellite
    # temperature and vegetation.
    data["heat_risk_score"] = (
        data["average_temperature_c"] * 1.8
        + data["building_density_pct"] * 0.25
        + data["impervious_surface_pct"] * 0.20
        + data["population_density"] / 5000
        + data["humidity_pct"] * 0.08
        - data["vegetation_pct"] * 0.35
    )

    data["heat_risk_score"] = data[
        "heat_risk_score"
    ].clip(0, 100)

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "Real feature table created."
    )
    print(
        f"Wards included: {len(data)}"
    )
    print(
        "Real satellite features included:"
    )
    print(
        "- NDVI"
    )
    print(
        "- Land-surface temperature"
    )
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    build_real_feature_table()