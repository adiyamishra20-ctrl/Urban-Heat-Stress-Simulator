from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import transform


WARD_FILE = Path("data/mumbai_electoral_2017.geojson")
OUTPUT_FILE = Path("data/mumbai_heat_data.csv")


def remove_z_coordinate(geometry):
    """Convert 3D geometry into standard 2D geometry."""

    return transform(
        lambda x, y, z=None: (x, y),
        geometry,
    )


def generate_mumbai_data(random_seed=42):
    """Create synthetic heat features for all 227 Mumbai wards."""

    rng = np.random.default_rng(random_seed)

    wards = gpd.read_file(WARD_FILE)

    wards = wards.to_crs(epsg=4326)
    wards["geometry"] = wards["geometry"].apply(
        remove_z_coordinate
    )

    wards["neighborhood_id"] = [
        f"MUM-{index:03d}"
        for index in range(1, len(wards) + 1)
    ]

    projected_wards = wards.to_crs(epsg=32643)

    centroids = projected_wards.geometry.centroid.to_crs(
        epsg=4326
    )

    wards["latitude"] = centroids.y
    wards["longitude"] = centroids.x

    wards["area_km2"] = (
        projected_wards.geometry.area / 1_000_000
    )

    wards["vegetation_pct"] = rng.uniform(
        5,
        55,
        len(wards),
    )

    wards["building_density_pct"] = rng.uniform(
        25,
        95,
        len(wards),
    )

    wards["impervious_surface_pct"] = np.clip(
        wards["building_density_pct"]
        + rng.normal(0, 8, len(wards)),
        15,
        100,
    )

    wards["population_density"] = rng.uniform(
        2000,
        35000,
        len(wards),
    )

    wards["humidity_pct"] = rng.uniform(
        55,
        85,
        len(wards),
    )

    wards["average_temperature_c"] = (
        30
        + (100 - wards["vegetation_pct"]) * 0.045
        + wards["building_density_pct"] * 0.025
        + wards["impervious_surface_pct"] * 0.018
        + rng.normal(0, 0.8, len(wards))
    )

    wards["heat_risk_score"] = (
        wards["average_temperature_c"] * 1.8
        + wards["building_density_pct"] * 0.25
        + wards["impervious_surface_pct"] * 0.20
        + wards["population_density"] / 5000
        + wards["humidity_pct"] * 0.08
        - wards["vegetation_pct"] * 0.35
        + rng.normal(0, 1.2, len(wards))
    )

    wards["heat_risk_score"] = wards[
        "heat_risk_score"
    ].clip(0, 100)

    output_columns = [
        "neighborhood_id",
        "latitude",
        "longitude",
        "area_km2",
        "vegetation_pct",
        "building_density_pct",
        "impervious_surface_pct",
        "population_density",
        "humidity_pct",
        "average_temperature_c",
        "heat_risk_score",
    ]

    output_data = wards[output_columns]

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    output_data.to_csv(OUTPUT_FILE, index=False)

    wards.to_file(
        "data/mumbai_wards_cleaned.geojson",
        driver="GeoJSON",
    )

    print(f"Created {len(output_data)} Mumbai wards.")
    print(
        f"Total study area: "
        f"{output_data['area_km2'].sum():.2f} km²"
    )
    print(f"Saved model data to: {OUTPUT_FILE}")
    print(
        "Saved cleaned polygons to: "
        "data/mumbai_wards_cleaned.geojson"
    )


if __name__ == "__main__":
    generate_mumbai_data()