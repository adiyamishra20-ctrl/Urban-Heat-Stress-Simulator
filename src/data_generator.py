from pathlib import Path

import numpy as np
import pandas as pd


def generate_neighborhood_data(
    number_of_neighborhoods: int = 120,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic neighborhood-level urban heat data.

    The generated dataset represents 120 neighborhoods across
    approximately 60 square kilometers.
    """

    rng = np.random.default_rng(random_seed)

    neighborhood_ids = [
        f"N{index:03d}"
        for index in range(1, number_of_neighborhoods + 1)
    ]

    # Synthetic coordinates for a 60 km² study area.
    # These represent a fictional city and are not real locations.
    latitude = rng.uniform(
        28.58,
        28.68,
        number_of_neighborhoods,
    )

    longitude = rng.uniform(
        77.15,
        77.28,
        number_of_neighborhoods,
    )

    area_km2 = rng.uniform(0.35, 0.65, number_of_neighborhoods)

    vegetation_pct = rng.uniform(
        5,
        55,
        number_of_neighborhoods,
    )

    building_density_pct = rng.uniform(
        20,
        95,
        number_of_neighborhoods,
    )

    impervious_surface_pct = np.clip(
        building_density_pct + rng.normal(0, 8, number_of_neighborhoods),
        15,
        100,
    )

    population_density = rng.uniform(
        1000,
        25000,
        number_of_neighborhoods,
    )

    humidity_pct = rng.uniform(
        45,
        80,
        number_of_neighborhoods,
    )

    average_temperature_c = (
        31
        + (100 - vegetation_pct) * 0.045
        + building_density_pct * 0.025
        + impervious_surface_pct * 0.018
        + rng.normal(0, 0.8, number_of_neighborhoods)
    )

    heat_risk_score = (
        average_temperature_c * 1.8
        + building_density_pct * 0.25
        + impervious_surface_pct * 0.20
        + population_density / 5000
        + humidity_pct * 0.08
        - vegetation_pct * 0.35
        + rng.normal(0, 1.2, number_of_neighborhoods)
    )

    heat_risk_score = np.clip(
        heat_risk_score,
        0,
        100,
    )

    data = pd.DataFrame(
        {
            "neighborhood_id": neighborhood_ids,
            "latitude": latitude,
            "longitude": longitude,
            "area_km2": area_km2,
            "vegetation_pct": vegetation_pct,
            "building_density_pct": building_density_pct,
            "impervious_surface_pct": impervious_surface_pct,
            "population_density": population_density,
            "humidity_pct": humidity_pct,
            "average_temperature_c": average_temperature_c,
            "heat_risk_score": heat_risk_score,
        }
    )

    return data


if __name__ == "__main__":
    output_directory = Path("data")
    output_directory.mkdir(exist_ok=True)

    output_file = output_directory / "neighborhood_heat_data.csv"

    neighborhood_data = generate_neighborhood_data()
    neighborhood_data.to_csv(output_file, index=False)

    print(f"Created {len(neighborhood_data)} neighborhoods.")
    print(f"Total area: {neighborhood_data['area_km2'].sum():.2f} km²")
    print(f"Saved dataset to: {output_file}")