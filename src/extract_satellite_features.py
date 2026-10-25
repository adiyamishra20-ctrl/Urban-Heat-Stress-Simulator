from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom


WARD_FILE = Path(
    "data/mumbai_wards_cleaned.geojson"
)

OUTPUT_FILE = Path(
    "data/mumbai_satellite_features.csv"
)

STAC_URL = (
    "https://planetarycomputer.microsoft.com/"
    "api/stac/v1"
)


def find_scene():
    """Find the selected low-cloud Landsat scene."""

    wards = gpd.read_file(WARD_FILE)

    min_x, min_y, max_x, max_y = (
        wards.total_bounds
    )

    catalog = Client.open(STAC_URL)

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=[
            min_x,
            min_y,
            max_x,
            max_y,
        ],
        datetime="2025-12-01/2025-12-31",
        query={
            "eo:cloud_cover": {
                "lt": 20
            }
        },
        max_items=20,
    )

    items = list(search.items())

    if not items:
        raise RuntimeError(
            "No Landsat scene found."
        )

    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover",
            100,
        )
    )

    return planetary_computer.sign(
        items[0]
    )


def read_band(asset):
    """Read one signed Landsat raster band."""

    with rasterio.open(asset.href) as source:
        return (
            source.read(1).astype("float32"),
            source.transform,
            source.crs,
            source.nodata,
        )


def calculate_features():
    """Calculate NDVI and land-surface temperature by ward."""

    wards = gpd.read_file(WARD_FILE)
    item = find_scene()

    red, transform, raster_crs, red_nodata = (
        read_band(item.assets["red"])
    )

    nir, _, _, nir_nodata = read_band(
        item.assets["nir08"]
    )

    thermal, _, _, thermal_nodata = read_band(
        item.assets["lwir11"]
    )

    wards_projected = wards.to_crs(
        raster_crs
    )

    red_reflectance = (
        red * 0.0000275
    ) - 0.2

    nir_reflectance = (
        nir * 0.0000275
    ) - 0.2

    ndvi = (
        (nir_reflectance - red_reflectance)
        / (
            nir_reflectance
            + red_reflectance
            + 0.000001
        )
    )

    land_surface_temperature_c = (
        thermal * 0.00341802
    ) + 149.0 - 273.15

    valid_pixels = (
        np.isfinite(ndvi)
        & np.isfinite(land_surface_temperature_c)
        & (ndvi >= -1)
        & (ndvi <= 1)
        & (land_surface_temperature_c > 0)
        & (land_surface_temperature_c < 80)
    )

    output_rows = []

    for _, ward in wards_projected.iterrows():
        ward_mask = geometry_mask(
            [ward.geometry],
            out_shape=ndvi.shape,
            transform=transform,
            invert=True,
        )

        valid_ward_pixels = (
            ward_mask
            & valid_pixels
        )

        if valid_ward_pixels.sum() == 0:
            mean_ndvi = np.nan
            mean_temperature = np.nan
        else:
            mean_ndvi = float(
                np.nanmean(
                    ndvi[valid_ward_pixels]
                )
            )

            mean_temperature = float(
                np.nanmean(
                    land_surface_temperature_c[
                        valid_ward_pixels
                    ]
                )
            )

        output_rows.append(
            {
                "neighborhood_id": ward[
                    "neighborhood_id"
                ],
                "ndvi": mean_ndvi,
                "land_surface_temperature_c": (
                    mean_temperature
                ),
            }
        )

    satellite_features = pd.DataFrame(
        output_rows
    )

    satellite_features = (
        satellite_features.dropna()
    )

    satellite_features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "Satellite feature extraction complete."
    )
    print(
        f"Wards with valid data: "
        f"{len(satellite_features)}"
    )
    print(
        f"Average NDVI: "
        f"{satellite_features['ndvi'].mean():.3f}"
    )
    print(
        "Average land-surface temperature: "
        f"{satellite_features['land_surface_temperature_c'].mean():.2f} °C"
    )
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    calculate_features()