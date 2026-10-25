from pathlib import Path

import geopandas as gpd
import planetary_computer
from pystac_client import Client


WARD_FILE = Path(
    "data/mumbai_wards_cleaned.geojson"
)


def find_landsat_scene():
    """Find a low-cloud Landsat scene covering Mumbai."""

    wards = gpd.read_file(WARD_FILE)

    min_x, min_y, max_x, max_y = wards.total_bounds
    bbox = [min_x, min_y, max_x, max_y]

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime="2025-01-01/2025-12-31",
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
            "No suitable Landsat scenes found."
        )

    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover",
            100,
        )
    )

    selected_item = items[0]

    signed_item = planetary_computer.sign(
        selected_item
    )

    print("Selected Landsat scene:")
    print(f"ID: {selected_item.id}")
    print(
        "Cloud cover: "
        f"{selected_item.properties.get('eo:cloud_cover'):.2f}%"
    )
    print(
        "Acquisition date: "
        f"{selected_item.datetime.date()}"
    )
    print()
    print("Available assets:")

    for asset_name, asset in signed_item.assets.items():
        print(
            f"{asset_name}: "
            f"{asset.href}"
        )


if __name__ == "__main__":
    find_landsat_scene()