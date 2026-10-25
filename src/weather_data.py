from pathlib import Path

import pandas as pd
import requests


OUTPUT_FILE = Path("data/mumbai_weather_2025.csv")

NASA_POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)

PARAMETERS = [
    "T2M",
    "RH2M",
    "WS10M",
    "PRECTOTCORR",
]


def download_mumbai_weather():
    """Download daily weather data for Mumbai from NASA POWER."""

    query_parameters = {
        "parameters": ",".join(PARAMETERS),
        "community": "RE",
        "longitude": 72.8777,
        "latitude": 19.0760,
        "start": "20250101",
        "end": "20251231",
        "format": "JSON",
    }

    response = requests.get(
        NASA_POWER_URL,
        params=query_parameters,
        timeout=60,
    )

    response.raise_for_status()

    weather_json = response.json()

    daily_data = weather_json["properties"]["parameter"]

    weather = pd.DataFrame(daily_data)

    weather.index.name = "date"
    weather = weather.reset_index()

    weather["date"] = pd.to_datetime(
        weather["date"],
        format="%Y%m%d",
    )

    weather = weather.rename(
        columns={
            "T2M": "temperature_c",
            "RH2M": "relative_humidity_pct",
            "WS10M": "wind_speed_mps",
            "PRECTOTCORR": "precipitation_mm",
        }
    )

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    weather.to_csv(OUTPUT_FILE, index=False)

    print("Mumbai weather data downloaded.")
    print(f"Rows downloaded: {len(weather)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print()
    print(weather.head())


if __name__ == "__main__":
    download_mumbai_weather()