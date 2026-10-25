# Mumbai Urban Heat Stress Simulator

An interactive digital-twin-inspired simulator for analyzing neighborhood heat risk and testing urban mitigation strategies across Mumbai’s 227 electoral wards.

## Live Demo

[Launch the Mumbai Urban Heat Stress Simulator](https://urban-heat-stress-simulator-2zorwkacmsbyvutac7na4k.streamlit.app/)

## Project Overview

This project combines geospatial analysis, satellite observations, weather data, and machine learning to estimate heat risk across Mumbai.

The Streamlit dashboard allows users to:

- Explore ward-level heat risk
- Compare three mitigation scenarios
- View an interactive polygon heat map
- Analyze real weather conditions
- Download filtered ward results
- Inspect model feature importance

## Key Results

- **227 Mumbai electoral wards**
- **468.11 km² total study area**
- **R² = 0.938**
- **RMSE = 2.424**
- **MAE = 1.928**
- **20.40% maximum simulated heat-risk reduction**

## Mitigation Scenarios

### 1. Urban Greening

Simulates increased vegetation and tree canopy, combined with lower local surface temperature.

### 2. Cool Roofs and Reflective Surfaces

Simulates reduced heat absorption from reflective roofs and lower impervious-surface impact.

### 3. Blue-Green Cooling Corridors

Combines vegetation, reduced impervious surfaces, water features, shade, and localized cooling effects.

The Blue-Green scenario includes an upper-bound calibration factor representing microclimate effects not fully captured by ward-level satellite averages.

## Data Sources

### Geographic Boundaries

Mumbai electoral ward polygons from the 2017 BMC ward boundary dataset.

### Satellite Data

Landsat 9 Collection 2 Level-2 imagery obtained through the Microsoft Planetary Computer STAC API.

Derived features:

- NDVI
- Land-surface temperature

### Weather Data

Daily 2025 Mumbai weather data from the NASA POWER API.

Used variables:

- Temperature
- Relative humidity
- Wind speed
- Precipitation

## Machine Learning Features

The heat-risk model uses:

- Vegetation percentage derived from NDVI
- Land-surface temperature
- Building density
- Impervious-surface percentage
- Population density
- Relative humidity

A Random Forest regression model is trained using an 80/20 train-test split.

## Project Pipeline

```text
Mumbai ward polygons
        ↓
Satellite feature extraction
        ↓
Weather data collection
        ↓
Ward-level feature table
        ↓
Random Forest heat-risk model
        ↓
Mitigation scenario simulation
        ↓
Interactive Streamlit dashboard
