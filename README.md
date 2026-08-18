# Urban Heat Stress Simulator

An end-to-end geospatial machine-learning prototype that estimates neighborhood-scale urban heat stress and tests mitigation strategies through an interactive Streamlit dashboard.

## Project highlights

- **121 neighborhoods** covering **121 km²** in a synthetic study area.
- **Random Forest regression** using urban form, vegetation, surface and demographic predictors.
- Test-set validation with **R² reported directly by the app**.
- Interactive comparison of three mitigation scenarios:
  1. Cool roofs
  2. Urban greening
  3. Combined strategy
- Neighborhood-level explorer and model feature importance.
- Built with **Python, scikit-learn, GeoPandas-compatible spatial data concepts, Pandas, Matplotlib and Streamlit**.

> **Important:** The bundled dataset is synthetic. The project is designed as a portfolio/demo implementation of the workflow, not as a claim of validated heat-risk prediction for a real city. For real deployment, replace the dataset with satellite, weather-station, land-use and census/GIS data and perform spatial cross-validation.

## Folder structure

```text
urban_heat_stress_simulator/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── neighborhood_heat_data.csv
└── src/
    ├── __init__.py
    └── model.py
```

## Run locally

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the simulator

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## Modeling workflow

```text
Neighborhood spatial data
        ↓
Feature engineering
        ↓
Train/test split
        ↓
Random Forest regression
        ↓
R² / MAE / RMSE validation
        ↓
Baseline heat-risk surface
        ↓
Mitigation scenario changes
        ↓
Predicted heat-risk surface
        ↓
Scenario comparison + neighborhood explorer
```

## Input features

| Feature | Interpretation |
|---|---|
| NDVI | Vegetation greenness |
| Tree canopy % | Share of neighborhood covered by tree canopy |
| Impervious % | Roads, roofs and other hard surfaces |
| Building density % | Built-form intensity |
| Road density | Road length per km² |
| Population density | Population per km² |
| Albedo | Surface reflectivity |
| Mean surface temperature | Baseline thermal condition |
| Wind speed | Local ventilation proxy |
| Elevation | Terrain proxy |

## Portfolio extension ideas

For a real-world version, replace the synthetic CSV with:

- Sentinel-2/Landsat NDVI and land-surface temperature
- OpenStreetMap road/building footprints
- Census population data
- DEM/elevation data
- Local weather-station observations
- Actual ward/neighborhood boundaries

Recommended next step: use **spatial train/test splits** rather than random splits to reduce spatial leakage, then publish model diagnostics and uncertainty alongside the dashboard.
