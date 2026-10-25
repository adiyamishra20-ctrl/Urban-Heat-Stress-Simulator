"""Scenario definitions and intervention parameters."""

SCENARIOS = {
    "Urban Greening": {
        "name": "Urban Greening",
        "description": "Increase tree canopy and vegetation to reduce surface temperature and exposure.",
        "target_reduction_pct": 12,
        "feature_adjustments": {"tree_cover_pct": 15, "ndvi": 0.10},
    },
    "Cool Roofs and Reflective Surfaces": {
        "name": "Cool Roofs and Reflective Surfaces",
        "description": "Increase roof and pavement reflectivity to reduce absorbed solar heat.",
        "target_reduction_pct": 10,
        "feature_adjustments": {"roof_albedo": 0.20, "impervious_surface_pct": -5},
    },
    "Blue-Green Cooling Corridors": {
        "name": "Blue-Green Cooling Corridors",
        "description": "Connect vegetation, water, shade, and permeable surfaces across high-risk areas.",
        "target_reduction_pct": 20,
        "feature_adjustments": {"tree_cover_pct": 10, "water_fraction": 0.05, "impervious_surface_pct": -8},
    },
}
