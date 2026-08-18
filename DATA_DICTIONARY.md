# Data dictionary

The supplied CSV is a deterministic synthetic dataset created for portfolio demonstration.

- `neighborhood_id`: unique neighborhood identifier
- `row`, `col`: grid position
- `x_min`, `y_min`: lower-left position in the synthetic coordinate system
- `width_km`, `height_km`: neighborhood dimensions
- `ndvi`: normalized vegetation index proxy
- `tree_canopy_pct`: tree canopy percentage
- `impervious_pct`: impervious surface percentage
- `building_density_pct`: building density proxy
- `road_density_km_per_km2`: road density
- `population_density`: population density
- `albedo`: surface reflectance proxy
- `mean_surface_temp_c`: mean surface temperature proxy
- `wind_speed_mps`: wind-speed proxy
- `elevation_m`: elevation proxy
- `heat_stress_index`: synthetic continuous target used to train the model
