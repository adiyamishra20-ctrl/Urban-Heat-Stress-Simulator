import geopandas as gpd
from shapely.geometry import box

def make_geodataframe(df):
    """Create a GeoDataFrame from the synthetic 1 km x 1 km neighborhood grid."""
    out = df.copy()
    out["geometry"] = [
        box(x, y, x+w, y+h)
        for x, y, w, h in zip(out.x_min, out.y_min, out.width_km, out.height_km)
    ]
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:3857")
