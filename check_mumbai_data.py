import geopandas as gpd


file_path = "data/mumbai_electoral_2017.geojson"

wards = gpd.read_file(file_path)

print("Number of wards:", len(wards))
print("Coordinate system:", wards.crs)
print("Columns:")
print(wards.columns.tolist())
print()
print(wards.head())