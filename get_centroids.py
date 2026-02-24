import urllib.request
import zipfile
import os
import pandas as pd
import geopandas as gpd

# Download Official NYC TLC Taxi Zones Shapefile using a more reliable OpenData endpoint
url = "https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=Shapefile"
zip_path = "taxi_zones.zip"

if not os.path.exists(zip_path):
    print("Downloading Shapefile from NYC OpenData...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
        out_file.write(response.read())

    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("taxi_zones")

# Read using Geopandas
print("Reading Shapefile...")
# The shapefile from OpenData might have a different internal name, find it
shp_file = [f for f in os.listdir("taxi_zones") if f.endswith('.shp')][0]
gdf = gpd.read_file(os.path.join("taxi_zones", shp_file))

print("CRS:", gdf.crs)
if gdf.crs is None:
    # If no CRS provided, assume it's WGS84 for this export type or check
    pass
elif gdf.crs != 'EPSG:4326':
    gdf = gdf.to_crs(epsg=4326)

# Calculate centroids
print("Calculating Centroids...")
import warnings
warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
centroids = gdf.geometry.centroid

gdf['lon'] = centroids.x
gdf['lat'] = centroids.y

print(gdf.head())

# location_id is usually LocationID
# zone is Zone
# borough is Borough
# OpenData export often lowercases everything
df = pd.DataFrame(gdf.drop(columns='geometry'))
id_col = 'LocationID' if 'LocationID' in df.columns else 'location_i'
zone_col = 'zone' if 'zone' in df.columns else 'Zone'
borough_col = 'borough' if 'borough' in df.columns else 'Borough'

df = df[[id_col, zone_col, borough_col, 'lon', 'lat']]
df = df.rename(columns={id_col: 'LocationID', zone_col: 'zone', borough_col: 'borough'})

# Save to CSV
print("Saving to CSV...")
df.to_csv("taxi_zones_centroids.csv", index=False)
print(df.head())
print(f"Total Zones: {len(df)}")
