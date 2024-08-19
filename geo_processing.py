import os
import numpy as np
import rasterio
from rasterio import transform
import geopandas as gpd

class GISProcessor:
    def __init__(self, dem_file, npy_file='featured_dem.npy'):
        self.dem_file = dem_file
        self.npy_file = npy_file
        self.dem_transform = None
        self.dem_array = self.load_dem()

    def load_dem(self):
        # Check if the npy file exists
        if os.path.exists(self.npy_file):
            print(f"Loading DEM data from {self.npy_file}")
            dem_array = np.load(self.npy_file)
            with rasterio.open(self.dem_file) as src:
                self.dem_transform = src.transform
        else:
            print(f"Loading DEM data from {self.dem_file}")
            with rasterio.open(self.dem_file) as src:
                dem_array = src.read(1)
                self.dem_transform = src.transform
            # Save the array for future use
            np.save(self.npy_file, dem_array)
        
        return dem_array

    def pixel_to_coords(self, row, col):
        """Convert pixel coordinates to real world coordinates."""
        x, y = rasterio.transform.xy(self.dem_transform, row, col)
        return x, y

    def pixel_distance_to_meters(self, pixel_distance):
        """Convert pixel distance to meters using the transform."""
        pixel_size_x = self.dem_transform[0]
        return pixel_distance * pixel_size_x

    def meters_to_pixel_distance(self, meters):
        """Convert distance in meters to pixel distance using the transform."""
        pixel_size_x = self.dem_transform[0]
        return meters / pixel_size_x

def load_shapefiles(directory):
    """Load all shapefiles from a directory into a list of GeoDataFrames."""
    shapefiles = []
    for file in os.listdir(directory):
        if file.endswith(".shp"):
            filepath = os.path.join(directory, file)
            gdf = gpd.read_file(filepath)
            shapefiles.append(gdf)
    return shapefiles
