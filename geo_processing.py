import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol, from_origin
from shapely.geometry import Point
import os
import matplotlib.pyplot as plt

class GISProcessor:
    def __init__(self, dem_path):
        self.dem = rasterio.open(dem_path)
        self.dem_array = self.dem.read(1)
        self.dem_transform = self.dem.transform
        print(f"DEM CRS: {self.dem.crs}")
        print(f"DEM bounds: {self.dem.bounds}")
        print(f"DEM shape: {self.dem_array.shape}")

    def define_region(self, shapefile):
        if shapefile.crs != self.dem.crs:
            shapefile = shapefile.to_crs(self.dem.crs)
        
        minx, miny, maxx, maxy = shapefile.total_bounds
        
        row_start, col_start = rowcol(self.dem_transform, minx, maxy)
        row_end, col_end = rowcol(self.dem_transform, maxx, miny)
        
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(self.dem_array.shape[0], row_end)
        col_end = min(self.dem_array.shape[1], col_end)
        
        print(f"Region of interest: rows {row_start}:{row_end}, cols {col_start}:{col_end}")
        return (row_start, row_end, col_start, col_end)

    def process_shapefile(self, shapefile, region):
        row_start, row_end, col_start, col_end = region
        result = np.zeros(self.dem_array.shape, dtype=np.uint8)
        
        print(f"Processing shapefile with CRS: {shapefile.crs}")
        print(f"Shapefile bounds: {shapefile.total_bounds}")
        
        if shapefile.crs != self.dem.crs:
            shapefile = shapefile.to_crs(self.dem.crs)
        
        for geometry in shapefile.geometry:
            if geometry.is_empty:
                continue
            geom_type = geometry.geom_type
            print(f"Processing geometry of type: {geom_type}")
            if geom_type in ["LineString", "MultiLineString"]:
                buffered_geom = geometry.buffer(1)  # Buffer to ensure points intersect with LINESTRING and MULTILINESTRING
            else:
                buffered_geom = geometry
            for i in range(self.dem_array.shape[0]):
                for j in range(self.dem_array.shape[1]):
                    x, y = self.dem_transform * (j, i)
                    point = Point(x, y)
                    if buffered_geom.intersects(point):
                        result[i, j] = 1
        
        print(f"Processed shapefile. Non-zero elements: {np.count_nonzero(result)}")
        return result[row_start:row_end, col_start:col_end]

    def load_or_process_array(self, filename, shapefile, region):
        if os.path.exists(filename):
            print(f"Loading existing array from {filename}")
            array = np.load(filename)
        else:
            print(f"Processing and saving array to {filename}")
            array = self.process_shapefile(shapefile, region)
            np.save(filename, array)
        self.check_non_zero_values(array, filename)
        return array

    def check_non_zero_values(self, array, filename):
        non_zero_count = np.count_nonzero(array)
        print(f"Number of non-zero elements in {filename}: {non_zero_count}")
        if non_zero_count > 0:
            print(f"Non-zero elements found in {filename}.")
        else:
            print(f"No non-zero elements found in {filename}.")

    def create_featured_dem(self, rirsv, wkmstrm, forestroad, road, watershed, channels,climbpath, region):
        row_start, row_end, col_start, col_end = region
        dem_slice = self.dem_array[row_start:row_end, col_start:col_end]
        
        rirsv_array = self.load_or_process_array('rirsv_array.npy', rirsv, region)
        wkmstrm_array = self.load_or_process_array('wkmstrm_array.npy', wkmstrm, region)
        forestroad_array = self.load_or_process_array('forestroad_array.npy', forestroad, region)
        road_array = self.load_or_process_array('road_array.npy', road, region)
        watershed_array = self.load_or_process_array('watershed_array.npy', watershed, region)
        channels_array = self.load_or_process_array('channels_array.npy', channels, region)
        climbpath_array = self.load_or_process_array('climbpath_array.npy',climbpath,region)

        print(f"DEM slice shape: {dem_slice.shape}")
        print(f"RIRSV array shape: {rirsv_array.shape}")
        print(f"WKMSTRM array shape: {wkmstrm_array.shape}")
        print(f"Climbpath array shape: {forestroad_array.shape}")
        print(f"Road array shape: {road_array.shape}")
        print(f"Watershed array shape: {watershed_array.shape}")
        print(f"Channels array shape: {channels_array.shape}")
        print(f"climbpath array shape: {climbpath_array.shape}")

        combined_array = np.stack((dem_slice, rirsv_array, wkmstrm_array,forestroad_array, road_array, watershed_array, channels_array,climbpath_array), axis=-1)
        print("Combined array shape:", combined_array.shape)

        return combined_array

def load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, forestroad_shp_file, road_shp_file, watershed_basins_shp_file,channels_shp_file ,climbpath_shp_file, area_difference_file):
    rirsv = gpd.read_file(rirsv_shp_file)
    wkmstrm = gpd.read_file(wkmstrm_shp_file)
    forestroad = gpd.read_file(forestroad_shp_file)
    road = gpd.read_file(road_shp_file)
    watershed_basins = gpd.read_file(watershed_basins_shp_file)
    channels = gpd.read_file(channels_shp_file)
    climbpath = gpd.read_file(climbpath_shp_file)
    test_area = gpd.read_file(area_difference_file)
    
    for name, shapefile in zip(["RIRSV", "WKMSTRM", "forestroad", "Road", "Watershed", "Channels","climbpath", "Test Area"],
                               [rirsv, wkmstrm, forestroad, road, watershed_basins, channels,climbpath, test_area]):
        print(f"{name} Shapefile CRS: {shapefile.crs}")
        print(f"{name} Shapefile bounds: {shapefile.total_bounds}")
    
    return rirsv, wkmstrm, forestroad, road, watershed_basins, channels,climbpath, test_area

def visualize_array(array, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(array, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    from config import *

    processor = GISProcessor(dem_file)
    rirsv, wkmstrm, forestroad, road, watershed_basins, channels,climbpath, test_area = load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, forestroad_shp_file, road_shp_file, watershed_basins_shp_file, channels_shp_file,climbpath_shp_file, area_difference_file)

    # Define region of interest
    region = processor.define_region(test_area)

    # Create featured DEM
    print("Creating featured DEM")
    featured_dem = processor.create_featured_dem(rirsv, wkmstrm, forestroad, road, watershed_basins, channels, climbpath,region)
    print("Featured DEM shape:", featured_dem.shape)

    # Save the array
    save_path = 'featured_dem.npy'
    np.save(save_path, featured_dem)
    print(f"Featured DEM saved to {save_path}")

    # Process test area
    test_area_result = processor.process_shapefile(test_area, region)
    np.save('test_area_result.npy', test_area_result)
    print("Test area result shape:", test_area_result.shape)

    # Visualize results
