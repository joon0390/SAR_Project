import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from shapely.geometry import Point, Polygon
import concurrent.futures
import os
import multiprocessing

class GISProcessor:
    def __init__(self, dem_path):
        self.dem_path = dem_path
        self.dem = rasterio.open(dem_path)
        self.dem_array = self.dem.read(1)
        self.dem_transform = self.dem.transform

    def _process_geom(self, geom, transformer, dem_transform, array, value, border_value, progress, total_geoms):
        coords = [(transformer.transform(coord[0], coord[1])) for coord in geom.exterior.coords]
        poly = Polygon(coords)
        min_x, min_y, max_x, max_y = poly.bounds

        # 경계와 내부를 동시에 처리
        for i in range(int(min_y), int(max_y) + 1):
            for j in range(int(min_x), int(max_x) + 1):
                px, py = transformer.transform(j, i)
                point = Point(px, py)
                row, col = rowcol(dem_transform, px, py)
                if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
                    if poly.contains(point):
                        array[row, col] = value
                    elif poly.touches(point):
                        array[row, col] = border_value

        with progress.get_lock():
            progress.value += 1
            if progress.value % 10 == 0 or progress.value == total_geoms:
                print(f"Progress: {progress.value / total_geoms * 100:.2f}%")

    def transform_shapefile_to_dem(self, shapefile, value=1):
        dem_array = np.zeros_like(self.dem_array)
        dem_transform = self.dem_transform
        shapefile_crs = shapefile.crs
        dem_crs = self.dem.crs
        transformer = Transformer.from_crs(shapefile_crs, dem_crs, always_xy=True)

        progress = multiprocessing.Value('i', 0)
        total_geoms = len(shapefile.geometry)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._process_geom, geom, transformer, dem_transform, dem_array, value, value, progress, total_geoms) for geom in shapefile.geometry]
            concurrent.futures.wait(futures)

        return dem_array

    def preprocess_watershed(self, shapefile):
        dem_array = np.zeros_like(self.dem_array)
        dem_transform = self.dem_transform
        dem_array = np.zeros_like(self.dem_array)
        dem_transform = self.dem_transform
        shapefile_crs = shapefile.crs
        dem_crs = self.dem.crs
        transformer = Transformer.from_crs(shapefile_crs, dem_crs, always_xy=True)

        progress = multiprocessing.Value('i', 0)
        total_geoms = len(shapefile.geometry)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._process_geom, geom, transformer, dem_transform, dem_array, 1, 0, progress, total_geoms) for geom in shapefile.geometry]
            concurrent.futures.wait(futures)

        return dem_array

    def create_featured_dem(self, rirsv_shapefile, wkmstrm_shapefile, road_shapefile, watershed_shapefile, channels_shapefile):
        rirsv_array = self.load_or_process_array('rirsv_array.npy', self.preprocess_rirsv, rirsv_shapefile)
        wkmstrm_array = self.load_or_process_array('wkmstrm_array.npy', self.preprocess_wkmstrm, wkmstrm_shapefile)
        road_array = self.load_or_process_array('road_array.npy', self.transform_shapefile_to_dem, road_shapefile, 3)
        watershed_array = self.load_or_process_array('watershed_array.npy', self.preprocess_watershed, watershed_shapefile)
        channels_array = self.load_or_process_array('channels_array.npy', self.transform_shapefile_to_dem, channels_shapefile, 5)

        combined_array = np.stack((self.dem_array, rirsv_array, wkmstrm_array, road_array, watershed_array, channels_array), axis=-1)

        return combined_array

    def load_or_process_array(self, filename, process_function, *args):
        if os.path.exists(filename):
            print(f"Loading existing array from {filename}")
            array = self.load_array(filename)
        else:
            print(f"Processing and saving array to {filename}")
            array = process_function(*args)
            self.save_and_check_array(array, filename)
        return array

    def preprocess_rirsv(self, shapefile):
        print("Preprocessing rirsv shapefile...")
        transformed_array = self.transform_shapefile_to_dem(shapefile, value=-10000)
        print("Transformed RIRSV Array:")
        print(transformed_array)
        return transformed_array
    
    def preprocess_wkmstrm(self, shapefile):
        print("Preprocessing wkmstrm shapefile...")
        transformed_array = self.transform_shapefile_to_dem(shapefile, value=2)
        print("Transformed WKSTRM Array:")
        print(transformed_array)
        return transformed_array

    def save_and_check_array(self, array, filename):
        np.save(filename, array)
        print(f"Array saved to {filename}")
        loaded_array = np.load(filename)
        print(f"Array loaded from {filename}")
        print(f"Loaded array shape: {loaded_array.shape}")
        return loaded_array

    def save_array(self, array, filename):
        np.save(filename, array)
        print(f"Array saved to {filename}")

    def load_array(self, filename):
        array = np.load(filename)
        print(f"Array loaded from {filename}")
        return array

def load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, road_shp_file, watershed_basins_shp_file, channels_shp_file):
    rirsv = gpd.read_file(rirsv_shp_file)
    wkmstrm = gpd.read_file(wkmstrm_shp_file)
    road = gpd.read_file(road_shp_file)
    watershed_basins = gpd.read_file(watershed_basins_shp_file)
    channels = gpd.read_file(channels_shp_file)
    
    return rirsv, wkmstrm, road, watershed_basins, channels

# Example usage
if __name__ == "__main__":
    from config import *

    processor = GISProcessor(dem_path)
    rirsv, wkmstrm, road, watershed_basins, channels = load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, road_shp_file, watershed_basins_shp_file, channels_shp_file)

    print("Creating featured DEM")
    featured_dem = processor.create_featured_dem(rirsv, wkmstrm, road, watershed_basins, channels)
    print("Featured DEM shape:", featured_dem.shape)

    # Save the array
    save_path = 'featured_dem.npy'
    processor.save_array(featured_dem, save_path)

    # Load the array
    loaded_array = processor.load_array(save_path)
    print("Loaded Array shape:", loaded_array.shape)

# Example usage
if __name__ == "__main__":
    from config import *

    processor = GISProcessor(dem_path)
    rirsv, wkmstrm, road, watershed_basins, channels = load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, road_shp_file, watershed_basins_shp_file, channels_shp_file)

    print("Creating featured DEM")
    featured_dem = processor.create_featured_dem(rirsv, wkmstrm, road, watershed_basins, channels)
    print("Featured DEM shape:", featured_dem.shape)

    # Save the array
    save_path = 'featured_dem.npy'
    processor.save_array(featured_dem, save_path)

    # Load the array
    loaded_array = processor.load_array(save_path)
    print("Loaded Array shape:", loaded_array.shape)
