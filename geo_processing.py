import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from shapely.geometry import Point, Polygon
import torch
import concurrent.futures
import os

class GISProcessor:
    def __init__(self, dem_path):
        self.dem_path = dem_path
        self.dem = rasterio.open(dem_path)
        self.dem_array = self.dem.read(1)
        self.dem_transform = self.dem.transform
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _process_geom(self, geom, transformer, dem_transform, array, value, border_value, progress, total_geoms):
        if geom.geom_type == 'Polygon':
            coords = [(transformer.transform(coord[0], coord[1])) for coord in geom.exterior.coords]
            poly = Polygon(coords)
            min_x, min_y, max_x, max_y = poly.bounds

            # 경계를 border_value로 설정
            for coord in poly.exterior.coords:
                y, x = int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[1]), int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[0])
                if 0 <= x < array.shape[0] and 0 <= y < array.shape[1]:
                    array[x, y] = border_value

            # 내부를 value로 설정
            for i in range(int(min_y), int(max_y) + 1):
                for j in range(int(min_x), int(max_x) + 1):
                    px, py = transformer.transform(j, i)
                    point = Point(px, py)
                    if poly.contains(point):
                        row, col = rowcol(dem_transform, px, py)
                        if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
                            array[row, col] = value

        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                self._process_geom(poly, transformer, dem_transform, array, value, border_value, progress, total_geoms)
        
        progress.value += 1
        print(f"Progress: {progress.value / total_geoms * 100:.2f}%")

    def transform_shapefile_to_dem(self, shapefile, value=1):
        dem_array = torch.tensor(self.dem_array, device=self.device)
        dem_transform = self.dem_transform
        shapefile_crs = shapefile.crs
        dem_crs = self.dem.crs
        transformer = Transformer.from_crs(shapefile_crs, dem_crs, always_xy=True)

        array = torch.zeros_like(dem_array)

        progress = torch.tensor(0, device=self.device)
        total_geoms = len(shapefile.geometry)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_geom, geom, transformer, dem_transform, array, value, value, progress, total_geoms) for geom in shapefile.geometry]
            concurrent.futures.wait(futures)

        return array.cpu().numpy()

    def preprocess_watershed(self, shapefile):
        dem_array = torch.tensor(self.dem_array, device=self.device)
        dem_transform = self.dem_transform
        shapefile_crs = shapefile.crs
        dem_crs = self.dem.crs
        transformer = Transformer.from_crs(shapefile_crs, dem_crs, always_xy=True)

        array = torch.zeros_like(dem_array)

        progress = torch.tensor(0, device=self.device)
        total_geoms = len(shapefile.geometry)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_geom, geom, transformer, dem_transform, array, 1, 0, progress, total_geoms) for geom in shapefile.geometry]
            concurrent.futures.wait(futures)

        return array.cpu().numpy()

    def create_featured_dem(self, rirsv_shapefile, wkmstrm_shapefile, road_shapefile, watershed_shapefile, channels_shapefile):
        rirsv_array = self.preprocess_rirsv(rirsv_shapefile)
        self.save_and_check_array(rirsv_array, 'rirsv_array.npy')

        wkmstrm_array = self.preprocess_wkmstrm(wkmstrm_shapefile)
        self.save_and_check_array(wkmstrm_array, 'wkmstrm_array.npy')

        road_array = self.transform_shapefile_to_dem(road_shapefile, value=3)
        self.save_and_check_array(road_array, 'road_array.npy')

        watershed_array = self.preprocess_watershed(watershed_shapefile)
        self.save_and_check_array(watershed_array, 'watershed_array.npy')

        channels_array = self.transform_shapefile_to_dem(channels_shapefile, value=5)
        self.save_and_check_array(channels_array, 'channels_array.npy')

        combined_array = np.stack((self.dem_array, rirsv_array, wkmstrm_array, road_array, watershed_array, channels_array), axis=-1)

        return combined_array

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
