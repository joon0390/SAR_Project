import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from shapely.geometry import Point, Polygon
import os

class GISProcessor:
    def __init__(self, dem_path):
        self.dem_path = dem_path
        self.dem = rasterio.open(dem_path)
        self.dem_array = self.dem.read(1)
        self.dem_transform = self.dem.transform

    def transform_shapefile_to_dem(self, shapefile, value=1):
        dem_array = self.dem_array
        dem_transform = self.dem_transform
        shapefile_crs = shapefile.crs
        dem_crs = self.dem.crs
        transformer = Transformer.from_crs(shapefile_crs, dem_crs, always_xy=True)

        array = np.zeros_like(dem_array)

        for geom in shapefile.geometry:
            if geom.geom_type == 'Polygon':
                coords = [(int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[1]), 
                           int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[0])) 
                          for coord in geom.exterior.coords]
                for y, x in coords:
                    if 0 <= x < array.shape[0] and 0 <= y < array.shape[1]:
                        array[x, y] = value

            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = [(int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[1]), 
                               int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[0])) 
                              for coord in poly.exterior.coords]
                    for y, x in coords:
                        if 0 <= x < array.shape[0] and 0 <= y < array.shape[1]:
                            array[x, y] = value

        return array

    def preprocess_watershed(self, shapefile):
        dem_array = self.dem_array
        dem_transform = self.dem_transform
        shapefile_crs = shapefile.crs
        dem_crs = self.dem.crs
        transformer = Transformer.from_crs(shapefile_crs, dem_crs, always_xy=True)

        array = np.zeros_like(dem_array)
        unique_value = 1

        for geom in shapefile.geometry:
            if geom.geom_type == 'Polygon':
                coords = [(transformer.transform(coord[0], coord[1])) for coord in geom.exterior.coords]
                poly = Polygon(coords)
                min_x, min_y, max_x, max_y = poly.bounds

                print(f"Processing Polygon with bounds: {min_x}, {min_y}, {max_x}, {max_y}")

                for i in range(int(min_y), int(max_y) + 1):
                    for j in range(int(min_x), int(max_x) + 1):
                        px, py = transformer.transform(j, i)
                        point = Point(px, py)
                        if poly.contains(point):
                            row, col = rowcol(dem_transform, px, py)
                            if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
                                array[row, col] = unique_value
                                print(f"Setting array[{row}, {col}] = {unique_value}")

                print(f"Completed processing Polygon with unique value {unique_value}")
                unique_value += 1   

            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = [(transformer.transform(coord[0], coord[1])) for coord in poly.exterior.coords]
                    poly = Polygon(coords)
                    min_x, min_y, max_x, max_y = poly.bounds

                    print(f"Processing MultiPolygon with bounds: {min_x}, {min_y}, {max_x}, {max_y}")

                    for i in range(int(min_y), int(max_y) + 1):
                        for j in range(int(min_x), int(max_x) + 1):
                            px, py = transformer.transform(j, i)
                            point = Point(px, py)
                            if poly.contains(point):
                                row, col = rowcol(dem_transform, px, py)
                                if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
                                    array[row, col] = unique_value
                                    print(f"Setting array[{row}, {col}] = {unique_value}")

                    print(f"Completed processing part of MultiPolygon with unique value {unique_value}")
                    unique_value += 1

        return array

    def create_featured_dem(self, rirsv_shapefile, wkmstrm_shapefile, road_shapefile, watershed_shapefile, channels_shapefile):
        rirsv_array = self.preprocess_rirsv(rirsv_shapefile)
        wkmstrm_array = self.preprocess_wkmstrm(wkmstrm_shapefile)
        road_array = self.transform_shapefile_to_dem(road_shapefile, value=3)
        watershed_array = self.preprocess_watershed(watershed_shapefile)
        channels_array = self.transform_shapefile_to_dem(channels_shapefile, value=5)

        # 중간 결과 확인
        print("RIRSV Array:")
        print(rirsv_array)
        print("WKSTRM Array:")
        print(wkmstrm_array)
        print("Road Array:")
        print(road_array)
        print("Watershed Array:")
        print(watershed_array)
        print("Channels Array:")
        print(channels_array)

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

    featured_dem = processor.create_featured_dem(rirsv, wkmstrm, road, watershed_basins, channels)
    print("Featured DEM shape:", featured_dem.shape)

    # Save the array
    save_path = 'featured_dem.npy'
    processor.save_array(featured_dem, save_path)

    # Load the array
    loaded_array = processor.load_array(save_path)
    print("Loaded Array shape:", loaded_array.shape)
