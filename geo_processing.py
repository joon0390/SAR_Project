import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from shapely.geometry import Point

class GISProcessor:
    def __init__(self, dem_path):
        self.dem_path = dem_path
        self.dem = rasterio.open(dem_path)
    
    def transform_shapefile_to_dem(self, shapefile):
        '''
        shapefile을 dem 영역에 맞춰 변환. 각 폴리곤은 고유한 값을 가지며 경계는 0으로 설정.
        '''
        dem_array = self.dem.read(1)
        dem_transform = self.dem.transform
        shapefile_crs = shapefile.crs
        dem_crs = self.dem.crs
        transformer = Transformer.from_crs(shapefile_crs, dem_crs, always_xy=True)

        array = np.zeros_like(dem_array)
        unique_value = 1

        for geom in shapefile.geometry:
            if geom.geom_type == 'Polygon':
                coords = [(int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[0]), 
                           int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[1])) 
                          for coord in geom.exterior.coords]
                for x, y in coords:
                    if 0 <= x < array.shape[0] and 0 <= y < array.shape[1]:
                        array[x, y] = 0  # 경계는 0으로 설정

                # 내부를 고유한 값으로 설정
                for i in range(min(x for x, y in coords), max(x for x, y in coords) + 1):
                    for j in range(min(y for x, y in coords), max(y for x, y in coords) + 1):
                        if array[i, j] != 0:
                            array[i, j] = unique_value
                unique_value += 1

            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = [(int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[0]), 
                               int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[1])) 
                              for coord in poly.exterior.coords]
                    for x, y in coords:
                        if 0 <= x < array.shape[0] and 0 <= y < array.shape[1]:
                            array[x, y] = 0  # 경계는 0으로 설정

                    # 내부를 고유한 값으로 설정
                    for i in range(min(x for x, y in coords), max(x for x, y in coords) + 1):
                        for j in range(min(y for x, y in coords), max(y for x, y in coords) + 1):
                            if array[i, j] != 0:
                                array[i, j] = unique_value
                    unique_value += 1

        return array
    
    def preprocess_watershed(self, shapefile):
        '''
        shapefile을 dem 영역에 맞춰 변환하되, 
        각 유역을 고유한 값으로 매핑하고 경계는 0으로 설정
        '''
        dem_array = self.dem.read(1)
        dem_transform = self.dem.transform
        shapefile_crs = shapefile.crs
        dem_crs = self.dem.crs
        transformer = Transformer.from_crs(shapefile_crs, dem_crs, always_xy=True)

        array = np.zeros_like(dem_array)
        unique_value = 1

        for geom in shapefile.geometry:
            if geom.geom_type == 'Polygon':
                coords = [(int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[0]), 
                           int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[1])) 
                          for coord in geom.exterior.coords]
                for x, y in coords:
                    if 0 <= x < array.shape[0] and 0 <= y < array.shape[1]:
                        array[x, y] = 0  # 경계를 0으로 설정

                # 내부를 고유한 값으로 설정
                min_x, min_y, max_x, max_y = geom.bounds
                for i in range(int(min_x), int(max_x) + 1):
                    for j in range(int(min_y), int(max_y) + 1):
                        px, py = transformer.transform(i, j)
                        if geom.contains(Point(px, py)):
                            array[i, j] = unique_value
                unique_value += 1

            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = [(int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[0]), 
                               int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[1])) 
                              for coord in poly.exterior.coords]
                    for x, y in coords:
                        if 0 <= x < array.shape[0] and 0 <= y < array.shape[1]:
                            array[x, y] = 0  # 경계를 0으로 설정

                    # 내부를 고유한 값으로 설정
                    min_x, min_y, max_x, max_y = poly.bounds
                    for i in range(int(min_x), int(max_x) + 1):
                        for j in range(int(min_y), int(max_y) + 1):
                            px, py = transformer.transform(i, j)
                            if poly.contains(Point(px, py)):
                                array[i, j] = unique_value
                    unique_value += 1

        return array
    
    
def load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, road_shp_file, watershed_basins_shp_file, channels_shp_file):
    rirsv = gpd.read_file(rirsv_shp_file)
    wkmstrm = gpd.read_file(wkmstrm_shp_file)
    road = gpd.read_file(road_shp_file)
    watershed_basins = gpd.read_file(watershed_basins_shp_file)
    channels = gpd.read_file(channels_shp_file)
    
    return rirsv, wkmstrm, road, watershed_basins, channels


class GISProcessor_old:
    def __init__(self, dem_path, area_path):
        self.dem_path = dem_path
        self.area_path = area_path
        self.dem = rasterio.open(dem_path)
        self.area = gpd.read_file(area_path)
    
    def transform_shapefile_to_dem(self, shapefile):
        '''
        shapefile을 dem 영역에 맞춰 변환
        '''
        dem_array = self.dem.read(1)
        dem_transform = self.dem.transform
        shapefile_crs = shapefile.crs
        dem_crs = self.dem.crs
        transformer = Transformer.from_crs(shapefile_crs, dem_crs, always_xy=True)

        array = np.zeros_like(dem_array)
        for geom in shapefile.geometry:
            if geom.geom_type == 'Polygon':
                coords = [(int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[0]), 
                           int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[1])) 
                          for coord in geom.exterior.coords]
                for x, y in coords:
                    if 0 <= x < array.shape[0] and 0 <= y < array.shape[1]:
                        array[x, y] = 1
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = [(int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[0]), 
                               int(rowcol(dem_transform, *transformer.transform(coord[0], coord[1]))[1])) 
                              for coord in poly.exterior.coords]
                    for x, y in coords:
                        if 0 <= x < array.shape[0] and 0 <= y < array.shape[1]:
                            array[x, y] = 1
        return array

def load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, road_shp_file, watershed_basins_shp_file, channels_shp_file):
    rirsv = gpd.read_file(rirsv_shp_file)
    wkmstrm = gpd.read_file(wkmstrm_shp_file)
    road = gpd.read_file(road_shp_file)
    watershed_basins = gpd.read_file(watershed_basins_shp_file)
    channels = gpd.read_file(channels_shp_file)
    
    return rirsv, wkmstrm, road, watershed_basins, channels
