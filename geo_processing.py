import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer

class GISProcessor:
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
