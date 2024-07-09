import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds, rowcol
from pyproj import Transformer
import os

class GISProcessor:
    def __init__(self, dem_path, area_path):
        self.dem_path = dem_path
        self.area_path = area_path
        self.dem = rasterio.open(dem_path)
        self.area = gpd.read_file(area_path)
    
    def clip_dem_to_area(self):
        '''
        area.shp 파일의 경계를 기준으로 DEM 데이터를 클리핑
        '''
        area_geometry = [self.area.geometry.unary_union]
        out_image, out_transform = mask(self.dem, area_geometry, crop=True)
        out_meta = self.dem.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        
        return out_image[0], out_transform, out_meta

    def transform_shapefile_to_dem(self, shapefile, dem_array, dem_transform):
        '''
        shapefile을 dem 영역에 맞춰 변환
        '''
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

def load_shapefiles():
    extracted_path = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704'

    rirsv_shp_file = os.path.join(extracted_path, 'river/it_c_rirsv.shp')
    wkmstrm_shp_file = os.path.join(extracted_path, 'river/lt_c_wkmstrm.shp')
    road_shp_file = os.path.join(extracted_path, 'road/lt_l_frstclimb.shp')
    watershed_basins_shp_file = os.path.join(extracted_path, 'watershed/basins.shp')
    channels_shp_file = os.path.join(extracted_path, 'watershed/channels.shp')

    rirsv = gpd.read_file(rirsv_shp_file)
    wkmstrm = gpd.read_file(wkmstrm_shp_file)
    road = gpd.read_file(road_shp_file)
    watershed_basins = gpd.read_file(watershed_basins_shp_file)
    channels = gpd.read_file(channels_shp_file)
    
    return rirsv, wkmstrm, road, watershed_basins, channels
