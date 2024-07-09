import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.transform import rowcol, from_bounds
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
    
    def reward_function(self, state):
            '''
            보상 함수: 보상 및 패널티를 구체화
            '''
            x, y, elevation, slope, rirsv, wkmstrm, road, watershed_basins, channels = state
            
            reward = -1  # 기본 패널티

            # 도로에 도달하면 높은 보상
            if road:
                reward += 20
            # 강이나 경사가 큰 지역에 있으면 큰 패널티
            if rirsv:
                reward -= 10
            if slope > 0.5:
                reward -= 10
            # 작은 강(개천) 근처에 있으면 중간 패널티
            if wkmstrm:
                reward += 5
            # 워터셰드 채널에 있으면 보상
            if channels:
                reward -= 5

            return reward

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

if __name__ == "__main__":
    from geo_processing import GISProcessor, load_shapefiles
    from q_learning import bayesian_q_learning, simulate_path
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from rasterio.plot import show
    import pandas as pd
    from utils import array_2_plot
    
    extracted_path = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704'
    dem_path = os.path.join(extracted_path, 'dem/dem.tif')
    area_path = os.path.join(extracted_path, 'area/area.shp')

    # GISProcessor 클래스 인스턴스 생성
    processor = GISProcessor(dem_path, area_path)

    # DEM 데이터를 area의 경계에 맞춰 클리핑
    dem_array, dem_transform, dem_meta = processor.clip_dem_to_area()

    # shapefiles 로드
    rirsv, wkmstrm, road, watershed_basins, channels = load_shapefiles()

    # 각 shapefile을 변환된 DEM 영역에 맞춰 변환
    rirsv_transformed = processor.transform_shapefile_to_dem(rirsv)
    wkmstrm_transformed = processor.transform_shapefile_to_dem(wkmstrm)
    road_transformed = processor.transform_shapefile_to_dem(road)
    watershed_basins_transformed = processor.transform_shapefile_to_dem(watershed_basins)
    channels_transformed = processor.transform_shapefile_to_dem(channels)

    
    array_2_plot(road_transformed)
    