import os
import numpy as np
from geo_processing import GISProcessor, load_shapefiles
from reward import RewardCalculator
from method.dqn import dqn_learning
from config import *

def main():
    # GISProcessor 클래스 인스턴스 생성
    processor = GISProcessor(dem_path)

    # DEM 데이터를 불러오기
    dem_array = processor.dem.read(1)

    # shapefiles 로드
    rirsv, wkmstrm, road, watershed_basins, channels = load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, road_shp_file, watershed_basins_shp_file, channels_shp_file)

    # 각 shapefile을 변환된 DEM 영역에 맞춰 변환
    rirsv_transformed = processor.transform_shapefile_to_dem(rirsv)
    wkmstrm_transformed = processor.transform_shapefile_to_dem(wkmstrm)
    road_transformed = processor.transform_shapefile_to_dem(road)
    watershed_basins_transformed = processor.preprocess_watershed(watershed_basins)
    channels_transformed = processor.transform_shapefile_to_dem(channels)

    # 보상 계산기 인스턴스 생성
    reward_calculator = RewardCalculator(dem_array, rirsv_transformed, wkmstrm_transformed, road_transformed, watershed_basins_transformed, channels_transformed)

    # DQN 학습 수행
    dqn_learning(dem_array, rirsv_transformed, wkmstrm_transformed, road_transformed, watershed_basins_transformed, channels_transformed, reward_calculator)

if __name__ == "__main__":
    main()
