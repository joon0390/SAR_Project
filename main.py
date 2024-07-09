from geo_processing import GISProcessor, load_shapefiles
from q_learning import bayesian_q_learning, simulate_path
import os
import numpy as np

if __name__ == "__main__":
    # 파일 경로 설정
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
    rirsv_array = processor.transform_shapefile_to_dem(rirsv, dem_array, dem_transform)
    wkmstrm_array = processor.transform_shapefile_to_dem(wkmstrm, dem_array, dem_transform)
    road_array = processor.transform_shapefile_to_dem(road, dem_array, dem_transform)
    watershed_basins_array = processor.transform_shapefile_to_dem(watershed_basins, dem_array, dem_transform)
    channels_array = processor.transform_shapefile_to_dem(channels, dem_array, dem_transform)

    # Q-러닝 수행
    q_mean = bayesian_q_learning(dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array)

    # 경로 시뮬레이션
    start_x, start_y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
    path = simulate_path(start_x, start_y, q_mean, dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array)

    # 경로 출력
    print("Path taken by the agent:")
    print(path)
