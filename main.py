from config import *
from geo_processing import GISProcessor, load_shapefiles
from method.q_learning import bayesian_q_learning, simulate_path
import numpy as np
from reward import RewardCalculator
from utils import show_path_with_arrows

if __name__ == "__main__":
    # GISProcessor 클래스 인스턴스 생성
    processor = GISProcessor(dem_path)

    # DEM 데이터를 로드
    dem_array = processor.dem.read(1)
    dem_transform = processor.dem.transform
    
    # shapefiles 로드
    rirsv, wkmstrm, road, watershed_basins, channels = load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, road_shp_file, watershed_basins_shp_file, channels_shp_file)

    # 각 shapefile을 변환된 DEM 영역에 맞춰 변환
    rirsv_transformed = processor.preprocess_rirsv(rirsv) #호수 내부는 0, 외부는 1
    watershed_basins_transformed = processor.preprocess_watershed(watershed_basins) # 같은 watershed는 0이 아닌 고유한 값, watershed의 경계는 0
    wkmstrm_transformed = processor.preprocess_wkmstrm(wkmstrm)
    road_transformed = processor.transform_shapefile_to_dem(road)
    channels_transformed = processor.transform_shapefile_to_dem(channels)

    # RewardCalculator 클래스 인스턴스 생성
    reward_calculator = RewardCalculator(dem_array, rirsv_transformed, wkmstrm_transformed, road_transformed, watershed_basins_transformed, channels_transformed)

    # Q-러닝 수행
    q_mean = bayesian_q_learning(dem_array, rirsv_transformed, wkmstrm_transformed, road_transformed, watershed_basins_transformed, channels_transformed, reward_calculator)
    
    # 경로 시뮬레이션
    start_x, start_y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
    path = simulate_path(start_x, start_y, q_mean, dem_array, rirsv_transformed, wkmstrm_transformed, road_transformed, watershed_basins_transformed, channels_transformed)

    # 경로 출력
    print("Path taken by the agent:")
    print(path)
    
    print("finished")

    # 경로를 DEM 위에 화살표로 시각화
    show_path_with_arrows(dem_array, path)
