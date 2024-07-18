from config import *
from geo_processing import GISProcessor, load_shapefiles
from method.q_learning import bayesian_q_learning, simulate_path
import numpy as np
from reward import RewardCalculator
from utils import show_path_with_arrows, load_and_print_npy
import os

if __name__ == "__main__":
    filename = '/Users/heekim/Documents/GitHub/SAR_Project/featured_dem.npy'

    # .npy 파일을 로드
    if os.path.exists(filename):
        combined_array = np.load(filename)
        print(f"Loaded combined array from {filename}")
        print(f"Combined array shape: {combined_array.shape}")
    else:
        print(f"{filename} does not exist. Please ensure the file is available.")
        exit(1)

    # 채널을 각각 분리
    dem_array = combined_array[:, :, 0]
    rirsv_transformed = combined_array[:, :, 1]
    wkmstrm_transformed = combined_array[:, :, 2]
    road_transformed = combined_array[:, :, 3]
    watershed_basins_transformed = combined_array[:, :, 4]
    channels_transformed = combined_array[:, :, 5]

    # RewardCalculator 클래스 인스턴스 생성
    reward_calculator = RewardCalculator(dem_array, rirsv_transformed, wkmstrm_transformed, road_transformed, watershed_basins_transformed, channels_transformed)

    # Q-러닝 수행
    q_mean, q_variance = bayesian_q_learning(dem_array, rirsv_transformed, wkmstrm_transformed, road_transformed, watershed_basins_transformed, channels_transformed, reward_calculator)
    
    # 경로 시뮬레이션
    start_x, start_y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
    path = simulate_path(start_x, start_y, q_mean, dem_array, rirsv_transformed, wkmstrm_transformed, road_transformed, watershed_basins_transformed, channels_transformed)

    # 경로 출력
    print("Path taken by the agent:")
    print(path)
    
    print("finished")

    # 경로를 DEM 위에 화살표로 시각화
    show_path_with_arrows(dem_array, path)
