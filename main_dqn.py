import os
import numpy as np
from geo_processing import GISProcessor, load_shapefiles
from reward import RewardCalculator
from method.Dqn import dqn_learning
from config import *



if __name__ == "__main__":
    filename = 'featured_dem.npy'
    '''
    slice_range = (slice(110, 120), slice(110, 120), slice(None))  # Example: rows 110-119, columns 110-119, all channels

    print(f"Checking {filename}")
    load_and_print_npy(filename, slice_range)
    print("\n")
    '''
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

    # 보상 계산기 인스턴스 생성
    reward_calculator = RewardCalculator(dem_array, rirsv_transformed, wkmstrm_transformed, road_transformed, watershed_basins_transformed, channels_transformed)

    # DQN 학습 수행
    dqn_learning(dem_array, rirsv_transformed, wkmstrm_transformed, road_transformed, watershed_basins_transformed, channels_transformed, reward_calculator)
