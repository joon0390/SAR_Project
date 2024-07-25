import os
import numpy as np
from geo_processing import GISProcessor, load_shapefiles
from reward import RewardCalculator
from method.Dqn import DQN, dqn_learning, simulate_path, load_model,Agent
from config import *
from utils import show_path_with_arrows, get_random_point_within_polygon

if __name__ == "__main__":
    filename = 'featured_dem.npy'

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
    climbpath_transformed = combined_array[:, :, 3]
    road_transformed = combined_array[:, :, 4]
    watershed_basins_transformed = combined_array[:, :, 5]
    channels_transformed = combined_array[:, :, 6]

    # 보상 계산기 인스턴스 생성
    reward_calculator = RewardCalculator(dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed)
    
    action_mode = '8_directions'  # or '8_directions'

    # Agent 인스턴스 생성∆
    agent = Agent(age_group='young', gender='male', health_status='good')

    # DQN 학습 수행
    dqn_learning(dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed, reward_calculator, agent, action_mode=action_mode)

    # 경로 시뮬레이션 예시
    #start_x, start_y = get_random_point_within_polygon(area_difference_file, dem_array.shape)
    start_x, start_y = 2545,3154
    model = load_model('dqn_model.pth', input_dim=10, output_dim=8 if action_mode == '8_directions' else 6)
    path = simulate_path(start_x, start_y, model, dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed, agent, action_mode=action_mode)
    
    print("Simulated Path:")
    print(path)

    show_path_with_arrows(dem_array, path)
