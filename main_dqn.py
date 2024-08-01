import os
import numpy as np
from geo_processing import GISProcessor, load_shapefiles
from reward import RewardCalculator
from method.Dqn import DQN, dqn_learning, simulate_path, load_model,Agent
from config import *
from utils import show_path_with_arrows, get_random_index

if __name__ == "__main__":
    filename = 'featured_dem.npy'
    num_iterations = 10  # 학습 반복 횟수

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
    
    action_mode = 'custom'  # or '8_directions'
    #action_mode = '8_directions'
    # Agent 인스턴스 생성
    agent = Agent(age_group='young', gender='male', health_status='good')

    dqn_learning(dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed, reward_calculator, agent, action_mode=action_mode)
j
    # 경로 시뮬레이션 예시
    test_area = np.load(filename)
    coord = get_random_index(test_area)
    start_x, start_y = coord[0], coord[1]

    model = load_model('dqn_model.pth', input_dim=10, output_dim=8 if action_mode == '8_directions' else 6)
    path = simulate_path(start_x, start_y, model, dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed, agent, action_mode=action_mode)
    
    print("Simulated Path:")
    print(path)

    show_path_with_arrows(dem_array, path)
