import os
import numpy as np
from geo_processing import GISProcessor, load_shapefiles
from reward import RewardCalculator
from method.Dqn import DQN, dqn_learning, simulate_path, load_model, Agent
from config import *
from utils import show_path_with_arrows, get_random_point_within_polygon

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
    climbpath_transformed = combined_array[:, :, 3]
    watershed_basins_transformed = combined_array[:, :, 4]
    channels_transformed = combined_array[:, :, 5]

    # 보상 계산기 인스턴스 생성
    reward_calculator = RewardCalculator(dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, watershed_basins_transformed, channels_transformed)
    
    # Agent 인스턴스 생성
    agent = Agent(age_group='young', gender='male', health_status='good')

    # 기존 모델 파일이 있으면 삭제
    model_filename = 'dqn_model.pth'
    if os.path.exists(model_filename):
        os.remove(model_filename)
        print(f"Deleted existing model file: {model_filename}")
    
    # DQN 학습 수행
    dqn_learning(dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, watershed_basins_transformed, channels_transformed, reward_calculator, agent, action_mode='custom')
    
    model = load_model('dqn_model.pth', input_dim=9, output_dim=6)  # Make sure to adjust the input_dim and output_dim as needed

    # 경로 시뮬레이션 예시

    start_x, start_y = get_random_point_within_polygon(area_difference_file, dem_array.shape)
    if start_x is None or start_y is None:
        print("다각형 내에서 유효한 시작 지점을 찾지 못했습니다.")
        exit(1)

    print(f"Starting coordinates: ({start_x}, {start_y})")  # 시작 좌표 출력

    # 함수 호출
    path = simulate_path(start_x, start_y, model, dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, watershed_basins_transformed, channels_transformed, agent=agent, action_mode='custom')

    print("모의 경로:")
    print(path)

    show_path_with_arrows(dem_array, path)

    