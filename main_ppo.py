import numpy as np
from method.ppo import PPOAgent, ppo_learning, simulate_path
from reward import RewardCalculator
from config import *
import os

def main():
    # 필요한 데이터 불러오기
    dem_array = np.load('dem_array.npy')
    rirsv_array = np.load('rirsv_array.npy')
    wkmstrm_array = np.load('wkmstrm_array.npy')
    climbpath_array = np.load('climbpath_array.npy')
    watershed_basins_array = np.load('watershed_basins_array.npy')
    channels_array = np.load('channels_array.npy')

    agent = Agent(age_group='young', gender='male', health_status='good')

    # RewardCalculator 정의
    reward_calculator = RewardCalculator()

    # 모델 학습
    model_filename = 'ppo_model.pth'
    load_existing = os.path.exists(model_filename)
    model = ppo_learning(dem_array, rirsv_array, wkmstrm_array, climbpath_array, watershed_basins_array, channels_array, reward_calculator, agent, action_mode='custom', load_existing=load_existing, model_filename=model_filename)

    # 경로 시뮬레이션
    start_x, start_y = 3110, 2647
    path = simulate_path(start_x, start_y, model, dem_array, rirsv_array, wkmstrm_array, climbpath_array, watershed_basins_array, channels_array, agent, action_mode='custom')
    print(f"Simulated Path: {path}")

if __name__ == "__main__":
    main()
