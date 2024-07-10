
import numpy as np
from utils import get_elevation, calculate_slope

# 하이퍼 파라미터 설정
alpha = 0.2  # 학습률
gamma = 0.9  # 할인 인자
epsilon = 0.8  # 탐험 vs 활용 비율
beta = 0.01  # 불확실성에 대한 가중치

def bayesian_q_learning(dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array, processor):
    # 베이지안 Q-러닝 파라미터 초기화
    q_values = {}  # Q-값을 저장할 딕셔너리 초기화

    for episode in range(1000):
        # 에피소드 초기 상태 무작위 설정
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        processor.start_x, processor.start_y = x, y  # 시작 좌표 저장
        # 현재 상태를 정의
        state = (x, y, get_elevation(x, y, dem_array), calculate_slope(x, y, dem_array),
                 rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y])

        done = False
        while not done:
            # Q-값을 선택하거나 무작위 선택
            if np.random.rand() < epsilon:
                action = np.random.randint(6)
            else:
                q_values_for_state = q_values.get(state, np.zeros(6))
                action = np.argmax(q_values_for_state)

            # 환경에서 다음 상태 및 보상 관찰
            next_state, reward, done = processor.take_action(state, action)

            # Q-값 업데이트
            q_values_for_state = q_values.get(state, np.zeros(6))
            q_values_for_next_state = q_values.get(next_state, np.zeros(6))
            q_values_for_state[action] = q_values_for_state[action] + alpha * (reward + gamma * np.max(q_values_for_next_state) - q_values_for_state[action])

            q_values[state] = q_values_for_state
            state = next_state
