import numpy as np
from utils import get_elevation, calculate_slope, discretize_state

# 하이퍼 파라미터 설정
alpha = 0.2  # 학습률
gamma = 0.9  # 할인 인자
epsilon = 0.8  # 탐험 vs 활용 비율
beta = 0.01  # 불확실성에 대한 가중치

def bayesian_q_learning(dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array, processor):
    # 베이지안 Q-러닝 파라미터 초기화
    q_mean = np.zeros((dem_array.shape[0] // 10, dem_array.shape[1] // 10, 6))  # Q-값의 평균을 저장할 배열 초기화
    q_variance = np.ones((dem_array.shape[0] // 10, dem_array.shape[1] // 10, 6))  # Q-값의 분산을 저장할 배열 초기화
    prev_path = []  # 이전 경로를 저장할 리스트 초기화

    for episode in range(1000):
        # 에피소드 초기 상태 무작위 설정
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        processor.start_x, processor.start_y = x, y  # 시작 좌표 저장
        # 현재 상태를 정의
        state = (x, y, get_elevation(x, y, dem_array), calculate_slope(x, y, dem_array),
                 rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y],
                 watershed_basins_array[x, y], channels_array[x, y])
        done = False  # 에피소드 완료 여부
        step = 0  # 현재 에피소드의 스텝 수
        prev_path.append((x, y))  # 초기 좌표를 이전 경로에 추가
        
        while not done and step < 100:
            # 현재 상태를 디스크리트 상태로 변환
            discretized_state = discretize_state(state, q_mean)
            
            # Epsilon-Greedy 정책으로 행동 선택
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(6)  # 탐험: 무작위로 행동 선택
            else:
                q_values = q_mean[discretized_state] + beta * np.sqrt(q_variance[discretized_state])
                action = np.argmax(q_values)  # 활용: Q-값이 최대인 행동 선택

            # 행동에 따른 다음 상태 결정
            next_x, next_y = x, y
            if action == 0:  # 무작위 걷기 (Random Walking, RW)
                next_x, next_y = (x + np.random.choice([-1, 1]), y + np.random.choice([-1, 1]))
            elif action == 1:  # 경로 여행 (Route Traveling, RT)
                if road_array[x, y]:
                    next_x, next_y = (x + np.random.choice([-1, 1]), y)
                else:
                    next_x, next_y = (x, y + np.random.choice([-1, 1]))
            elif action == 2:  # 방향 여행 (Direction Traveling, DT)
                direction = np.random.choice(['up', 'down', 'left', 'right'])
                if direction == 'up':
                    next_x, next_y = (max(1, x - 1), y)
                elif direction == 'down':
                    next_x, next_y = (min(dem_array.shape[0] - 2, x + 1), y)
                elif direction == 'left':
                    next_x, next_y = (x, max(1, y - 1))
                elif direction == 'right':
                    next_x, next_y = (x, min(dem_array.shape[1] - 2, y + 1))
            elif action == 3:  # 제자리에 머무르기 (Staying Put, SP)
                next_x, next_y = x, y
            elif action == 4:  # 시야 확보 (View Enhancing, VE)
                highest_elevation = get_elevation(x, y, dem_array)
                highest_coord = (x, y)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if 0 <= x + i < dem_array.shape[0] and 0 <= y + j < dem_array.shape[1]:
                            elevation = get_elevation(x + i, y + j, dem_array)
                            if elevation > highest_elevation:
                                highest_elevation = elevation
                                highest_coord = (x + i, y + j)
                next_x, next_y = highest_coord
            elif action == 5 and len(prev_path) > 1:  # 되돌아가기 (Backtracking, BT)
                next_x, next_y = prev_path[-2]
            
            # 다음 상태의 좌표가 유효한지 확인
            next_x = min(max(next_x, 0), dem_array.shape[0] - 1)
            next_y = min(max(next_y, 0), dem_array.shape[1] - 1)

            # 다음 상태 정의
            next_state = (next_x, next_y, get_elevation(next_x, next_y, dem_array), calculate_slope(next_x, next_y, dem_array),
                          rirsv_array[next_x, next_y], wkmstrm_array[next_x, next_y], road_array[next_x, next_y],
                          watershed_basins_array[next_x, next_y], channels_array[next_x, next_y])

            # 다음 상태를 디스크리트 상태로 변환
            next_discretized_state = discretize_state(next_state, q_mean)
            # 다음 상태에 대한 보상 계산
            reward = processor.reward_function(next_state)

            # Q-값 업데이트
            q_mean[discretized_state][action] = (1 - alpha) * q_mean[discretized_state][action] + alpha * (reward + gamma * np.max(q_mean[next_discretized_state]))
            q_variance[discretized_state][action] = (1 - alpha) * q_variance[discretized_state][action] + alpha * ((reward + gamma * np.max(q_mean[next_discretized_state]) - q_mean[discretized_state][action]) ** 2)

            prev_path.append((next_x, next_y))  # 다음 상태 좌표를 이전 경로에 추가
            state = next_state  # 상태 업데이트
            x, y = next_x, next_y  # 좌표 업데이트

            if step % 10 == 0:  # 매 10 스텝마다
                expected_reward = np.max(q_mean[discretized_state])  # 예상 보상 계산
                print(f"Episode {episode}, Step {step}, State ({x}, {y}), Expected Reward: {expected_reward:.2f}")

            step += 1  # 스텝 수 증가

            if road_array[x, y]:  # 도로에 도달하면 에피소드 완료
                done = True

    return q_mean

def simulate_path(start_x, start_y, q_mean, dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array):
    # 경로 시뮬레이션 함수: 주어진 시작점에서 학습된 Q-값을 사용하여 경로를 생성
    path = [(start_x, start_y)]
    x, y = start_x, start_y
    max_steps = 100

    def discretize_state(state):
        x, y = state[:2]
        max_x, max_y = q_mean.shape[0] - 1, q_mean.shape[1] - 1
        return min(x // 10, max_x), min(y // 10, max_y)

    for step in range(max_steps):
        # 현재 상태 정의
        state = (x, y, get_elevation(x, y, dem_array), calculate_slope(x, y, dem_array),
                 rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y],
                 watershed_basins_array[x, y], channels_array[x, y])
        discretized_state = discretize_state(state)  # 상태를 디스크리트 상태로 변환
        action = np.argmax(q_mean[discretized_state])  # Q-값이 최대인 행동 선택

        # 행동에 따른 다음 상태 결정
        if action == 0:  # 무작위 걷기 (Random Walking, RW)
            next_x, next_y = (x + np.random.choice([-1, 1]), y + np.random.choice([-1, 1]))
        elif action == 1:  # 경로 여행 (Route Traveling, RT)
            if road_array[x, y]:
                next_x, next_y = (x + np.random.choice([-1, 1]), y)
            else:
                next_x, next_y = (x, y + np.random.choice([-1, 1]))
        elif action == 2:  # 방향 여행 (Direction Traveling, DT)
            direction = np.random.choice(['up', 'down', 'left', 'right'])
            if direction == 'up':
                next_x, next_y = (max(1, x - 1), y)
            elif direction == 'down':
                next_x, next_y = (min(dem_array.shape[0] - 2, x + 1), y)
            elif direction == 'left':
                next_x, next_y = (x, max(1, y - 1))
            elif direction == 'right':
                next_x, next_y = (x, min(dem_array.shape[1] - 2, y + 1))
        elif action == 3:  # 제자리에 머무르기 (Staying Put, SP)
            next_x, next_y = x, y
        elif action == 4:  # 시야 확보 (View Enhancing, VE)
            highest_elevation = 0
            highest_coord = (x, y)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= x + i < dem_array.shape[0] and 0 <= y + j < dem_array.shape[1]:
                        elevation = get_elevation(x + i, y + j, dem_array)
                        if elevation > highest_elevation:
                            highest_elevation = elevation
                            highest_coord = (x + i, y + j)
            next_x, next_y = highest_coord
        elif action == 5 and len(path) > 1:  # 되돌아가기 (Backtracking, BT)
            next_x, next_y = path[-2]

        next_x = min(max(next_x, 0), dem_array.shape[0] - 1)
        next_y = min(max(next_y, 0), dem_array.shape[1] - 1)

        path.append((next_x, next_y))  # 경로에 추가

        x, y = next_x, next_y  # 현재 좌표 업데이트

        if road_array[x, y]:  # 도로에 도달하면 시뮬레이션 종료
            break

    return path
