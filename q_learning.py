import numpy as np

def get_elevation(x, y, dem_array):
    # 주어진 좌표의 고도 값 반환
    return dem_array[x, y]

def calculate_slope(x, y, dem_array):
    # 주어진 좌표의 경사 값 계산
    if x <= 0 or x >= dem_array.shape[0] - 1 or y <= 0 or y >= dem_array.shape[1] - 1:
        return 0
    dzdx = (dem_array[x + 1, y] - dem_array[x - 1, y]) / 2
    dzdy = (dem_array[x, y + 1] - dem_array[x, y - 1]) / 2
    slope = np.sqrt(dzdx**2 + dzdy**2)
    return slope

def reward_function(state):
    # 보상 함수: 보상 및 패널티를 구체화
    x, y, elevation, slope, rirsv, wkmstrm, road, watershed_basins, channels = state
    
    reward = -1  # 기본 패널티

    # 도로에 도달하면 높은 보상
    if road:
        reward += 20
    # 강이나 경사가 큰 지역에 있으면 큰 패널티
    if rirsv:
        reward -= 10
    if slope > 0.5:
        reward -= 10
    # 작은 강(개천) 근처에 있으면 중간 패널티
    if wkmstrm:
        reward -= 5
    # 워터셰드 채널에 있으면 보상
    if channels:
        reward += 5

    return reward

def discretize_state(state, q_mean):
    # 상태를 디스크리트 상태로 변환
    x, y = state[:2]
    max_x, max_y = q_mean.shape[0] - 1, q_mean.shape[1] - 1
    return min(x // 10, max_x), min(y // 10, max_y)

def bayesian_q_learning(dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array):
    # 베이지안 Q-러닝 파라미터 초기화
    q_mean = np.zeros((dem_array.shape[0] // 10, dem_array.shape[1] // 10, 6))
    q_variance = np.ones((dem_array.shape[0] // 10, dem_array.shape[1] // 10, 6))
    alpha = 0.1  # 학습률
    gamma = 0.9  # 할인 인자
    epsilon = 0.9  # 탐험 vs 활용 비율
    beta = 0.01  # 불확실성에 대한 가중치
    prev_path = []

    for episode in range(1000):
        # 에피소드 초기 상태 무작위 설정
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        state = (x, y, get_elevation(x, y, dem_array), calculate_slope(x, y, dem_array),
                 rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y],
                 watershed_basins_array[x, y], channels_array[x, y])
        done = False
        step = 0
        prev_path.append((x, y))
        
        while not done and step < 100:
            # 현재 상태를 디스크리트 상태로 변환
            discretized_state = discretize_state(state, q_mean)
            
            # Epsilon-Greedy 정책으로 행동 선택
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(6)
            else:
                q_values = q_mean[discretized_state] + beta * np.sqrt(q_variance[discretized_state])
                action = np.argmax(q_values)

            # 행동에 따른 다음 상태 결정
            next_x, next_y = x, y
            if action == 1:
                next_x, next_y = (x + np.random.choice([-1, 1]), y) if road_array[x, y] else (x, y + np.random.choice([-1, 1]))
            elif action == 2:
                direction = np.random.choice(['up', 'down', 'left', 'right'])
                if direction == 'up':
                    next_x, next_y = (max(1, x - 1), y)
                elif direction == 'down':
                    next_x, next_y = (min(dem_array.shape[0] - 2, x + 1), y)
                elif direction == 'left':
                    next_x, next_y = (x, max(1, y - 1))
                elif direction == 'right':
                    next_x, next_y = (x, min(dem_array.shape[1] - 2, y + 1))
            elif action == 4:
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
            elif action == 5 and len(prev_path) > 1:
                next_x, next_y = prev_path[-2]
            
            # 다음 상태의 좌표가 유효한지 확인
            next_x = min(max(next_x, 0), dem_array.shape[0] - 1)
            next_y = min(max(next_y, 0), dem_array.shape[1] - 1)

            next_state = (next_x, next_y, get_elevation(next_x, next_y, dem_array), calculate_slope(next_x, next_y, dem_array),
                          rirsv_array[next_x, next_y], wkmstrm_array[next_x, next_y], road_array[next_x, next_y],
                          watershed_basins_array[next_x, next_y], channels_array[next_x, next_y])

            next_discretized_state = discretize_state(next_state, q_mean)
            reward = reward_function(next_state)

            # Q-값 업데이트
            q_mean[discretized_state][action] = (1 - alpha) * q_mean[discretized_state][action] + alpha * (reward + gamma * np.max(q_mean[next_discretized_state]))
            q_variance[discretized_state][action] = (1 - alpha) * q_variance[discretized_state][action] + alpha * ((reward + gamma * np.max(q_mean[next_discretized_state]) - q_mean[discretized_state][action]) ** 2)

            prev_path.append((next_x, next_y))
            state = next_state
            x, y = next_x, next_y

            if step % 10 == 0:
                expected_reward = np.max(q_mean[discretized_state])
                print(f"Episode {episode}, Step {step}, State ({x}, {y}), Expected Reward: {expected_reward:.2f}")

            step += 1

            if road_array[x, y]:
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
        state = (x, y, get_elevation(x, y, dem_array), calculate_slope(x, y, dem_array),
                 rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y],
                 watershed_basins_array[x, y], channels_array[x, y])
        discretized_state = discretize_state(state)
        action = np.argmax(q_mean[discretized_state])

        if action == 0:
            next_x, next_y = (x, y)
        elif action == 1:
            next_x, next_y = (x + np.random.choice([-1, 1]), y) if road_array[x, y] else (x, y + np.random.choice([-1, 1]))
        elif action == 2:
            direction = np.random.choice(['up', 'down', 'left', 'right'])
            if direction == 'up':
                next_x, next_y = (max(1, x - 1), y)
            elif direction == 'down':
                next_x, next_y = (min(dem_array.shape[0] - 2, x + 1), y)
            elif direction == 'left':
                next_x, next_y = (x, max(1, y - 1))
            elif direction == 'right':
                next_x, next_y = (x, min(dem_array.shape[1] - 2, y + 1))
        elif action == 4:
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
        elif action == 5 and len(path) > 1:
            next_x, next_y = path[-2]

        next_x = min(max(next_x, 0), dem_array.shape[0] - 1)
        next_y = min(max(next_y, 0), dem_array.shape[1] - 1)

        path.append((next_x, next_y))

        x, y = next_x, next_y

        if road_array[x, y]:
            break

    return path
