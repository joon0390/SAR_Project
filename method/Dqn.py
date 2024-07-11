import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from utils import get_elevation, calculate_slope

# 신경망 아키텍처 정의
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 하이퍼 파라미터 설정
alpha = 0.001  # 학습률
gamma = 0.9  # 할인 인자
epsilon = 0.8  # 탐험 vs 활용 비율
beta = 0.01  # 불확실성에 대한 가중치
replay_buffer = deque(maxlen=10000)
batch_size = 64
input_dim = 9  # 상태의 차원
output_dim = 6  # 행동의 차원

# 모델과 옵티마이저 초기화
model = DQN(input_dim, output_dim)
target_model = DQN(input_dim, output_dim)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

def is_done(state):
    x, y = state[:2]
    # 예시 조건: 특정 위치에 도달하거나 최대 스텝 수에 도달했을 때 에피소드 종료
    # 이 조건은 필요에 따라 변경할 수 있습니다.
    if (x == 0 and y == 0) or (x == state[0].shape[0]-1 and y == state[0].shape[1]-1):
        return True
    return False

def train_model():
    if len(replay_buffer) < batch_size:
        return
    
    minibatch = random.sample(replay_buffer, batch_size)
    state_batch = torch.tensor([s[0] for s in minibatch], dtype=torch.float32)
    action_batch = torch.tensor([s[1] for s in minibatch])
    reward_batch = torch.tensor([s[2] for s in minibatch])
    next_state_batch = torch.tensor([s[3] for s in minibatch], dtype=torch.float32)
    done_batch = torch.tensor([s[4] for s in minibatch])

    q_values = model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
    next_q_values = target_model(next_state_batch).max(1)[0]
    expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    loss = loss_fn(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def dqn_learning(dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array, reward_calculator):
    for episode in range(1000):
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        state = (x, y, get_elevation(x, y, dem_array), calculate_slope(x, y, dem_array),
                 rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y],
                 watershed_basins_array[x, y], channels_array[x, y])
        done = False
        step = 0

        while not done and step < 1000:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(6)
            else:
                with torch.no_grad():
                    action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()

            next_x, next_y = x, y
            if action == 0:  # 무작위 걷기 (Random Walking, RW)
                next_x, next_y = (x + np.random.choice([-1, 1]), y + np.random.choice([-1, 1]))
            elif action == 1:  # 경로 여행 (Route Traveling, RT)
                if road_array[x, y]:
                    next_x, next_y = (x + np.random.choice([-1, 1]), y)

            next_state = (next_x, next_y, get_elevation(next_x, next_y, dem_array), calculate_slope(next_x, next_y, dem_array),
                          rirsv_array[next_x, next_y], wkmstrm_array[next_x, next_y], road_array[next_x, next_y],
                          watershed_basins_array[next_x, next_y], channels_array[next_x, next_y])
            reward = reward_calculator.calculate(state, action, next_state)
            done = is_done(next_state)

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            step += 1
            
            train_model()
        
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())
