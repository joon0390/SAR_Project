import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from utils import get_elevation, calculate_slope
from BQL import load_data, shapefile_to_array
import pickle
import os

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



model = DQN(input_dim, output_dim)
target_model = DQN(input_dim, output_dim)
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

def is_done(state):
    x, y = state[:2]
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

def dqn_learning(dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array, reward_calculator, load_existing=False, model_filename='dqn_model.pth'):
    if load_existing and os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename))
        print(f"Loaded existing model from {model_filename}")
    
    prev_path = []  # 이전 경로를 저장할 리스트 초기화

    for episode in range(1000):
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        state = (x, y, get_elevation(x, y, dem_array), calculate_slope(x, y, dem_array),
                 rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y],
                 watershed_basins_array[x, y], channels_array[x, y])
        done = False
        step = 0

        while not done and step < 1000:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(output_dim)
            else:
                with torch.no_grad():
                    action = model(torch.tensor(state, dtype=torch.float32)).argmax().item()

            next_x, next_y = x, y
            if action == 0:  # 무작위 걷기 (Random Walking, RW)
                next_x, next_y = (x + np.random.choice([-10, 10]), y + np.random.choice([-10, 10]))
            elif action == 1:  # 경로 여행 (Route Traveling, RT)
                if road_array[x, y]:
                    next_x, next_y = (x + np.random.choice([-10, 10]), y)
                else:
                    next_x, next_y = (x, y + np.random.choice([-10, 10]))
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

            next_state = (next_x, next_y, get_elevation(next_x, next_y, dem_array), calculate_slope(next_x, next_y, dem_array),
                          rirsv_array[next_x, next_y], wkmstrm_array[next_x, next_y], road_array[next_x, next_y],
                          watershed_basins_array[next_x, next_y], channels_array[next_x, next_y])
            reward = reward_calculator.calculate(state, action, next_state)
            done = is_done(next_state)

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            step += 1
            prev_path.append((next_x, next_y))  # 경로 저장
            
            train_model()
        
        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())
            torch.save(model.state_dict(), model_filename)
            print(f"Saved model to {model_filename}")

# Load data using functions from BQL.py
area, dem, rirsv, wkmstrm, road, watershed_basins, channels = load_data()

# Convert shapefiles to arrays
dem_array = dem.read(1)
rirsv_array = shapefile_to_array(rirsv, dem_array, dem.transform)
wkmstrm_array = shapefile_to_array(wkmstrm, dem_array, dem.transform)
road_array = shapefile_to_array(road, dem_array, dem.transform)
watershed_basins_array = shapefile_to_array(watershed_basins, dem_array, dem.transform)
channels_array = shapefile_to_array(channels, dem_array, dem.transform)

# Example reward calculator (replace with actual implementation)
class RewardCalculator:
    def calculate(self, state, action, next_state):
        return 1.0  # Placeholder implementation

reward_calculator = RewardCalculator()

# Call the learning function
dqn_learning(dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array, reward_calculator)
