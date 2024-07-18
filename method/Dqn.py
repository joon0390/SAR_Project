import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_elevation, calculate_slope
from config import *
import os

# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, input_dim=9, output_dim=6):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def save_model(model, filename='dqn_model.pth'):
    torch.save(model.state_dict(), filename)

def load_model(filename='dqn_model.pth', input_dim=9, output_dim=6):
    model = DQN(input_dim, output_dim)
    model.load_state_dict(torch.load(filename))
    return model

def dqn_learning(dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array, reward_calculator, load_existing=False, model_filename='dqn_model.pth'):
    global epsilon

    state_size = 9  # 상태 크기 (9개의 요소로 구성된 튜플)
    action_size = 6  # 행동 크기 (6개의 행동)
    
    if load_existing and os.path.exists(model_filename):
        model = load_model(model_filename, state_size, action_size)
        print(f"Loaded existing model from {model_filename}")
    else:
        model = DQN(state_size, action_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epsilon_start = epsilon
    epsilon_end = 0.1
    epsilon_decay = 0.995
    epsilon = epsilon_start

    for episode in range(episodes):
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        reward_calculator.start_x, reward_calculator.start_y = x, y
        state = torch.tensor([x, y, get_elevation(dem_array[:, :], x, y), calculate_slope(dem_array[:, :], x, y),
                              rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y],
                              watershed_basins_array[x, y], channels_array[x, y]], dtype=torch.float32)
        reward_calculator.state_buffer.clear()  # 에피소드 시작 시 버퍼 초기화
        done = False
        step = 0
        prev_path = [(x, y)]

        while not done and step < max_steps:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(action_size)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()

            next_x, next_y = x, y
            if action == 0:
                next_x, next_y = (x + np.random.choice([-speed, speed]), y + np.random.choice([-speed, speed]))
            elif action == 1:
                if road_array[x, y]:
                    next_x, next_y = (x + np.random.choice([-speed, speed]), y)
                else:
                    next_x, next_y = (x, y + np.random.choice([-speed, speed]))
            elif action == 2:
                direction = np.random.choice(['up', 'down', 'left', 'right'])
                if direction == 'up':
                    next_x, next_y = (max(1, x - speed), y)
                elif direction == 'down':
                    next_x, next_y = (min(dem_array.shape[0] - 2, x + speed), y)
                elif direction == 'left':
                    next_x, next_y = (x, max(1, y - speed))
                elif direction == 'right':
                    next_x, next_y = (x, min(dem_array.shape[1] - 2, y + speed))
            elif action == 3:
                next_x, next_y = x, y
            elif action == 4:
                highest_elevation = get_elevation(dem_array[:, :], x, y)
                highest_coord = (x, y)
                for i in range(-speed, speed + 1):
                    for j in range(-speed, speed + 1):
                        if 0 <= x + i < dem_array.shape[0] and 0 <= y + j < dem_array.shape[1]:
                            elevation = get_elevation(dem_array[:, :], x + i, y + j)
                            if elevation > highest_elevation:
                                highest_elevation = elevation
                                highest_coord = (x + i, y + j)
                next_x, next_y = highest_coord
            elif action == 5 and len(prev_path) > 1:
                next_x, next_y = prev_path[-2]

            next_x = min(max(next_x, 0), dem_array.shape[0] - 1)
            next_y = min(max(next_y, 0), dem_array.shape[1] - 1)

            next_state = torch.tensor([next_x, next_y, get_elevation(dem_array[:, :], next_x, next_y), calculate_slope(dem_array[:, :], next_x, next_y),
                                       rirsv_array[next_x, next_y], wkmstrm_array[next_x, next_y], road_array[next_x, next_y],
                                       watershed_basins_array[next_x, next_y], channels_array[next_x, next_y]], dtype=torch.float32)

            reward = reward_calculator.reward_function(next_state)

            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + gamma * torch.max(model(next_state)).item()

            q_values = model(state)
            q_values[action] = target

            loss = criterion(model(state), q_values.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            x, y = next_x, next_y
            prev_path.append((x, y))

            if step % 10 == 0:
                with torch.no_grad():
                    expected_reward = torch.max(model(state)).item()
                print(f"Episode {episode}, Step {step}, State ({x}, {y}), Expected Reward: {expected_reward:.2f}")

            step += 1

            if road_array[x, y] or step >= max_steps:
                done = True

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    save_model(model, filename=model_filename)

    return model

def simulate_path(start_x, start_y, model, dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array):
    path = [(start_x, start_y)]
    x, y = start_x, start_y
    max_steps = simulation_max_steps

    for step in range(max_steps):
        state = torch.tensor([x, y, get_elevation(dem_array[:, :], x, y), calculate_slope(dem_array[:, :], x, y),
                              rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y],
                              watershed_basins_array[x, y], channels_array[x, y]], dtype=torch.float32)
        with torch.no_grad():
            action = torch.argmax(model(state)).item()

        if action == 0:
            next_x, next_y = (x + np.random.choice([-1, 1]), y + np.random.choice([-1, 1]))
        elif action == 1:
            if road_array[x, y]:
                next_x, next_y = (x + np.random.choice([-1, 1]), y)
            else:
                next_x, next_y = (x, y + np.random.choice([-1, 1]))
        elif action == 2:
            direction = np.random.choice(['up', 'down', 'left', 'right'])
            if direction == 'up':
                next_x, next_y = (max(1, x - 1), y)
            elif direction == 'down':
                next_x, next_y = (min(dem_array.shape[0] - 2, x + 1), y)
            elif direction == 'left':
                next_x, next_y = (x, max(1, y - 1))
            elif direction == 'right':
                next_x, next_y = (x, min(y + 1, dem_array.shape[1] - 1))
        elif action == 3:
            next_x, next_y = x, y
        elif action == 4:
            highest_elevation = 0
            highest_coord = (x, y)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= x + i < dem_array.shape[0] and 0 <= y + j < dem_array.shape[1]:
                        elevation = get_elevation(dem_array[:, :, 0], x + i, y + j)
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
