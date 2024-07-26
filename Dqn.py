import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_elevation, calculate_slope
from collections import defaultdict  # 추가
import os
from reward import RewardCalculator
from geo_processing import GISProcessor, load_shapefiles
from reward import RewardCalculator
from config import *
from utils import show_path_with_arrows, get_random_point_within_polygon



class Agent:
    def __init__(self, age_group='young', gender='male', health_status='good'):
        self.age_group = age_group
        self.gender = gender
        self.health_status = health_status
        self.set_speed_and_explore_ratio()

    def set_speed_and_explore_ratio(self):
        if self.age_group == 'young':
            self.speed = 2
            self.explore_ratio = 0.8
        elif self.age_group == 'middle':
            self.speed = 1.5
            self.explore_ratio = 0.5
        elif self.age_group == 'old':
            self.speed = 1
            self.explore_ratio = 0.2

        if self.health_status == 'good':
            self.stay_put_probability = 0.1
        elif self.health_status == 'bad':
            self.stay_put_probability = 0.5


class DQN(nn.Module):
    def __init__(self, input_dim=12, output_dim=8):
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

def load_model(filename='dqn_model.pth', input_dim=12, output_dim=8):
    model = DQN(input_dim, output_dim)
    model.load_state_dict(torch.load(filename))
    return model

def dqn_learning(dem_array, rirsv_array, wkmstrm_array, climbpath_array, road_array, watershed_basins_array, channels_array,forestroad_array,hiking_array, reward_calculator, agent, action_mode='8_directions', load_existing=False, model_filename='dqn_model.pth'):
    state_size = 12 # 상태 크기 (9개의 요소로 구성된 튜플)
    if action_mode == '8_directions':
        action_size = 8  # 행동 크기 (8개의 행동)
    elif action_mode == 'custom':
        action_size = 6  # 행동 크기 (6개의 행동)
    
    if load_existing and os.path.exists(model_filename):
        model = load_model(model_filename, state_size, action_size)
        print(f"Loaded existing model from {model_filename}")
    else:
        model = DQN(state_size, action_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    epsilon = epsilon_start

    for episode in range(episodes):
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        reward_calculator.start_x, reward_calculator.start_y = x, y
        state = torch.tensor([x, y, reward_calculator.get_elevation(x, y), reward_calculator.calculate_slope(x, y),
                              rirsv_array[x, y], wkmstrm_array[x, y], climbpath_array[x, y],
                              road_array[x, y], watershed_basins_array[x, y], channels_array[x, y],
                              forestroad_array[x,y],hiking_array[x,y]], dtype=torch.float32)
        reward_calculator.state_buffer.clear()  # 에피소드 시작 시 버퍼 초기화
        reward_calculator.visited_count.clear()  # 방문한 좌표 초기화
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

            if action_mode == '8_directions':
                if action == 0:  # 상
                    next_x, next_y = (x - 1, y)
                elif action == 1:  # 하
                    next_x, next_y = (x + 1, y)
                elif action == 2:  # 좌
                    next_x, next_y = (x, y - 1)
                elif action == 3:  # 우
                    next_x, next_y = (x, y + 1)
                elif action == 4:  # 좌상
                    next_x, next_y = (x - 1, y - 1)
                elif action == 5:  # 우상
                    next_x, next_y = (x - 1, y + 1)
                elif action == 6:  # 좌하
                    next_x, next_y = (x + 1, y - 1)
                elif action == 7:  # 우하
                    next_x, next_y = (x + 1, y + 1)
            
            elif action_mode == 'custom':
                if action == 0:  # 무작위 걷기 (Random Walking, RW)
                    next_x, next_y = (x + np.random.choice([-agent.speed, agent.speed]), y + np.random.choice([-agent.speed, agent.speed]))
                elif action == 1:  # 경로 여행 (Route Traveling, RT)
                    if climbpath_array[x, y]:
                        next_x, next_y = (x + np.random.choice([-agent.speed, agent.speed]), y)
                    else:
                        next_x, next_y = (x, y + np.random.choice([-agent.speed, agent.speed]))
                elif action == 2:  # 방향 여행 (Direction Traveling, DT)
                    direction = np.random.choice(['up', 'down', 'left', 'right'])
                    if direction == 'up':
                        next_x, next_y = (max(1, x - agent.speed), y)
                    elif direction == 'down':
                        next_x, next_y = (min(dem_array.shape[0] - 2, x + agent.speed), y)
                    elif direction == 'left':
                        next_x, next_y = (x, max(1, y - agent.speed))
                    elif direction == 'right':
                        next_x, next_y = (x, min(dem_array.shape[1] - 2, y + agent.speed))
                elif action == 3:  # 제자리에 머무르기 (Staying Put, SP)
                    next_x, next_y = x, y
                elif action == 4:  # 시야 확보 (View Enhancing, VE)
                    highest_elevation = reward_calculator.get_elevation(x, y)
                    highest_coord = (x, y)
                    for i in range(-agent.speed, agent.speed + 1):
                        for j in range(-agent.speed, agent.speed + 1):
                            if 0 <= x + i < dem_array.shape[0] and 0 <= y + j < dem_array.shape[1]:
                                elevation = reward_calculator.get_elevation(x + i, y + j)
                                if elevation > highest_elevation:
                                    highest_elevation = elevation
                                    highest_coord = (x + i, y + j)
                    next_x, next_y = highest_coord
                elif action == 5 and len(prev_path) > 1:  # 되돌아가기 (Backtracking, BT)
                    next_x, next_y = prev_path[-2]

            next_x = min(max(next_x, 0), dem_array.shape[0] - 1)
            next_y = min(max(next_y, 0), dem_array.shape[1] - 1)

            next_state = torch.tensor([next_x, next_y, reward_calculator.get_elevation(next_x, next_y), reward_calculator.calculate_slope(next_x, next_y),
                                       rirsv_array[next_x, next_y], wkmstrm_array[next_x, next_y], climbpath_array[next_x, next_y],
                                       road_array[next_x, next_y], watershed_basins_array[next_x, next_y], channels_array[next_x, next_y],
                                       forestroad_array[next_x,next_y],hiking_array[next_x,next_y]], dtype=torch.float32)

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

            reward_calculator.update_visited_count(x, y)  # 방문 횟수 업데이트

            if step % 10 == 0:
                with torch.no_grad():
                    expected_reward = torch.max(model(state)).item()
                print(f"Episode {episode}, Step {step}, State ({x}, {y}), Expected Reward: {expected_reward:.2f}")

            step += 1

            if climbpath_array[x, y] or step >= max_steps:
                done = True

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    save_model(model, filename=model_filename)

    return model

def simulate_path(start_x, start_y, model, dem_array, rirsv_array, wkmstrm_array, climbpath_array, road_array, watershed_basins_array, channels_array, forestroad_array,hiking_array,agent, action_mode='8_directions'):
    path = [(start_x, start_y)]
    x, y = start_x, start_y
    max_steps = simulation_max_steps

    visited_count = defaultdict(int)  # 방문한 좌표와 그 횟수를 저장할 딕셔너리

    for step in range(max_steps):
        state = torch.tensor([x, y, get_elevation(dem_array, x, y), calculate_slope(dem_array, x, y),
                              rirsv_array[x, y], wkmstrm_array[x, y], climbpath_array[x, y],
                              road_array[x, y], watershed_basins_array[x, y], channels_array[x, y],
                              forestroad_array[x,y],hiking_array[x,y]], dtype=torch.float32)
        with torch.no_grad():
            action = torch.argmax(model(state)).item()

        next_x, next_y = x, y

        if action_mode == '8_directions':
            if action == 0:
                next_x, next_y = (x - 1, y)
            elif action == 1:
                next_x, next_y = (x + 1, y)
            elif action == 2:
                next_x, next_y = (x, y - 1)
            elif action == 3:
                next_x, next_y = (x, y + 1)
            elif action == 4:
                next_x, next_y = (x - 1, y - 1)
            elif action == 5:
                next_x, next_y = (x - 1, y + 1)
            elif action == 6:
                next_x, next_y = (x + 1, y - 1)
            elif action == 7:
                next_x, next_y = (x + 1, y + 1)

        elif action_mode == 'custom':
            if action == 0:  # 무작위 걷기 (Random Walking, RW)
                next_x, next_y = (x + np.random.choice([-agent.speed, agent.speed]), y + np.random.choice([-agent.speed, agent.speed]))
            elif action == 1:  # 경로 여행 (Route Traveling, RT)
                if climbpath_array[x, y]:
                    next_x, next_y = (x + np.random.choice([-agent.speed, agent.speed]), y)
                else:
                    next_x, next_y = (x, y + np.random.choice([-agent.speed, agent.speed]))
            elif action == 2:  # 방향 여행 (Direction Traveling, DT)
                direction = np.random.choice(['up', 'down', 'left', 'right'])
                if direction == 'up':
                    next_x, next_y = (max(1, x - agent.speed), y)
                elif direction == 'down':
                    next_x, next_y = (min(dem_array.shape[0] - 2, x + agent.speed), y)
                elif direction == 'left':
                    next_x, next_y = (x, max(1, y - agent.speed))
                elif direction == 'right':
                    next_x, next_y = (x, min(y + agent.speed, dem_array.shape[1] - 1))
            elif action == 3:  # 제자리에 머무르기 (Staying Put, SP)
                next_x, next_y = x, y
            elif action == 4:  # 시야 확보 (View Enhancing, VE)
                highest_elevation = 0
                highest_coord = (x, y)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if 0 <= x + i < dem_array.shape[0] and 0 <= y + j < dem_array.shape[1]:
                            elevation = get_elevation(dem_array, x + i, y + j)
                            if elevation > highest_elevation:
                                highest_elevation = elevation
                                highest_coord = (x + i, y + j)
                next_x, next_y = highest_coord
            elif action == 5 and len(path) > 1:  # 되돌아가기 (Backtracking, BT)
                next_x, next_y = path[-2]

        next_x = min(max(next_x, 0), dem_array.shape[0] - 1)
        next_y = min(max(next_y, 0), dem_array.shape[1] - 1)

        path.append((next_x, next_y))

        visited_count[(next_x, next_y)] += 1  # 방문 횟수 업데이트

        x, y = next_x, next_y

        if climbpath_array[x, y]:
            break

    return path


