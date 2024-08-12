import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_elevation, calculate_slope
from collections import defaultdict
import os, json
from reward import RewardCalculator
from geo_processing import GISProcessor, load_shapefiles
from utils import show_path_with_arrows, get_random_index
from config import *
from collections import deque
import random
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            self.speed = 2
            self.explore_ratio = 0.6
        elif self.age_group == 'old':
            self.speed = 1
            self.explore_ratio = 0.2

        self.stay_put_probability = 0.0  # 제자리에 머무는 확률을 0으로 설정
        self.speed_mean = self.speed  # 속도 평균 설정
        self.speed_std = 1  # 속도 표준편차 설정

    def sample_speed(self):
        # Truncated normal distribution: a = (lower_bound - mean) / std, b = (upper_bound - mean) / std
        a, b = (1 - self.speed_mean) / self.speed_std, np.inf  # lower bound is 1 (speed cannot be less than 1)
        self.speed = truncnorm(a, b, loc=self.speed_mean, scale=self.speed_std).rvs()
    
    def update_speed(self, decay_factor=0.99):
        self.sample_speed()
        self.speed = self.speed * decay_factor

class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def get_latest_experience(self):
        return self.buffer[-1] if len(self.buffer) > 0 else None
    
    def __len__(self):
        return len(self.buffer)

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

def save_to_json(all_losses, filename):
    with open(filename, 'w') as f:
        json.dump(all_losses, f)

def load_losses_from_json(filename='losses.json'):
    with open(filename, 'r') as f:
        all_losses = json.load(f)
    return all_losses

def plot_losses_from_json(filename='losses.json'):
    with open(filename, 'r') as f:
        all_losses = json.load(f)
    
    # all_losses가 2D 리스트인 경우 1D 리스트로 변환
    if isinstance(all_losses, list) and isinstance(all_losses[0], list):
        all_losses = [loss for episode_losses in all_losses for loss in episode_losses]
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, label='Total Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over Episodes')
    plt.legend()
    plt.show()

def plot_rewards_from_json(filename='rewards.json'):
    with open(filename, 'r') as f:
        all_rewards = json.load(f)
    
    # all_rewards가 2D 리스트인 경우 1D 리스트로 변환
    if isinstance(all_rewards, list) and isinstance(all_rewards[0], list):
        all_rewards = [reward for episode_rewards in all_rewards for reward in episode_rewards]
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Episodes')
    plt.legend()
    plt.show()

def dqn_learning(dem_array, rirsv_array, wkmstrm_array, climbpath_array, road_array, watershed_basins_array, channels_array, forestroad_array, hiking_array, reward_calculator, agent, action_mode='custom', load_existing=False, model_filename='dqn_model.pth', _lr=0.001, _gamma=0.9, buffer_size=10000, max_steps=1000, episodes=500, target_update_freq=500, reward_function_index=1):
    state_size = 12  # 상태 크기 (12개의 요소로 구성된 튜플)
    if action_mode == '8_directions':
        action_size = 8  # 행동 크기 (8개의 행동)
    elif action_mode == 'custom':
        action_size = 5  # 행동 크기 (5개의 행동)
    
    if load_existing and os.path.exists(model_filename):
        model = load_model(model_filename, state_size, action_size).to(device)
        print(f"Loaded existing model from {model_filename}")
    else:
        model = DQN(state_size, action_size).to(device)
    
    target_model = DQN(state_size, action_size).to(device)
    target_model.load_state_dict(model.state_dict())  # 초기에는 주 네트워크와 동일하게 설정

    optimizer = optim.Adam(model.parameters(), lr=_lr)
    criterion = nn.HuberLoss()
    gamma = _gamma  # 할인 인자
    epsilon_start = agent.explore_ratio
    epsilon_end = 0.1
    epsilon_decay = 0.95
    epsilon = epsilon_start

    replay_buffer = ReplayBuffer(buffer_size)  # 리플레이 버퍼 생성

    agent.speed = int(agent.speed)  # agent.speed를 정수로 변환

    if reward_function_index == 1:
        reward_function = reward_calculator.reward_function1
    elif reward_function_index == 2:
        reward_function = reward_calculator.reward_function2
    elif reward_function_index == 3:
        reward_function = reward_calculator.reward_function3
    elif reward_function_index == 4:
        reward_function = reward_calculator.reward_function4
    elif reward_function_index == 5:
        reward_function = reward_calculator.reward_function5
    else:
        raise ValueError("Invalid reward function index")

    all_losses = []  
    all_rewards = []
    step_rewards = []  # 모든 스텝별 리워드를 저장할 리스트

    for episode in range(episodes):
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        while rirsv_array[x, y] == 1 or channels_array[x, y] == 1 or wkmstrm_array[x, y] == 1:  # 시작점이 저수지 영역, 하천망, 강이면 다시 선택
            x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        
        reward_calculator.start_x, reward_calculator.start_y = x, y
        state = torch.tensor([x, y, reward_calculator.get_elevation(x, y), reward_calculator.calculate_slope(x, y),
                              rirsv_array[int(x), int(y)], wkmstrm_array[int(x), int(y)], climbpath_array[int(x), int(y)],
                              road_array[int(x), int(y)], watershed_basins_array[int(x), int(y)], channels_array[int(x), int(y)],
                              forestroad_array[int(x), int(y)], hiking_array[int(x), int(y)]], dtype=torch.float32).to(device)

        reward_calculator.state_buffer.clear()  # 에피소드 시작 시 버퍼 초기화
        done = False
        step = 0
        prev_path = [(x, y)]

        episode_losses = []  # 각 에피소드의 손실을 저장할 리스트
        episode_reward = 0  # 각 에피소드의 총 보상을 추적
        episode_step_rewards = []  # 각 에피소드의 스텝별 보상을 저장할 리스트

        a = 0
        while not done and step < max_steps:
            agent.update_speed(step)

            if np.random.uniform(0, 1) < epsilon or a == 1:
                action = np.random.randint(action_size)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()

            next_x, next_y = x, y

            # 행동에 따른 다음 위치 계산
            if action_mode == '8_directions':
                if action == 0:  # 상
                    next_x, next_y = (x - agent.speed, y)
                elif action == 1:  # 하
                    next_x, next_y = (x + agent.speed, y)
                elif action == 2:  # 좌
                    next_x, next_y = (x, y - agent.speed)
                elif action == 3:  # 우
                    next_x, next_y = (x, y + agent.speed)
                elif action == 4:  # 좌상
                    next_x, next_y = (x - agent.speed, y - agent.speed)
                elif action == 5:  # 우상
                    next_x, next_y = (x - agent.speed, y + agent.speed)
                elif action == 6:  # 좌하
                    next_x, next_y = (x + agent.speed, y - agent.speed)
                elif action == 7:  # 우하
                    next_x, next_y = (x + agent.speed, y + agent.speed)
            
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
                    for i in range(-int(agent.speed), int(agent.speed) + 1):
                        for j in range(-int(agent.speed), int(agent.speed) + 1):
                            if 0 <= x + i < dem_array.shape[0] and 0 <= y + j < dem_array.shape[1]:
                                elevation = reward_calculator.get_elevation(x + i, y + j)
                                if elevation > highest_elevation:
                                    highest_elevation = elevation
                                    highest_coord = (x + i, y + j)
                    next_x, next_y = highest_coord
                elif action == 5 and len(prev_path) > 1:  # 되돌아가기 (Backtracking, BT)
                    next_x, next_y = prev_path[-2]

            # 경계 조건 확인
            next_x = int(min(max(next_x, 0), dem_array.shape[0] - 1))
            next_y = int(min(max(next_y, 0), dem_array.shape[1] - 1))

            if rirsv_array[next_x, next_y] == 1 or channels_array[next_x, next_y] == 1 or wkmstrm_array[next_x, next_y] == 1:
                a = 1
                continue
            else:
                a = 0
                next_state = torch.tensor([next_x, next_y, reward_calculator.get_elevation(next_x, next_y), reward_calculator.calculate_slope(next_x, next_y),
                                           rirsv_array[next_x, next_y], wkmstrm_array[next_x, next_y], climbpath_array[next_x, next_y],
                                           road_array[next_x, next_y], watershed_basins_array[next_x, next_y], channels_array[next_x, next_y],
                                           forestroad_array[next_x, next_y], hiking_array[next_x, next_y]], dtype=torch.float32).to(device)

            reward = reward_function(next_state)
            episode_reward += reward
            episode_step_rewards.append(reward.item())

            replay_buffer.add(state, action, reward, next_state, done)

            # 단일 샘플 학습
            experience = replay_buffer.get_latest_experience()
            if experience:
                state_sample, action_sample, reward_sample, next_state_sample, done_sample = experience
                state_sample, action_sample, reward_sample, next_state_sample, done_sample = (
                    state_sample.to(device),
                    torch.tensor(action_sample).to(device),
                    torch.tensor(reward_sample).to(device),
                    next_state_sample.to(device),
                    torch.tensor(done_sample).to(device)
                )

                q_values = model(state_sample.unsqueeze(0))
                next_q_values = target_model(next_state_sample.unsqueeze(0)).detach()  # 타겟 네트워크 사용

                # 벨만 방정식을 이용한 타겟 계산
                targets = reward_sample + (1 - done_sample.float()) * gamma * torch.max(next_q_values, dim=1)[0]

                # 손실 계산
                q_values = q_values.gather(1, action_sample.unsqueeze(0).unsqueeze(0)).squeeze(1)
                loss = criterion(q_values, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_losses.append(loss.item())  # 손실 저장

            state = next_state
            x, y = next_x, next_y
            prev_path.append((x, y))

            step += 1

            if step % 10 == 0:
                with torch.no_grad():
                    expected_reward = torch.max(model(state.unsqueeze(0))).item()
                print(f"Episode {episode}, Step {step}, State ({x}, {y}), Expected Reward: {expected_reward:.2f}")

            if step >= max_steps:
                done = True

        all_losses = [loss.tolist() if isinstance(loss, torch.Tensor) else loss for loss in all_losses]
        all_rewards = [reward.tolist() if isinstance(reward, torch.Tensor) else reward for reward in all_rewards]
        step_rewards.append(episode_step_rewards)  # 에피소드의 스텝별 보상을 저장

        epsilon = max(epsilon_end, epsilon * epsilon_decay)  # 에피소드 후에 epsilon 감소

        # 일정 주기마다 타겟 네트워크 업데이트
        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

    save_model(model, filename=model_filename)
    save_to_json(all_losses, filename='losses.json')  # 손실을 JSON 파일로 저장
    save_to_json(all_rewards, filename='rewards.json')  # 보상을 JSON 파일로 저장
    save_to_json(step_rewards, filename='step_rewards.json')  # 스텝별 보상을 JSON 파일로 저장

    return model, all_losses, all_rewards

def simulate_path(start_x, start_y, model, dem_array, rirsv_array, wkmstrm_array, climbpath_array, road_array, watershed_basins_array, channels_array, forestroad_array, hiking_array, agent, action_mode='8_directions'):
    path = [(int(start_x), int(start_y))]
    x, y = int(start_x), int(start_y)

    step = 0

    visited_count = defaultdict(int)  # 방문한 좌표와 그 횟수를 저장할 딕셔너리

    while step < simulation_max_steps:
        agent.update_speed(step)  # 속도 업데이트 추가
        state = torch.tensor([x, y, get_elevation(dem_array, x, y), calculate_slope(dem_array, x, y),
                              rirsv_array[x, y], wkmstrm_array[x, y], climbpath_array[x, y],
                              road_array[x, y], watershed_basins_array[x, y], channels_array[x, y],
                              forestroad_array[x, y], hiking_array[x, y]], dtype=torch.float32)
        with torch.no_grad():
            action = torch.argmax(model(state)).item()

        # 기본 값으로 현재 위치를 설정
        next_x, next_y = x, y

        if action_mode == '8_directions':
            if action == 0:  # 상
                next_x, next_y = (x - agent.speed, y)
            elif action == 1:  # 하
                next_x, next_y = (x + agent.speed, y)
            elif action == 2:  # 좌
                next_x, next_y = (x, y - agent.speed)
            elif action == 3:  # 우
                next_x, next_y = (x, y + agent.speed)
            elif action == 4:  # 좌상
                next_x, next_y = (x - agent.speed, y - agent.speed)
            elif action == 5:  # 우상
                next_x, next_y = (x - agent.speed, y + agent.speed)
            elif action == 6:  # 좌하
                next_x, next_y = (x + agent.speed, y - agent.speed)
            elif action == 7:  # 우하
                next_x, next_y = (x + agent.speed, y + agent.speed)

        elif action_mode == 'custom':
            if action == 0:  # 무작위 걷기 (Random Walking, RW)
                next_x, next_y = (x + np.random.choice([-agent.speed, agent.speed]), y + np.random.choice([-agent.speed, agent.speed]))
            elif action == 1:  # 경로 여행 (Route Traveling, RT)
                if climbpath_array[int(x), int(y)]:
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
            elif action == 3:  # 시야 확보 (View Enhancing, VE)
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
            elif action == 4 and len(path) > 1:  # 되돌아가기 (Backtracking, BT)
                next_x, next_y = path[-2]

        # 경계 조건 확인
        next_x = int(min(max(next_x, 0), dem_array.shape[0] - 1))
        next_y = int(min(max(next_y, 0), dem_array.shape[1] - 1))

        if rirsv_array[next_x, next_y] == 1 or channels_array[next_x, next_y] == 1 or wkmstrm_array[next_x, next_y] == 1:
            continue  # 다른 방향으로 이동하도록 함

        path.append((next_x, next_y))

        visited_count[(next_x, next_y)] += 1  # 방문 횟수 업데이트

        x, y = next_x, next_y

        step += 1

    return path

def plot_loss_from_json(filename='losses.json'):
    all_losses = load_losses_from_json(filename)
    flattened_losses = [loss for episode_losses in all_losses for loss in episode_losses]
    plt.figure(figsize=(10, 5))
    plt.plot(flattened_losses, label='Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss over Steps')
    plt.legend()
    plt.show()
