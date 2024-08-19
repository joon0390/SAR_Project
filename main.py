import torch
import os
import numpy as np

from geo_processing import GISProcessor
from agent import Agent
from network import DQN
from replay_buffer import ReplayBuffer
from reward import RewardCalculator
from utils import *
from config import *

def save_model(model, filename='dqn_model.pth'):
    torch.save(model.state_dict(), filename)

def load_model(filename='dqn_model.pth', input_dim=11, output_dim=8):
    model = DQN(input_dim, output_dim)
    model.load_state_dict(torch.load(filename))
    return model

def dqn_learning(dem_array, rirsv_array, wkmstrm_array, climbpath_array, road_array, watershed_basins_array, channels_array, forestroad_array,
                reward_calculator, 
                agent=Agent(age_group="young",gender='male',health_status='good'), 
                action_mode='8_directions', load_existing=False, model_filename='dqn_model.pth',
                _lr=0.001, _gamma=0.99, batch_size=32 ,buffer_size=1000, max_steps=1000, 
                episodes=500, target_update_freq=20, reward_function_index=2):

    def optimize(mini_batch, model, target_model):
        for state, action, reward, next_state, done in mini_batch:
            state = state.to(device)
            next_state = next_state.to(device)
            reward = torch.tensor(reward).to(device)
            action = torch.tensor(action).to(device)
            done = torch.tensor(done).to(device)

            q_values = model(state)
            next_q_values = target_model(next_state).detach()

            targets = reward + (1 - done.float()) * gamma * torch.max(next_q_values, dim=0)[0]
            q_values = q_values.gather(0, action.unsqueeze(0)).squeeze(0)

            loss = criterion(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_losses.append(loss.item())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_size = 11
    if action_mode == '8_directions':
        action_size = 8
    elif action_mode == 'custom':
        action_size = 5

    model = DQN(state_size, action_size).to(device)
    target_model = DQN(state_size, action_size).to(device)
    if load_existing and os.path.exists(model_filename):
        model.load_state_dict(torch.load(model_filename))
        target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
    criterion = torch.nn.HuberLoss()
    gamma = _gamma

    replay_buffer = ReplayBuffer(buffer_size)

    agent.speed = int(agent.speed)

    all_losses = []
    all_rewards = []
    
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

    for episode in range(episodes):
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        while rirsv_array[x, y] == 1 or channels_array[x, y] == 1 or wkmstrm_array[x, y] == 1:  # 시작점이 저수지 영역, 하천망, 강이면 다시 선택
            x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
    
        reward_calculator.start_x, reward_calculator.start_y = x, y
        state = torch.tensor([x, y, reward_calculator.get_elevation(x, y), reward_calculator.calculate_slope(x, y),
                            rirsv_array[int(x), int(y)], wkmstrm_array[int(x), int(y)], climbpath_array[int(x), int(y)],
                            road_array[int(x), int(y)], watershed_basins_array[int(x), int(y)], channels_array[int(x), int(y)],
                            forestroad_array[int(x), int(y)]], dtype=torch.float32).to(device)
        reward_calculator.state_buffer.clear()  # 에피소드 시작 시 버퍼 초기화

        done = False
        episode_losses = []
        episode_reward = 0
        step = 0
        prev_path = [(x, y)]
        obstacle_encountered = False #물에 닿는 것에 대한 변수

        while not done and step < max_steps:
            agent.update_speed(step)

            if np.random.uniform(0, 1) < agent.explore_ratio or obstacle_encountered:
                action = np.random.randint(action_size)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()

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
                obstacle_encountered = True
                continue
            else:
                obstacle_encountered = False
                next_state = torch.tensor([next_x, next_y, reward_calculator.get_elevation(next_x, next_y), reward_calculator.calculate_slope(next_x, next_y),
                                           rirsv_array[next_x, next_y], wkmstrm_array[next_x, next_y], climbpath_array[next_x, next_y],
                                           road_array[next_x, next_y], watershed_basins_array[next_x, next_y], channels_array[next_x, next_y],
                                           forestroad_array[next_x, next_y]], dtype=torch.float32).to(device)
            reward = reward_function(next_state)

            print(f"Calculated Reward: {reward}, Position: ({x}, {y}) -> ({next_x}, {next_y})")

            if step % 10 == 0:  # 10 스텝마다 출력
                print(f"Episode {episode}, Step {step}, Reward: {reward:.2f}, Position: ({x}, {y}) -> ({next_x}, {next_y})")
            
            episode_reward += reward

            replay_buffer.add(state, action, reward, next_state, done)

            if len(replay_buffer) >= batch_size:
                mini_batch = replay_buffer.sample(batch_size)
                optimize(mini_batch, model, target_model)  

            state = next_state
            x, y = next_x, next_y
            prev_path.append((x, y))
            step += 1


        all_losses.append(episode_losses)
        all_rewards.append(episode_reward)

        if episode % target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())

        save_model(model, filename=model_filename)
        save_to_json(all_losses, filename='losses.json')  
        save_to_json(all_rewards, filename='rewards.json')  

    return model, all_losses, all_rewards


def train(filename, dem_file, test_area_npy, train_iter, buffer_size, max_steps, episodes, target_update_freq):
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
    rirsv_array = combined_array[:, :, 1]
    wkmstrm_array = combined_array[:, :, 2]
    climbpath_array = combined_array[:, :, 3]
    road_array = combined_array[:, :, 4]
    watershed_basins_array = combined_array[:, :, 5]
    channels_array = combined_array[:, :, 6]
    forestroad_array = combined_array[:, :, 7]

    # GISProcessor 인스턴스를 통해 DEM 변환 정보 가져오기
    gis_processor = GISProcessor(dem_file, featured_npy)
    dem_transform = gis_processor.dem_transform

    # 보상 계산기 인스턴스 생성
    reward_calculator = RewardCalculator(dem_array, rirsv_array, wkmstrm_array, forestroad_array, road_array, watershed_basins_array, channels_array, climbpath_array, dem_transform)
    
    action_mode = '8_directions'
    agent = Agent(age_group='young', gender='male', health_status='good')
    _path = []
    paths = []
    start_points = []
    index = 40
    lr_array = np.array([0.0001])
    gamma_array = np.array([0.99])
    decay_factor = 0.99
    reward_function_index = 2

    # lr과 gamma를 임의로 선택
    lr = np.random.choice(lr_array)
    gamma = np.random.choice(gamma_array)
    print(f"Selected parameters - lr: {lr}, gamma: {gamma}")

    # train_iter 동안 동일한 lr과 gamma로 반복
    for i in range(train_iter):
        print(f"===== Iteration {i + 1}/{train_iter} =====")
        epsilon = agent.explore_ratio
        print('epsilon :', epsilon, "lr :", lr, "gamma:", gamma)

        # 동일한 시작점 선택
        test_area = np.load(test_area_npy)
        coord = get_random_index(test_area)
        start_x, start_y = coord[0], coord[1]
        start_points.append((start_x, start_y))
        print("State Point : ({0},{1})".format(start_x,start_y))
        # 에이전트 속도 감소 적용
        agent.update_speed(decay_factor=decay_factor)
        
        print("Before dqn_learning call")

        model, all_losses, all_rewards = dqn_learning(
            dem_array, rirsv_array, wkmstrm_array, climbpath_array, road_array,
            watershed_basins_array, channels_array, forestroad_array,
            reward_calculator, agent, action_mode=action_mode, load_existing=False, model_filename='dqn_model.pth',
            _lr=lr, _gamma=gamma, buffer_size=buffer_size, max_steps=max_steps, episodes=episodes,
            target_update_freq=target_update_freq, reward_function_index=reward_function_index)
        
        print("After dqn_learning call")

        model = load_model('dqn_model.pth', input_dim=11, output_dim=8 if action_mode == '8_directions' else 5)
        path = simulate_path(start_x, start_y, model, dem_array, rirsv_array, wkmstrm_array, climbpath_array, road_array, watershed_basins_array, channels_array, forestroad_array, agent, action_mode=action_mode)
        paths.append(path)
        _path.append(path[index - 1])  # path에서 index번째 추가
        print("---------------", i + 1, "번째 model의 path", "-------------------------")

        state = torch.tensor([start_x, start_y, reward_calculator.get_elevation(start_x, start_y), reward_calculator.calculate_slope(start_x, start_y),
                          rirsv_array[start_x, start_y], wkmstrm_array[start_x, start_y], climbpath_array[start_x, start_y],
                          road_array[start_x, start_y], watershed_basins_array[start_x, start_y], channels_array[start_x, start_y],
                          forestroad_array[start_x, start_y]], dtype=torch.float32)

        for episode, reward in enumerate(all_rewards):
            for step in range(max_steps):
                with torch.no_grad():
                    expected_reward = torch.max(model(state.unsqueeze(0))).item()
                print(f"Episode {episode + 1}, Step {step + 1}, State ({start_x}, {start_y}), Expected Reward: {expected_reward:.2f}")
    _path = np.array(_path)
    
    # 시각화 함수 호출
    visualize_paths(dem_array, _path, dem_transform, index, start_points)

    # 손실 및 보상 시각화 함수 호출
    plot_losses_and_rewards(all_losses, all_rewards)

def simulate_path(start_x, start_y, model, dem_array, rirsv_array, wkmstrm_array, climbpath_array, road_array, watershed_basins_array, channels_array, forestroad_array, agent, action_mode='8_directions'):
    path = [(int(start_x), int(start_y))]
    x, y = int(start_x), int(start_y)

    step = 0

    while step < simulation_max_steps:
        agent.update_speed(step)  # 속도 업데이트 추가
        state = torch.tensor([x, y, get_elevation(dem_array, x, y), calculate_slope(dem_array, x, y),
                              rirsv_array[x, y], wkmstrm_array[x, y], climbpath_array[x, y],
                              road_array[x, y], watershed_basins_array[x, y], channels_array[x, y],
                              forestroad_array[x, y]], dtype=torch.float32)
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

        x, y = next_x, next_y

        step += 1

    return path

if __name__ == "__main__":
    filename = '/Users/heekim/Desktop/heekimjun/SAR_Project_Agent/final_rl/new_dem.npy'
    dem_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/dem/dem.tif'  
    test_area_npy = '/Users/heekim/Desktop/heekimjun/SAR_Project_Agent/final_rl/test_area_result.npy'  
    train_iter = train_iter  # 훈련 반복 횟수
    buffer_size = buffer_size
    max_steps = max_steps
    episodes = episodes
    target_update_freq = target_update_freq

    train(filename, dem_file, test_area_npy, train_iter, buffer_size, max_steps, episodes, target_update_freq)
