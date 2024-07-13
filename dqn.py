import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from utils import get_elevation, calculate_slope
from collections import deque
from config import *

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

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = alpha
        self.target_update = target_update

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if random.random() < 0.01:
            print(f"Loss: {loss.item()}")

def run_dqn(dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array, reward_calculator, episodes):
    state_dim = 6
    action_dim = 6
    agent = DQNAgent(state_dim, action_dim)

    for episode in range(episodes):
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        reward_calculator.start_x, reward_calculator.start_y = x, y
        state = (x, y, get_elevation(x, y, dem_array), calculate_slope(x, y, dem_array),
                 rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y],
                 watershed_basins_array[x, y], channels_array[x, y])
        done = False
        step = 0
        prev_path = [(x, y)]

        while not done and step < 1000:
            action = agent.select_action(state)
            next_x, next_y = x, y
            if action == 0:
                next_x, next_y = (x + np.random.choice([-10, 10]), y + np.random.choice([-10, 10]))
            elif action == 1:
                if road_array[x, y]:
                    next_x, next_y = (x + np.random.choice([-10, 10]), y)
                else:
                    next_x, next_y = (x, y + np.random.choice([-10, 10]))
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
            elif action == 3:
                next_x, next_y = x, y
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

            next_x = min(max(next_x, 0), dem_array.shape[0] - 1)
            next_y = min(max(next_y, 0), dem_array.shape[1] - 1)

            next_state = (next_x, next_y, get_elevation(next_x, next_y, dem_array), calculate_slope(next_x, next_y, dem_array),
                          rirsv_array[next_x, next_y], wkmstrm_array[next_x, next_y], road_array[next_x, next_y],
                          watershed_basins_array[next_x, next_y], channels_array[next_x, next_y])

            reward = reward_calculator.reward_function(next_state)
            done = road_array[next_x, next_y]

            agent.store_transition(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            x, y = next_x, next_y
            step += 1

        if episode % agent.target_update == 0:
            agent.update_target_net()

        print(f"Episode {episode} completed")

    torch.save(agent.policy_net.state_dict(), 'dqn_model.pth')
    return agent

def simulate_path(start_x, start_y, agent, dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array):
    path = [(start_x, start_y)]
    x, y = start_x, start_y
    max_steps = 1000

    for step in range(max_steps):
        state = (x, y, get_elevation(x, y, dem_array), calculate_slope(x, y, dem_array),
                 rirsv_array[x, y], wkmstrm_array[x, y], road_array[x, y],
                 watershed_basins_array[x, y], channels_array[x, y])
        action = agent.select_action(state)

        next_x, next_y = x, y
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