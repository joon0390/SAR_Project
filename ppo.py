import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

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
            self.stay_put_probability = 0.3

class PPO(nn.Module):
    def __init__(self, input_dim=9, action_dim=8, hidden_dim=64):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.actor(x), self.critic(x)
    
    def act(self, state):
        actor_output, _ = self.forward(state)
        probs = torch.softmax(actor_output, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, state, action):
        actor_output, critic_output = self.forward(state)
        probs = torch.softmax(actor_output, dim=-1)
        dist = Categorical(probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_log_probs, torch.squeeze(critic_output), dist_entropy

class PPOAgent:
    def __init__(self, input_dim=9, action_dim=8, hidden_dim=64, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.policy = PPO(input_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPO(input_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

    def update(self, memory):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluations
            logprobs, state_values, dist_entropy = self.policy.evaluate_actions(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ppo_learning(dem_array, rirsv_array, wkmstrm_array, climbpath_array, watershed_basins_array, channels_array, reward_calculator, agent, action_mode='8_directions', load_existing=False, model_filename='ppo_model.pth'):
    global epsilon

    state_size = 9
    if action_mode == '8_directions':
        action_size = 8
    elif action_mode == 'custom':
        action_size = 6
    
    ppo_agent = PPOAgent(input_dim=state_size, action_dim=action_size)
    memory = Memory()
    
    if load_existing and os.path.exists(model_filename):
        ppo_agent.policy.load_state_dict(torch.load(model_filename))
        print(f"Loaded existing model from {model_filename}")

    for episode in range(episodes):
        x, y = np.random.randint(1, dem_array.shape[0] - 1), np.random.randint(1, dem_array.shape[1] - 1)
        reward_calculator.start_x, reward_calculator.start_y = x, y
        state = torch.tensor([x, y, get_elevation(dem_array, x, y), calculate_slope(dem_array, x, y),
                              rirsv_array[x, y], wkmstrm_array[x, y], climbpath_array[x, y],
                              watershed_basins_array[x, y], channels_array[x, y]], dtype=torch.float32).to(device)
        reward_calculator.state_buffer.clear()
        reward_calculator.visited_count.clear()
        done = False
        step = 0
        prev_path = [(x, y)]

        while not done and step < max_steps:
            action, log_prob, _ = ppo_agent.policy_old.act(state)
            memory.states.append(state)
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(log_prob)

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
                if action == 0:
                    next_x, next_y = (x + np.random.choice([-agent.speed, agent.speed]), y + np.random.choice([-agent.speed, agent.speed]))
                elif action == 1:
                    if climbpath_array[x, y]:
                        next_x, next_y = (x + np.random.choice([-agent.speed, agent.speed]), y)
                    else:
                        next_x, next_y = (x, y + np.random.choice([-agent.speed, agent.speed]))
                elif action == 2:
                    direction = np.random.choice(['up', 'down', 'left', 'right'])
                    if direction == 'up':
                        next_x, next_y = (max(1, x - agent.speed), y)
                    elif direction == 'down':
                        next_x, next_y = (min(dem_array.shape[0] - 2, x + agent.speed), y)
                    elif direction == 'left':
                        next_x, next_y = (x, max(1, y - agent.speed))
                    elif direction == 'right':
                        next_x, next_y = (x, min(dem_array.shape[1] - 2, y + agent.speed))
                elif action == 3:
                    next_x, next_y = x, y
                elif action == 4:
                    highest_elevation = get_elevation(dem_array, x, y)
                    highest_coord = (x, y)
                    for i in range(-agent.speed, agent.speed + 1):
                        for j in range(-agent.speed, agent.speed + 1):
                            if 0 <= x + i < dem_array.shape[0] and 0 <= y + j < dem_array.shape[1]:
                                elevation = get_elevation(dem_array, x + i, y + j)
                                if elevation > highest_elevation:
                                    highest_elevation = elevation
                                    highest_coord = (x + i, y + j)
                    next_x, next_y = highest_coord
                elif action == 5 and len(prev_path) > 1:
                    next_x, next_y = prev_path[-2]

            next_x = min(max(next_x, 0), dem_array.shape[0] - 1)
            next_y = min(max(next_y, 0), dem_array.shape[1] - 1)

            next_state = torch.tensor([next_x, next_y, get_elevation(dem_array, next_x, next_y), calculate_slope(dem_array, next_x, next_y),
                                       rirsv_array[next_x, next_y], wkmstrm_array[next_x, next_y], climbpath_array[next_x, next_y],
                                       watershed_basins_array[next_x, next_y], channels_array[next_x, next_y]], dtype=torch.float32).to(device)

            reward = reward_calculator.reward_function(next_state)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            state = next_state
            x, y = next_x, next_y
            prev_path.append((x, y))

            reward_calculator.update_visited_count(x, y)

            if step % 10 == 0:
                with torch.no_grad():
                    expected_reward = torch.max(ppo_agent.policy(state)[0]).item()
                print(f"Episode {episode}, Step {step}, State ({x}, {y}), Expected Reward: {expected_reward:.2f}")

            step += 1

            if climbpath_array[x, y] or step >= max_steps:
                done = True

        ppo_agent.update(memory)
        memory.clear_memory()

    torch.save(ppo_agent.policy.state_dict(), model_filename)
    return ppo_agent

def simulate_path(start_x, start_y, model, dem_array, rirsv_array, wkmstrm_array, climbpath_array, watershed_basins_array, channels_array, agent, action_mode='8_directions'):
    path = [(start_x, start_y)]
    x, y = start_x, start_y
    max_steps = simulation_max_steps

    visited_count = defaultdict(int)

    for step in range(max_steps):
        state = torch.tensor([x, y, get_elevation(dem_array, x, y), calculate_slope(dem_array, x, y),
                              rirsv_array[x, y], wkmstrm_array[x, y], climbpath_array[x, y],
                              watershed_basins_array[x, y], channels_array[x, y]], dtype=torch.float32).to(device)
        with torch.no_grad():
            action = torch.argmax(model.policy(state)[0]).item()

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
            if action == 0:
                next_x, next_y = (x + np.random.choice([-1, 1]), y + np.random.choice([-1, 1]))
            elif action == 1:
                if climbpath_array[x, y]:
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
                    next_x, next_y = (x, min(dem_array.shape[1] - 2, y + 1))
            elif action == 3:
                next_x, next_y = x, y
            elif action == 4:
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
            elif action == 5 and len(path) > 1:
                next_x, next_y = path[-2]

        next_x = min(max(next_x, 0), dem_array.shape[0] - 1)
        next_y = min(max(next_y, 0), dem_array.shape[1] - 1)

        path.append((next_x, next_y))

        visited_count[(next_x, next_y)] += 1

        x, y = next_x, next_y

        if climbpath_array[x, y]:
            break

    return path

if __name__ == "__main__":
    main()
