from collections import deque
import random
import torch

class ReplayBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=64):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)
