import torch
from scipy.stats import truncnorm
import numpy as np

# class Agent:
#     def __init__(self, age_group='young', gender='male', health_status='good'):
#         self.age_group = age_group
#         self.gender = gender
#         self.health_status = health_status
#         self.set_speed_and_explore_ratio()
        
#     def set_speed_and_explore_ratio(self):
#         if self.age_group == 'young':
#             self.speed = 2
#             self.explore_ratio = 0.8
#         elif self.age_group == 'middle':
#             self.speed = 2
#             self.explore_ratio = 0.7
#         elif self.age_group == 'old':
#             self.speed = 1
#             self.explore_ratio = 0.2

#         self.stay_put_probability = 0.0
#         self.speed_mean = self.speed
#         self.speed_std = 0.2

#     def sample_speed(self):
#         a, b = (1 - self.speed_mean) / self.speed_std, (3 - self.speed_mean) / self.speed_std
#         self.speed = truncnorm(a, b, loc=self.speed_mean, scale=self.speed_std).rvs()
    
#     def update_speed(self, decay_factor=0.99):
#         self.sample_speed()
#         self.speed *= decay_factor
#         self.speed = min(self.speed, 3)

class Agent:
    def __init__(self, age_group='young', gender='male', health_status='good'):
        self.age_group = age_group
        self.gender = gender
        self.health_status = health_status
        self.set_speed_and_explore_ratio()
        
    def set_speed_and_explore_ratio(self):
        if self.age_group == 'young':
            self.explore_ratio = 0.8
            self.speed_probabilities = [0.2, 0.5, 0.3]  
        elif self.age_group == 'middle':
            self.explore_ratio = 0.7
            self.speed_probabilities = [0.4, 0.4, 0.2]
        elif self.age_group == 'old':
            self.explore_ratio = 0.2
            self.speed_probabilities = [0.7, 0.2, 0.1]

        self.speeds = [1, 2, 3] 

    def sample_speed(self):
        self.speed = np.random.choice(self.speeds, p=self.speed_probabilities)
    
    def update_speed(self, decay_factor=0.99):
        self.sample_speed()
        # 3이 될 확률을 점진적으로 줄일 수 있음.
