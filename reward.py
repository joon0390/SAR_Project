import numpy as np
from collections import deque, defaultdict  # Add defaultdict to imports

class RewardCalculator:
    def __init__(self, dem_array, rirsv_array, wkmstrm_array, climbpath_array, watershed_basins_array, channels_array):
        self.dem_array = dem_array
        self.rirsv_array = rirsv_array
        self.wkmstrm_array = wkmstrm_array
        self.climbpath_array = climbpath_array
        self.watershed_basins_array = watershed_basins_array
        self.channels_array = channels_array
        self.current_watershed = None
        self.state_buffer = deque(maxlen=5)
        self.visited_count = defaultdict(int)
        self.start_x = 0
        self.start_y = 0

    def get_elevation(self, x, y):
        return self.dem_array[x, y]

    def calculate_slope(self, x, y):
        if x <= 0 or x >= self.dem_array.shape[0] - 1 or y <= 0 or y >= self.dem_array.shape[1] - 1:
            return 0
        dzdx = (self.dem_array[x + 1, y] - self.dem_array[x - 1, y]) / 2
        dzdy = (self.dem_array[x, y + 1] - self.dem_array[x, y - 1]) / 2
        slope = np.sqrt(dzdx**2 + dzdy**2)
        return slope

    def update_visited_count(self, x, y):
        self.visited_count[(x, y)] += 1

    def reward_function(self, state):
        x, y, elevation, slope, rirsv, wkmstrm, climbpath, watershed_basins, channels = state

        if rirsv:
            return -10000

        reward = -1

        if climbpath:
            reward += 50

        if rirsv:
            reward -= 10

        if slope > 0.5:
            reward -= 10
        else:
            reward += 10

        if wkmstrm:
            reward -= 50

        if channels:
            reward -= 5

        if watershed_basins:
            reward -= 5

        reward -= 0.1 * (abs(state[0] - self.start_x) + abs(state[1] - self.start_y))

        self.state_buffer.append(state)

        if len(self.state_buffer) > 1:
            for i in range(1, len(self.state_buffer)):
                prev_state = self.state_buffer[i - 1]
                curr_state = self.state_buffer[i]

                if prev_state[2] > curr_state[2]:
                    reward += 5

                if prev_state[6] and curr_state[6]:
                    reward += 10

                if not prev_state[7] and curr_state[7]:
                    reward -= 10

                if prev_state[3] < curr_state[3]:
                    reward -= 5
                elif prev_state[3] > curr_state[3]:
                    reward += 5

                if prev_state[5] and curr_state[5]:
                    reward -= 10

                if prev_state[4] and not curr_state[4]:
                    reward += 100
                if prev_state[5] and not curr_state[5]:
                    reward += 10
                if not prev_state[6] and curr_state[6]:
                    reward += 50
                if prev_state[8] and not curr_state[8]:
                    reward += 5

        visits = self.visited_count[(x, y)]
        reward -= visits * 5

        return reward
