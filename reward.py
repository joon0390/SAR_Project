import numpy as np
from collections import deque, defaultdict  # Add defaultdict to imports

class RewardCalculator:
    def __init__(self, dem_array, rirsv_array, wkmstrm_array, climbpath_array, road_array, watershed_basins_array, channels_array,forestroad_array,hiking_array):
        self.dem_array = dem_array
        self.rirsv_array = rirsv_array
        self.wkmstrm_array = wkmstrm_array
        self.climbpath_array = climbpath_array
        self.road_array = road_array
        self.watershed_basins_array = watershed_basins_array
        self.channels_array = channels_array
        self.forestroad_array = forestroad_array
        self.hiking_array = hiking_array
        self.current_watershed = None
        self.state_buffer = deque(maxlen=5)
        self.visited_count = defaultdict(int)
        self.start_x = 0
        self.start_y = 0

    def get_elevation(self, x, y):
        """현재 위치의 고도를 반환합니다."""
        return self.dem_array[x, y]

    def calculate_slope(self, x, y):
        """현재 위치의 경사를 계산합니다."""
        if x <= 0 or x >= self.dem_array.shape[0] - 1 or y <= 0 or y >= self.dem_array.shape[1] - 1:
            return 0
        dzdx = (self.dem_array[x + 1, y] - self.dem_array[x - 1, y]) / 2
        dzdy = (self.dem_array[x, y + 1] - self.dem_array[x, y - 1]) / 2
        slope = np.sqrt(dzdx**2 + dzdy**2)
        return slope

    def update_visited_count(self, x, y):
        """방문한 좌표의 방문 횟수를 업데이트합니다."""
        self.visited_count[(x, y)] += 1

    def reward_function(self, state):
        """
        주어진 상태에 대해 보상을 계산합니다.
        상태는 (x, y, 고도, 경사, 물저장소, 물길, 등산 경로, 도로, 유역, 채널)로 구성됩니다.
        """
        x, y, elevation, slope, rirsv, wkmstrm, climbpath, road, watershed_basins, channels,forestroad,hiking = state

        # 물저장소에 있는 경우 큰 벌칙을 부여합니다.
        if rirsv:
            return -10000

        reward = -1
    
        # 토지피복도에 있는 경우 보상을 부여합니다.
        if climbpath:
            reward += 50

        # 도로에 있는 경우 큰 보상을 부여합니다.
        if road:
            reward += 1000

        # 경사가 큰 경우 벌칙을 부여합니다.
        if slope > 0.5:
            reward -= 10
        else:
            reward += 10

        # 물길에 있는 경우 벌칙을 부여합니다.
        if wkmstrm:
            reward -= 50

        # 채널에 있는 경우 벌칙을 부여합니다.
        if channels:
            reward -= 5

        # 유역에 있는 경우 벌칙을 부여합니다.
        if watershed_basins:
            reward -= 5
        #도로에 있는 경우 큰 보상
        if forestroad : 
            reward += 1000
        #등산로에 있는 경우 큰 보상
        if hiking :
            reward += 20
        

        # 시작점에서 멀어질수록 벌칙을 부여합니다.
        # reward -= 0.1 * (abs(state[0] - self.start_x) + abs(state[1] - self.start_y))

        # 현재 상태를 state_buffer에 추가합니다.
        self.state_buffer.append(state)

        # state_buffer에 2개 이상의 상태가 있는 경우, 이전 상태와 현재 상태를 비교합니다.
        if len(self.state_buffer) > 1:
            for i in range(1, len(self.state_buffer)):
                prev_state = self.state_buffer[i - 1]  # 이전 상태
                curr_state = self.state_buffer[i]      # 현재 상태

                # 이전 상태와 비교하여 보상 및 벌칙을 부여합니다.
                if prev_state[2] > curr_state[2]:  # 이전 상태보다 고도가 낮아진 경우
                    reward += 5

                if prev_state[6] and curr_state[6]:  # 이전 상태와 현재 상태 모두 등산 경로인 경우
                    reward += 10

                if not prev_state[7] and curr_state[7]:  # 이전 상태는 도로가 없었지만 현재 상태는 도로인 경우
                    reward += 1000
                if prev_state[7] and not curr_state[7]:  # 이전 상태는 도로였지만 현재 상태는 도로가 아닌 경우
                    reward -= 1000

                if prev_state[3] < curr_state[3]:  # 이전 상태보다 경사가 커진 경우
                    reward -= 5
                elif prev_state[3] > curr_state[3]:  # 이전 상태보다 경사가 작아진 경우
                    reward += 5

                if prev_state[5] and curr_state[5]:  # 이전 상태와 현재 상태 모두 물길인 경우
                    reward -= 10

                if prev_state[4] and not curr_state[4]:  # 이전 상태는 물저장소였지만 현재 상태는 물저장소가 아닌 경우
                    reward += 100
                if prev_state[5] and not curr_state[5]:  # 이전 상태는 물길이었지만 현재 상태는 물길이 아닌 경우
                    reward += 10
                if not prev_state[6] and curr_state[6]:  # 이전 상태는 등산 경로가 아니었지만 현재 상태는 등산 경로인 경우
                    reward += 50
                if prev_state[8] and not curr_state[8]:  # 이전 상태는 유역이었지만 현재 상태는 유역이 아닌 경우
                    reward += 5

        # 방문한 횟수에 따라 벌칙을 부여합니다.
        visits = self.visited_count[(x, y)]
        reward -= visits * 5

        return reward
