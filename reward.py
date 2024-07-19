import numpy as np
from collections import deque, defaultdict

class RewardCalculator:
    def __init__(self, dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array):
        self.dem_array = dem_array
        self.rirsv_array = rirsv_array
        self.wkmstrm_array = wkmstrm_array
        self.road_array = road_array
        self.watershed_basins_array = watershed_basins_array
        self.channels_array = channels_array
        self.current_watershed = None
        self.state_buffer = deque(maxlen=5)  # 최근 5개의 state를 저장할 버퍼
        self.visited_count = defaultdict(int)  # 방문 횟수를 저장할 딕셔너리
        self.start_x = 0
        self.start_y = 0

    def get_elevation(self, x, y):
        # 주어진 좌표의 고도 값 반환
        return self.dem_array[x, y]

    def calculate_slope(self, x, y):
        # 주어진 좌표의 경사 값 계산
        if x <= 0 or x >= self.dem_array.shape[0] - 1 or y <= 0 or y >= self.dem_array.shape[1] - 1:
            return 0
        dzdx = (self.dem_array[x + 1, y] - self.dem_array[x - 1, y]) / 2
        dzdy = (self.dem_array[x, y + 1] - self.dem_array[x, y - 1]) / 2
        slope = np.sqrt(dzdx**2 + dzdy**2)
        return slope

    def reward_function(self, state):
        '''
        보상 함수: 보상 및 패널티를 구체화
        '''
        x, y, elevation, slope, rirsv, wkmstrm, road, watershed_basins, channels = state

        if rirsv:
            return -1000

        reward = -1  # 기본 패널티

        # 탐험 보상
        if self.visited_count[(x, y)] == 0:
            reward += 10

        # 도로에 도달하면 높은 보상
        if road:
            reward += 50

        # 강이나 경사가 큰 지역에 있으면 큰 패널티
        if rirsv:
            reward -= 10

        if slope > 0.5:
            reward -= 10

        if slope <= 0.5:
            reward += 10

        # 작은 강(개천) 근처에 있으면 중간 패널티
        if wkmstrm:
            reward -= 50

        # 워터셰드 채널에 있으면 보상
        if channels:
            reward -= 5

        # 워터셰드 경계에 있으면 패널티
        if watershed_basins:
            reward -= 5

        # 방문 횟수에 따른 패널티 추가
        reward -= self.visited_count[(x, y)] * 10

        # 이동 거리에 따른 보상/패널티 추가
        reward -= 0.1 * (abs(state[0] - self.start_x) + abs(state[1] - self.start_y))
        
        # 버퍼에 현재 상태 추가
        self.state_buffer.append(state)

        # 버퍼의 이전 상태와 비교하여 보상/패널티 추가
        if len(self.state_buffer) > 1:
            for i in range(1, len(self.state_buffer)):
                prev_state = self.state_buffer[i - 1]
                curr_state = self.state_buffer[i]
                
                # 1. 연속적으로 고도가 낮아진 경우 보상
                if prev_state[2] > curr_state[2]:
                    reward += 5
                
                # 2. 연속적으로 road를 따라간 경우 보상
                if prev_state[6] and curr_state[6]:
                    reward += 50
                
                # 3. 이전 timestep과 달리 watershed의 경계에 도달한 경우 패널티
                if not prev_state[7] and curr_state[7]:
                    reward -= 20
                
                # 4. 경사가 연속적으로 커질때 패널티
                if prev_state[3] < curr_state[3]:
                    reward -= 5
                
                # 5. 경사가 연속적으로 완만해질 때 보상
                if prev_state[3] > curr_state[3]:
                    reward += 5
                
                # 6. 연속적으로 wkmstrm에 머무를때 패널티
                if prev_state[5] and curr_state[5]:
                    reward -= 10
                
                # 특정 상태에서 벗어나는 경우 추가 보상
                if prev_state[4] and not curr_state[4]:  # rirsv에서 벗어난 경우
                    reward += 100
                if prev_state[5] and not curr_state[5]:  # wkmstrm에서 벗어난 경우
                    reward += 10
                if not prev_state[6] and curr_state[6]:  # road에 도달한 경우
                    reward += 50
                if prev_state[8] and not curr_state[8]:  # channels에서 벗어난 경우
                    reward += 5

        # 방문한 장소에 대한 패널티 증가
        if self.visited_count[(x, y)] > 1:
            reward -= 20 * self.visited_count[(x, y)]  # 방문 횟수에 따라 패널티 증가

        return reward

    def update_visited_count(self, x, y):
        self.visited_count[(x, y)] += 1
