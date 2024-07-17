import numpy as np

class RewardCalculator:
    def __init__(self, dem_array, rirsv_array, wkmstrm_array, road_array, watershed_basins_array, channels_array):
        self.dem_array = dem_array
        self.rirsv_array = rirsv_array
        self.wkmstrm_array = wkmstrm_array
        self.road_array = road_array
        self.watershed_basins_array = watershed_basins_array
        self.channels_array = channels_array

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
            return -10000
        
        reward = -1  # 기본 패널티

        # 도로에 도달하면 높은 보상
        if road:
            reward += 20
        # 강이나 경사가 큰 지역에 있으면 큰 패널티
        if rirsv:
            reward -= 10
        if slope > 0.5:
            reward -= 10
        # 작은 강(개천) 근처에 있으면 중간 패널티
        if wkmstrm:
            reward += 5
        # 워터셰드 채널에 있으면 보상
        if channels:
            reward -= 5

        # 이동 거리에 따른 보상/패널티 추가
        reward -= 0.1 * (abs(state[0] - self.start_x) + abs(state[1] - self.start_y))

        return reward
