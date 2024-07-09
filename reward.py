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
        보상함수에 대한 구체적인 명시가 필요
        '''
        x, y, elevation, slope, rirsv, wkmstrm, road, watershed_basins, channels = state
        
        reward = -1  # 기본 패널티

        # 도로에 도달하면 높은 보상
        if road:
            reward += 20
        # 강이나 경사가 큰 지역에 있으면
        if rirsv:
            reward -= 10
        if slope > 0.5:
            reward -= 10
        # 작은 강(개천) 근처에 있으면
        if wkmstrm:
            reward += 5
        # 워터셰드 채널에 있으면
        if channels:
            reward -= 5

        return reward

def discretize_state(state, q_mean):
    # 상태를 디스크리트 상태로 변환
    x, y = state[:2]
    max_x, max_y = q_mean.shape[0] - 1, q_mean.shape[1] - 1
    return min(x // 10, max_x), min(y // 10, max_y)
