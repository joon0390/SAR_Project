import numpy as np
from collections import deque  # Add defaultdict to imports

class RewardCalculator:
    def __init__(self, dem_array, rirsv_array, wkmstrm_array, forestroad_array, road_array, watershed_basins_array, channels_array, climbpath_array):
        self.dem_array = dem_array
        self.rirsv_array = rirsv_array
        self.wkmstrm_array = wkmstrm_array
        self.forestroad_array = forestroad_array
        self.road_array = road_array
        self.watershed_basins_array = watershed_basins_array
        self.channels_array = channels_array
        self.climbpath_array = climbpath_array
        self.current_watershed = None
        self.state_buffer = deque(maxlen=5)
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


    def reward_function1(self, state):
        """
        일반적인 경우의 보상
        """
        x, y, elevation, slope, rirsv, wkmstrm, forestroad, road, watershed_basins, channels, climbpath = state

        reward = 0
     
        # 등산 경로에 있는 경우 보상을 부여합니다.
        if climbpath:
            reward += 50

        # 도로에 있는 경우 큰 보상을 부여합니다.
        if road:
            reward += 100

        # 경사가 큰 경우 벌칙을 부여합니다.
        if slope > 0.5:
            reward -= 10
        else:
            reward += 10

   
        # 유역에 있는 경우 벌칙을 부여합니다.
        if watershed_basins:
            reward -= 5

        # 도로에 있는 경우 큰 보상
        if forestroad:
            reward += 1000



        # 시작점에서 멀어질수록 보상을 부여합니다.
        # reward += 0.1 * (abs(state[0] - self.start_x) + abs(state[1] - self.start_y))

        # 현재 상태를 state_buffer에 추가합니다.
        self.state_buffer.append(state)

        # state_buffer에 2개 이상의 상태가 있는 경우, 이전 상태와 현재 상태를 비교합니다.
        if len(self.state_buffer) > 1:
            for i in range(1, len(self.state_buffer)):
                prev_state = self.state_buffer[i - 1]  # 이전 상태
                curr_state = self.state_buffer[i]      # 현재 상태

                # 이전 상태와 비교하여 보상 및 벌칙을 부여합니다.


                if prev_state[-1] and curr_state[-1]:  # 이전 상태와 현재 상태 모두 등산 경로인 경우
                    reward += 10

                if not prev_state[7] and curr_state[7]:  # 이전 상태는 도로가 없었지만 현재 상태는 도로인 경우
                    reward += 1000
                if prev_state[7] and not curr_state[7]:  # 이전 상태는 도로였지만 현재 상태는 도로가 아닌 경우
                    reward -= 1000

                if prev_state[3] < curr_state[3]:  # 이전 상태보다 경사가 커진 경우
                    reward -= 5
                elif prev_state[3] > curr_state[3]:  # 이전 상태보다 경사가 작아진 경우
                    reward += 5
              
                if not prev_state[6] and curr_state[6]:  # 이전 상태는 등산 경로가 아니었지만 현재 상태는 등산 경로인 경우
                    reward += 50
                if prev_state[8] and not curr_state[8]:  # 이전 상태는 유역이었지만 현재 상태는 유역이 아닌 경우
                    reward += 5

       

        return reward



    def reward_function2(self, state):
        """
        주어진 상태에 대해 보상을 계산합니다. 이 함수는 등산 경로와 도로를 강하게 선호합니다.
        """
        x, y, elevation, slope, rirsv, wkmstrm, forestroad, road, watershed_basins, channels, climbpath = state

        reward = -1

        if climbpath: #등산로
            reward += 10
        if road: #모든 유형의 도로 multi_polygon
            reward += 20

        # if slope > 0.5:
        #     reward -= 150
        # else:
        #     reward += 1500
            
        if forestroad: #임도
            reward += 20

       

        # reward += 0.15 * (abs(state[0] - self.start_x) + abs(state[1] - self.start_y))

        self.state_buffer.append(state)

        if len(self.state_buffer) > 1:
            for i in range(1, len(self.state_buffer)):
                prev_state = self.state_buffer[i - 1]
                curr_state = self.state_buffer[i]

                # if prev_state[2] > curr_state[2]:
                #     reward += 800
        #6,7,10 
                if prev_state[6] and curr_state[6]: #임도 o -> 임도 o 
                    reward += 3
                if prev_state[7] and curr_state[7]: #도로 o -> 도로 o 
                    reward += 3
                if prev_state[10] and curr_state[10]: #등산로 o -> 등산로 o 
                    reward += 3
                

                if not prev_state[6] and curr_state[6]: #임도 x -> 임도o
                    reward += 10
                if not prev_state[7] and curr_state[7]: #도로 x -> 도로o
                    reward += 10
                if not prev_state[10] and curr_state[10]: #등산로 x -> 등산로o
                    reward += 10

                

                if prev_state[6] and not curr_state[6]: # 임도o -> 임도x
                    reward -= 15
                if prev_state[7] and not curr_state[7]: # 로드o -> 로드x
                    reward -= 15
                if prev_state[10] and not curr_state[10]: # 등산로o -> 등산로x
                    reward -= 15
               

                if prev_state[10] and curr_state[10] :
                    if prev_state[2] > prev_state[2] :
                        reward += 4
                    else :
                        reward -= 2

                # if prev_state[3] < curr_state[3]: # 경사 높아진 경우
                #     reward -= 5
                # elif prev_state[3] > curr_state[3]: #경사 낮아진 경우
                #     reward += 5


        

        return reward
    
    def reward_function3(self, state):
        """
        주어진 상태에 대해 보상을 계산합니다. 이 함수는 고도가 낮고 경사가 적은 경로를 선호합니다.
        """
        x, y, elevation, slope, rirsv, wkmstrm, forestroad, road, watershed_basins, channels, climbpath = state

        reward = 0

        if slope > 0.3:
            reward -= 20
        else:
            reward += 20



        if forestroad:
            reward += 200
        
        if climbpath:
            reward += 70

        if road:
            reward += 200

 
        # reward += 0.05 * (abs(state[0] - self.start_x) + abs(state[1] - self.start_y))

        self.state_buffer.append(state)

        if len(self.state_buffer) > 1:
            for i in range(1, len(self.state_buffer)):
                prev_state = self.state_buffer[i - 1]
                curr_state = self.state_buffer[i]

                if prev_state[2] > curr_state[2]: #고도가 낮아진 경우 보상
                    reward += 2





        if len(self.state_buffer) > 1:
            for i in range(1, len(self.state_buffer)):
                prev_state = self.state_buffer[i - 1]
                curr_state = self.state_buffer[i]

                # if prev_state[2] > curr_state[2]:
                #     reward += 800
        #6,7,10 
                if prev_state[6] and curr_state[6]: #임도 o -> 임도 o 
                    reward += 3
                if prev_state[7] and curr_state[7]: #도로 o -> 도로 o 
                    reward += 3
                if prev_state[10] and curr_state[10]: #등산로 o -> 등산로 o 
                    reward += 3
                

                if not prev_state[6] and curr_state[6]: #임도 x -> 임도o
                    reward += 10
                if not prev_state[7] and curr_state[7]: #도로 x -> 도로o
                    reward += 10
                if not prev_state[10] and curr_state[10]: #등산로 x -> 등산로o
                    reward += 10

                

                if prev_state[6] and not curr_state[6]: # 임도o -> 임도x
                    reward -= 15
                if prev_state[7] and not curr_state[7]: # 로드o -> 로드x
                    reward -= 15
                if prev_state[10] and not curr_state[10]: # 등산로o -> 등산로x
                    reward -= 15
               

                if prev_state[10] and curr_state[10] :
                    if prev_state[2] > prev_state[2] :
                        reward += 4
                    else :
                        reward -= 2


        return reward

    def reward_function4(self, state):
        """
        주어진 상태에 대해 보상을 계산합니다. 이 함수는 고도가 높고 경사가 적은 경로를 선호합니다.
        """
        x, y, elevation, slope, rirsv, wkmstrm, forestroad, road, watershed_basins, channels, climbpath = state

        reward = 0

        if slope > 0.3:
            reward -= 20
        else:
            reward += 20



        if forestroad:
            reward += 200
        
        if climbpath:
            reward += 70

        if road:
            reward += 200


        if watershed_basins :
            reward += 30


        self.state_buffer.append(state)

        if len(self.state_buffer) > 1:
            for i in range(1, len(self.state_buffer)):
                prev_state = self.state_buffer[i - 1]
                curr_state = self.state_buffer[i]

                if prev_state[2] < curr_state[2]: #고도가 높아진 경우 보상
                    reward += 20

        if len(self.state_buffer) > 1:
            for i in range(1, len(self.state_buffer)):
                prev_state = self.state_buffer[i - 1]
                curr_state = self.state_buffer[i]

                # if prev_state[2] > curr_state[2]:
                #     reward += 800
        #6,7,10 
                if prev_state[6] and curr_state[6]: #임도 o -> 임도 o 
                    reward += 3
                if prev_state[7] and curr_state[7]: #도로 o -> 도로 o 
                    reward += 3
                if prev_state[10] and curr_state[10]: #등산로 o -> 등산로 o 
                    reward += 3
                

                if not prev_state[6] and curr_state[6]: #임도 x -> 임도o
                    reward += 10
                if not prev_state[7] and curr_state[7]: #도로 x -> 도로o
                    reward += 10
                if not prev_state[10] and curr_state[10]: #등산로 x -> 등산로o
                    reward += 10

                

                if prev_state[6] and not curr_state[6]: # 임도o -> 임도x
                    reward -= 15
                if prev_state[7] and not curr_state[7]: # 로드o -> 로드x
                    reward -= 15
                if prev_state[10] and not curr_state[10]: # 등산로o -> 등산로x
                    reward -= 15
               

                if prev_state[10] and curr_state[10] :
                    if prev_state[2] > prev_state[2] :
                        reward += 4
                    else :
                        reward -= 2

        return reward
    
    def reward_function5(self, state):
        """
        일반적인 경우의 보상 (보상을 국소적으로 함)
        """
        x, y, elevation, slope, rirsv, wkmstrm, forestroad, road, watershed_basins, channels, climbpath = state

        reward = 0
     
        # 등산 경로에 있는 경우 보상을 부여합니다.
        if climbpath:
            reward += 5

        # 도로에 있는 경우 큰 보상을 부여합니다.
        if road:
            reward += 5

        # 경사가 큰 경우 벌칙을 부여합니다.
        if slope > 0.5:
            reward -= 1
        else:
            reward += 1

        # 유역에 있는 경우 벌칙을 부여합니다.
        if watershed_basins:
            reward -= 5

        # 도로에 있는 경우 큰 보상
        if forestroad:
            reward += 10

        # 등산로에 있는 경우 큰 보상
      
        # 시작점에서 멀어질수록 보상을 부여합니다.
        # reward += 0.1 * (abs(state[0] - self.start_x) + abs(state[1] - self.start_y))

        # 현재 상태를 state_buffer에 추가합니다.
        self.state_buffer.append(state)

        # state_buffer에 2개 이상의 상태가 있는 경우, 이전 상태와 현재 상태를 비교합니다.
        if len(self.state_buffer) > 1:
            for i in range(1, len(self.state_buffer)):
                prev_state = self.state_buffer[i - 1]
                curr_state = self.state_buffer[i]

                # if prev_state[2] > curr_state[2]:
                #     reward += 800
        #6,7,10 
                if prev_state[6] and curr_state[6]: #임도 o -> 임도 o 
                    reward += 3
                if prev_state[7] and curr_state[7]: #도로 o -> 도로 o 
                    reward += 3
                if prev_state[10] and curr_state[10]: #등산로 o -> 등산로 o 
                    reward += 3
                

                if not prev_state[6] and curr_state[6]: #임도 x -> 임도o
                    reward += 10
                if not prev_state[7] and curr_state[7]: #도로 x -> 도로o
                    reward += 10
                if not prev_state[10] and curr_state[10]: #등산로 x -> 등산로o
                    reward += 10

                

                if prev_state[6] and not curr_state[6]: # 임도o -> 임도x
                    reward -= 15
                if prev_state[7] and not curr_state[7]: # 로드o -> 로드x
                    reward -= 15
                if prev_state[10] and not curr_state[10]: # 등산로o -> 등산로x
                    reward -= 15
               

                if prev_state[10] and curr_state[10] :
                    if prev_state[2] > prev_state[2] :
                        reward += 4
                    else :
                        reward -= 2



   
        return reward




    def calculate_reward(self, state, reward_function_index=1):
        if reward_function_index == 1:
            return self.reward_function1(state)
        elif reward_function_index == 2:
            return self.reward_function2(state)
        elif reward_function_index == 3:
            return self.reward_function3(state)
        elif reward_function_index == 4:
            return self.reward_function4(state)
        elif reward_function_index == 5:
            return self.reward_function5(state)
        else:
            raise ValueError(f"Unknown reward function index: {reward_function_index}")
