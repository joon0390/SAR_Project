import os
import numpy as np
from geo_processing import GISProcessor, load_shapefiles
from reward import RewardCalculator
from kmeans_dqn import  dqn_learning, simulate_path, load_model,Agent,DQN
from utils import show_path_with_arrows, get_random_index
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from config import *
if __name__ == "__main__":
    filename = 'featured_dem.npy'

    # .npy 파일을 로드
    if os.path.exists(filename):
        combined_array = np.load(filename)
        print(f"Loaded combined array from {filename}")
        print(f"Combined array shape: {combined_array.shape}")
    else:
        print(f"{filename} does not exist. Please ensure the file is available.")
        exit(1)

    # 채널을 각각 분리
    dem_array = combined_array[:, :, 0]
    rirsv_transformed = combined_array[:, :, 1]
    wkmstrm_transformed = combined_array[:, :, 2]
    climbpath_transformed = combined_array[:, :, 3]
    road_transformed = combined_array[:, :, 4]
    watershed_basins_transformed = combined_array[:, :, 5]
    channels_transformed = combined_array[:, :, 6]
    forestroad_transformed = combined_array[:,:,7]
    hiking_transformed = combined_array[:,:,8]

    # 보상 계산기 인스턴스 생성
    reward_calculator = RewardCalculator(dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed,forestroad_transformed,hiking_transformed)
    
    action_mode = '8_directions'  # or '8_directions'

    # Agent 인스턴스 생성∆
    agent = Agent(age_group='young', gender='male', health_status='good')
    _path = []
    paths = []
    index = 100 #임의로 100번째 step 위치
    epsilon_array = np.array([0.9,0.85,0.8,0.75,0.7,0.65])
    lr_array = np.array([0.001,0.002,0.003,0.004,0.005,0.01,0.1])
    gamma_array = np.array([0.95,0.9,0.85,0.8])

    for i in range(25):
    # DQN 학습 수행
            print(i+1,"번째 파라미터")
            epsilon = np.random.choice(epsilon_array)
            lr = np.random.choice(lr_array)
            gamma = float(np.random.choice(gamma_array))
            print('epsilon :' ,epsilon,"lr :",lr,"gamma:",gamma)

            dqn_learning(dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed,forestroad_transformed, hiking_transformed,reward_calculator, agent, action_mode=action_mode,_lr = lr,_epsilon =epsilon,_gamma=gamma)
            # 경로 시뮬레이션 예시
            test_area = np.load(test_area_npy)
            coord = get_random_index(test_area)
            start_x, start_y = coord[0], coord[1]
            
            #start_x, start_y = 2546,3155
            model = load_model('dqn_model.pth', input_dim=12, output_dim=8 if action_mode == '8_directions' else 6)
            path = simulate_path(start_x, start_y, model, dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed, forestroad_transformed,hiking_transformed,agent, action_mode=action_mode)
            paths.append(path)
            _path.append(path[index-1]) #path에서 index번째 추가
            print("---------------",i+1,"번째 model의 path","-------------------------")

    _path = np.array(_path)

    # _path를 K-means 클러스터링
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(_path)

    # 클러스터 중심 계산
    centers = kmeans.cluster_centers_

    # 각 클러스터의 반지름 계산 (최대 거리 사용)
    radii = [np.max(np.linalg.norm(_path[kmeans.labels_ == i] - centers[i], axis=1)) for i in range(3)]

    # 클러스터 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(dem_array, cmap='terrain')

    colors = ['r', 'g', 'b']
    for i in range(3):
        cluster_points = _path[kmeans.labels_ == i]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], c=colors[i], label=f'Cluster {i+1}')
        
        # 원 추가
        circle = plt.Circle((centers[i][1], centers[i][0]), radii[i], color=colors[i], fill=False, linestyle='--')
        plt.gca().add_patch(circle)

    plt.title('K-means Clustering of Paths')
    plt.legend()
    plt.show()

