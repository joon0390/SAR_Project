import os
import numpy as np
from geo_processing import GISProcessor, load_shapefiles
from reward import RewardCalculator
from kmeans_dqn import dqn_learning, simulate_path, load_model, Agent, DQN, plot_loss_from_json

from utils import show_path_with_arrows, get_random_index
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from config import *
import rasterio

def pixel_to_coords(row, col, transform):
    """Convert pixel coordinates to real world coordinates."""
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y

def pixel_distance_to_meters(pixel_distance, transform):
    """Convert pixel distance to meters using the transform."""
    pixel_size_x = transform[0]
    return pixel_distance * pixel_size_x

def meters_to_pixel_distance(meters, transform):
    """Convert distance in meters to pixel distance using the transform."""
    pixel_size_x = transform[0]
    return meters / pixel_size_x

if __name__ == "__main__":
    filename = 'featured_dem.npy'
    test_area_npy = 'test_area_result.npy'

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
    forestroad_transformed = combined_array[:, :, 7]
    hiking_transformed = combined_array[:, :, 8]

    # GISProcessor 인스턴스를 통해 DEM 변환 정보 가져오기
    gis_processor = GISProcessor(dem_file)
    dem_transform = gis_processor.dem_transform

    # 보상 계산기 인스턴스 생성
    reward_calculator = RewardCalculator(dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed, forestroad_transformed, hiking_transformed)
    
    action_mode = 'custom'  # or '8_directions'

    # Agent 인스턴스 생성
    agent = Agent(age_group='young', gender='male', health_status='good')
    _path = []
    paths = []
    start_points = []
    index = 100 # 임의로 100번째 step 위치
    epsilon_array = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65])
    lr_array = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    gamma_array = np.array([0.99, 0.97, 0.95, 0.93])
    decay_factor = 0.95
    reward_function_index = 1
    
    # 동일한 시작점 선택
    test_area = np.load(test_area_npy)
    coord = get_random_index(test_area)
    start_x, start_y = coord[0], coord[1]
    start_points.append((start_x, start_y))

    for i in range(20):
        print(i + 1, "번째 파라미터")
        epsilon = np.random.choice(epsilon_array)
        lr = np.random.choice(lr_array)
        gamma = float(np.random.choice(gamma_array))
        print('epsilon :', epsilon, "lr :", lr, "gamma:", gamma)

        # 에이전트 속도 감소 적용 (지수 감소 방식)
        agent.speed = max(1, agent.speed * decay_factor)

        model, all_losses = dqn_learning(dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed, forestroad_transformed, hiking_transformed, reward_calculator, agent, action_mode=action_mode, load_existing = False, model_filename = 'dqn_model.pth',_lr=lr, _epsilon=epsilon, _gamma=gamma, reward_function_index=reward_function_index)
        
        model = load_model('dqn_model.pth', input_dim=12, output_dim=8 if action_mode == '8_directions' else 5)
        path = simulate_path(start_x, start_y, model, dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed, forestroad_transformed, hiking_transformed, agent, action_mode=action_mode)
        paths.append(path)
        _path.append(path[index - 1]) # path에서 index번째 추가
        print("---------------", i + 1, "번째 model의 path", "-------------------------")

    _path = np.array(_path)

    colors = ['r', 'g', 'b']

    # _path를 K-means 클러스터링
    #kmeans = KMeans(n_clusters=len(colors), n_init=10)
    kmeans = KMeans(n_clusters=len(colors))

    kmeans.fit(_path)

    # 클러스터 중심 계산
    centers = kmeans.cluster_centers_

    # 각 클러스터의 반지름 계산 (최대 거리 사용)
    radii = [np.max(np.linalg.norm(_path[kmeans.labels_ == i] - centers[i], axis=1)) for i in range(len(colors))]

    # 각 클러스터의 점 개수
    cluster_sizes = [np.sum(kmeans.labels_ == i) for i in range(len(colors))]
    
    # 클러스터 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(dem_array, cmap='terrain')

    for i in range(len(colors)):
        cluster_points = _path[kmeans.labels_ == i]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], c=colors[i], label=f'Cluster {i+1} ({cluster_sizes[i]} points)')
        
        # 원 추가
        circle = plt.Circle((centers[i][1], centers[i][0]), radii[i], color=colors[i], fill=False, linestyle='--')
        plt.gca().add_patch(circle)

        # 원의 중심점과 반지름을 미터로 변환하여 출력
        center_x_meters, center_y_meters = pixel_to_coords(centers[i][0], centers[i][1], dem_transform)
        radius_in_meters = pixel_distance_to_meters(radii[i], dem_transform)
        print(f"Cluster {i+1} Center: ({centers[i][1]}, {centers[i][0]}) with {cluster_sizes[i]} points and Radius: {radius_in_meters:.2f} meters")

    # 시작점을 검정색 점으로 표시
    start_points = np.array(start_points)
    plt.scatter(start_points[:, 1], start_points[:, 0], c='k', marker='o', label='Start Point')

    plt.title('K-means Clustering of Paths')
    plt.legend()
    plt.show()
    
    plot_loss_from_json('/Users/heekim/Desktop/heekimjun/SAR_Project_Agent/losses.json')
