import os
import numpy as np
from geo_processing import GISProcessor, load_shapefiles
from reward2 import RewardCalculator
from kmeans_dqn2 import dqn_learning, simulate_path, load_model, Agent, DQN ,plot_loss_from_json
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
    # Assuming square pixels, we use the x pixel size for conversion.
    return pixel_distance * pixel_size_x

def meters_to_pixel_distance(meters, transform):
    """Convert distance in meters to pixel distance using the transform."""
    pixel_size_x = transform[0]
    # Assuming square pixels, we use the x pixel size for conversion.
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
    
    action_mode = '8_directions'  # or '8_directions'

    # Agent 인스턴스 생성
    agent = Agent(age_group='young', gender='male', health_status='good')
    _path = []
    paths = []
    index = 3 #임의로 100번째 step 위치
    epsilon_array = np.array([0.9,0.85,0.8,0.75,0.7])
    lr_array = np.array([0.001,0.005,0.01])
    gamma_array = np.array([0.9,0.8])
    len_path = []

    reward_function_index = 1
    test_area = np.load(test_area_npy)
    coord = get_random_index(test_area)
    start_x, start_y = coord[0], coord[1]
    th = 1
    for epsilon in epsilon_array:
        for lr in lr_array:
             for gamma in gamma_array:
        # DQN 학습 수행
                print(th,"번째 파라미터")
                print('epsilon :' ,epsilon,"lr :",lr,"gamma:",gamma)

                model, all_losses = dqn_learning(dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed, forestroad_transformed, hiking_transformed, reward_calculator, agent, action_mode=action_mode, load_existing = False, model_filename = 'dqn_model.pth',_lr=lr, _gamma=gamma,max_steps=max_steps,episodes=episodes, reward_function_index=reward_function_index)
                
                # 경로 시뮬레이션 예시
                
                

                model = load_model('dqn_model.pth', input_dim=12, output_dim=8 if action_mode == '8_directions' else 6)
                path = simulate_path(start_x, start_y, model, dem_array, rirsv_transformed, wkmstrm_transformed, climbpath_transformed, road_transformed, watershed_basins_transformed, channels_transformed, forestroad_transformed,hiking_transformed,agent, action_mode=action_mode)
                paths.append(path)
                _path.append(path[index-1]) #path에서 index번째 추가
                print("---------------",th+1,"번째 model의 path","-------------------------")
                th+= 1
    
    _path = np.array(_path)
    # _path를 K-means 클러스터링

    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(_path)

    '-------------------------'
    # 클러스터 중심 계산
    
    centers = kmeans.cluster_centers_

    # 각 클러스터의 반지름 계산 (최대 50 인덱스 기준)
    def calculate_radius(cluster_points, center, max_radius_index=50):
        distances = np.linalg.norm(cluster_points - center, axis=1)
        max_radius = min(np.max(distances), max_radius_index)
        return max_radius

    # 각 클러스터의 반지름 계산
    radii = [calculate_radius(_path[kmeans.labels_ == i], centers[i]) for i in range(3)]

    #각 클러스태의 개수

    cluster_sizes = [np.sum(kmeans.labels_ == i) for i in range(3)]

    plt.figure(figsize=(10, 10))
    plt.imshow(dem_array, cmap='terrain')
    colors = ['r', 'g', 'b']
    for i in range(3):
        cluster_points = _path[kmeans.labels_ == i]
        plt.scatter(cluster_points[:, 1], cluster_points[:, 0], c=colors[i], label=f'Cluster {i+1}')
        
        # 원 추가
        circle = plt.Circle((centers[i][1], centers[i][0]), radii[i], color=colors[i], fill=False, linestyle='--')
        plt.gca().add_patch(circle)
        
        # 원 중심점 추가 (작게)
        plt.scatter(centers[i][1], centers[i][0], c=colors[i], marker='x', s=50)
        
        center_x, center_y = pixel_to_coords(centers[i][0], centers[i][1], dem_transform)
        radius_in_meters = pixel_distance_to_meters(radii[i], dem_transform)
        print(f"Cluster {i+1} Center: ({center_x}, {center_y}) with {cluster_sizes[i]} points and Radius: {radius_in_meters:.2f} meters")
        point_xy = []
        # 원의 중심 좌표와 추출한 점의 좌표 출력
        for points in cluster_points:
            point_x,point_y = pixel_to_coords(points[0],points[1],dem_transform)
            point_xy.append([point_x,point_y])
        
        print(f"Cluster {i+1} points: {point_xy}")

        # 각 클러스터의 반경 출력
    for i, radius in enumerate(radii):
        print(f"Cluster {i+1} radius: {radius * 5:.2f} meters")
    
    plt.scatter(x=start_x,y=start_y,c="black")

    plt.title('K-means Clustering of Paths')
    plt.legend()
    plt.show()

    plot_loss_from_json('losses.json')
    # colors = ['r', 'g', 'b']
    # for i in range(3):        
    #     # 원 추가

    #     # 원의 중심점 실제 좌표로 변환하여 출력
    #     center_x, center_y = pixel_to_coords(centers[i][0], centers[i][1], dem_transform)
    #     radius_in_meters = pixel_distance_to_meters(radii[i], dem_transform)
    #     print(f"Cluster {i+1} Center: ({center_x}, {center_y}) with {cluster_sizes[i]} points and Radius: {radius_in_meters:.2f} meters")

    # plt.title('K-means Clustering of Paths')
    # plt.legend()
    # plt.show()

    #     plot_loss_from_json('/Users/heekim/Desktop/heekimjun/SAR_Project_Agent/losses.json')



