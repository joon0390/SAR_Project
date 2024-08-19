import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import json
import pandas as pd

def visualize_paths(dem_array, _path, dem_transform, index, start_points, output_file=None):
    """
    시각화 함수: K-means 클러스터링을 통해 경로를 시각화하고, 경로 중심점과 반지름을 계산하여 출력합니다.
    
    Parameters:
    dem_array (numpy.ndarray): 지형 데이터 배열.
    _path (numpy.ndarray): 에이전트의 경로 데이터.
    dem_transform (Affine): DEM 변환 정보.
    index (int): 경로 인덱스.
    start_points (list): 시작점 목록.
    output_file (str, optional): 시각화 결과를 저장할 파일 이름.
    """
    colors = ['r', 'g', 'b']

    # _path를 K-means 클러스터링
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
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1} ({cluster_sizes[i]} points)')
        
        # 원 추가
        circle = plt.Circle((centers[i][0], centers[i][1]), radii[i], color=colors[i], fill=False, linestyle='--')
        plt.gca().add_patch(circle)

        # 원의 중심점과 반지름을 미터로 변환하여 출력
        center_x_meters, center_y_meters = pixel_to_coords(centers[i][0], centers[i][1], dem_transform)
        radius_in_meters = pixel_distance_to_meters(radii[i], dem_transform)
        print(f"Cluster {i+1} Center: ({centers[i][0]}, {centers[i][1]}) with {cluster_sizes[i]} points and Radius: {radius_in_meters:.2f} meters")
    
    print("Starting Point : ", '(', start_points[0][0], start_points[0][1], ')')
    
    start_points = np.array(start_points)
    plt.scatter(start_points[:, 0], start_points[:, 1], c='k', marker='o', label='Start Point')

    plt.title('K-means Clustering of Paths')
    plt.legend()

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def pixel_to_coords(row, col, transform):
    """Convert pixel coordinates to real world coordinates."""
    x, y = transform * (row, col)
    return x, y

def pixel_distance_to_meters(pixel_distance, transform):
    """Convert pixel distance to meters using the transform."""
    pixel_size_x = transform[0]
    return pixel_distance * pixel_size_x

def plot_losses_and_rewards(losses, rewards, output_file=None):
    """
    손실 및 보상을 시각화하는 함수입니다.

    Parameters:
    losses (list): 학습 손실.
    rewards (list): 에피소드 보상.
    output_file (str, optional): 시각화 결과를 저장할 파일 이름.
    """
    # Create new graph 
    plt.figure(figsize=(14, 6))

    # Plot average rewards (Y-axis) vs episodes (X-axis)
    plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
    plt.plot(rewards)
    plt.title('Rewards Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    
    # Plot losses (Y-axis) vs episodes (X-axis)
    plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
    plt.plot(losses)
    plt.title('Losses Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_losses_from_json(filename='losses.json'):
    with open(filename, 'r') as f:
        all_losses = json.load(f)
    return all_losses

def plot_losses_from_json(filename='losses.json'):
    with open(filename, 'r') as f:
        all_losses = json.load(f)


def array_2_plot(array):
    array = pd.DataFrame(array)
    fig, ax = plt.subplots(figsize=(20, 20)) 
    ax.imshow(array, cmap='gray', interpolation='none') 
    ax.set_title('Array Visualization')
    plt.show()

def get_elevation(dem_array, x, y):
    # 주어진 좌표의 고도 값 반환 (첫 번째 채널 사용)
    return dem_array[x, y]

def calculate_slope(dem_array, x, y):
    # 주어진 좌표의 경사 값 계산 (첫 번째 채널 사용)
    if x <= 0 or x >= dem_array.shape[0] - 1 or y <= 0 or y >= dem_array.shape[1] - 1:
        return 0  # 경계 조건에서 경사는 0으로 설정
    dzdx = (dem_array[x + 1, y] - dem_array[x - 1, y]) / 2  # x 방향의 경사도 계산
    dzdy = (dem_array[x, y + 1] - dem_array[x, y - 1]) / 2  # y 방향의 경사도 계산
    slope = np.sqrt(dzdx**2 + dzdy**2)  # 경사도 계산
    return slope

def get_random_index(array):
    non_zero_indices = np.nonzero(array)
    non_zero_indices = list(zip(*non_zero_indices))
    random_index = np.random.choice(len(non_zero_indices))
    return non_zero_indices[random_index]

def check_true_values_in_array(target_array, array_name="Array"):
    true_count = np.sum(target_array)
    print(f"Debug: Total number of True values in {array_name}: {true_count}")
    if true_count > 0:
        print(f"Debug: True values found in {array_name}")
    else:
        print(f"Debug: No True values found in {array_name}")


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    def plotter(json_file):
        # JSON 파일에서 손실 데이터를 로드합니다.
        with open(json_file, 'r') as file:
            losses = json.load(file)
        
        # 손실 데이터가 리스트 또는 단일 값인 경우를 처리합니다.
        if isinstance(losses, list) and isinstance(losses[0], list):
            # 각 에피소드의 손실 리스트에서 평균 손실 값을 계산합니다.
            avg_losses = [sum(episode_losses)/len(episode_losses) if len(episode_losses) > 0 else 0 for episode_losses in losses]
        elif isinstance(losses, list):
            # 손실 데이터가 리스트의 리스트가 아닌 경우 그대로 사용합니다.
            avg_losses = losses
        else:
            raise ValueError("Unsupported data format in the JSON file")

        # 그래프를 생성합니다.
        plt.figure(figsize=(10, 6))
        plt.plot(avg_losses, label='Total Rewards per Episode', color='blue')
        
        # 그래프에 제목, 레이블, 범례 등을 추가합니다.
        plt.title('Reward over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Total Rewards')
        plt.legend()
        plt.grid(True)
        
        # 그래프를 화면에 출력합니다.
        plt.show()



    plotter('/Users/heekim/Desktop/losses.json')
    plotter('/Users/heekim/Desktop/rewards.json')
