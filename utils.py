import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from pyproj import Transformer
from rasterio.transform import Affine

def show_path_on_dem(dem_array, path):
    '''
    경로를 DEM 지도 위에 시각화
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    show(dem_array, ax=ax, cmap='terrain')
    
    # 경로 플롯 (DEM 배열의 인덱스를 사용)
    path_x, path_y = zip(*path)
    ax.plot(path_y, path_x, marker='o', color='red', linewidth=2, markersize=5, label='Path')

    ax.set_title('Path Visualization on DEM')
    ax.legend()
    
    plt.show()

def show_path_with_arrows(dem_array, path):
    '''
    경로를 DEM 지도 위에 화살표로 시각화하는 함수
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    show(dem_array, ax=ax, cmap='terrain')

    # 경로 시각화
    path_x, path_y = zip(*path)

    # 화살표의 방향을 계산
    dx = np.diff(path_y)
    dy = np.diff(path_x)

    # 경로 시작점과 화살표 표시
    ax.quiver(path_y[:-1], path_x[:-1], dx, dy, scale_units='xy', angles='xy', scale=1, color='red', label='Path', headwidth=3, headlength=5)
    ax.plot(path_y, path_x, 'ro')  # 경로의 각 지점에 점 표시

    ax.set_title('Path Visualization on DEM with Arrows')
    ax.legend()

    plt.show()

def array_2_plot(array):
    '''
    각 shapefile을 변환된 DEM 영역에 맞춰 변환한 array가 input
    '''
    array = pd.DataFrame(array)
    fig, ax = plt.subplots(figsize=(20, 20)) 
    ax.imshow(array, cmap='gray', interpolation='none') 
    ax.set_title('Array Visualization')
    plt.show()

def get_elevation(x, y, dem_array):
    # 주어진 좌표의 고도 값 반환
    return dem_array[x, y]

def calculate_slope(x, y, dem_array):
    # 주어진 좌표의 경사 값 계산
    if x <= 0 or x >= dem_array.shape[0] - 1 or y <= 0 or y >= dem_array.shape[1] - 1:
        return 0  # 경계 조건에서 경사는 0으로 설정
    dzdx = (dem_array[x + 1, y] - dem_array[x - 1, y]) / 2  # x 방향의 경사도 계산
    dzdy = (dem_array[x, y + 1] - dem_array[x, y - 1]) / 2  # y 방향의 경사도 계산
    slope = np.sqrt(dzdx**2 + dzdy**2)  # 경사도 계산
    return slope

def plot_dem_and_shapefile(dem_file_path, shapefile_path, cmap='terrain', shapefile_color='red'):
    """
    DEM 파일과 셰이프파일을 함께 그리는 함수
    
    Parameters:
    - dem_file_path: str, DEM 파일의 경로
    - shapefile_path: str, 셰이프파일의 경로
    - cmap: str, DEM 파일의 컬러맵 (기본값은 'terrain')
    - shapefile_color: str, 셰이프파일의 경계선 색상 (기본값은 'red')
    """
    # DEM 파일 읽기
    with rasterio.open(dem_file_path) as dem_dataset:
        dem_data = dem_dataset.read(1)
        dem_transform = dem_dataset.transform

    # 셰이프 파일 읽기
    shapefile_data = gpd.read_file(shapefile_path)

    # 플롯 설정
    fig, ax = plt.subplots(figsize=(10, 10))

    # DEM 데이터 플롯
    show(dem_data, transform=dem_transform, ax=ax, cmap=cmap)

    # 셰이프 파일 플롯
    shapefile_data.plot(ax=ax, facecolor='none', edgecolor=shapefile_color)

    plt.title('DEM and Shapefile Overlay')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def print_dem_and_shapefile_values(dem_file_path, shapefile_path):
    """
    DEM 파일과 셰이프파일의 값을 출력하는 함수
    
    Parameters:
    - dem_file_path: str, DEM 파일의 경로
    - shapefile_path: str, 셰이프파일의 경로
    """
    # DEM 파일 읽기
    with rasterio.open(dem_file_path) as dem_dataset:
        dem_data = dem_dataset.read(1)
        dem_transform = dem_dataset.transform

    # 셰이프 파일 읽기
    shapefile_data = gpd.read_file(shapefile_path)

    # DEM 데이터 출력
    print("DEM Data:")
    print(dem_data)

    # 셰이프 파일 데이터 출력
    print("\nShapefile Data:")
    print(shapefile_data)

if __name__ == '__main__':
    import pickle
    def load_q_table(file_path):
        with open(file_path, "rb") as f:
            q_mean, q_variance = pickle.load(f)
        return q_mean, q_variance
    def print_q_values(q_mean, x, y):
        print(f"Q-Values at ({x}, {y}): {q_mean[x, y]}")

    from config import *
    dem = dem_path
    shapefile = rirsv_shp_file
    print_dem_and_shapefile_values(dem, shapefile)


    q_mean, q_variance = load_q_table("q_table.pkl")

    # 특정 상태의 Q-값 확인 (예: x=100, y=200)
    x, y = 1000, 300
    print_q_values(q_mean, x, y)