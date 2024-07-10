import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.plot import show
from pyproj import Transformer

def show_path(dem_array, dem_transform, path, road_transformed):
    '''
    경로를 DEM 지도 위에 시각화
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    show(dem_array, transform=dem_transform, ax=ax, cmap='terrain')
    
    # 경로 플롯
    path_x, path_y = zip(*path)
    ax.plot(path_y, path_x, marker='o', color='red', linewidth=2, markersize=5, label='Path')

    # 도로 플롯
    road_coords = np.column_stack(np.where(road_transformed == 1))
    ax.scatter(road_coords[:, 1], road_coords[:, 0], color='blue', s=1, label='Roads')

    ax.set_title('Path Visualization on DEM')
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

