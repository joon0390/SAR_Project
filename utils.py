import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.plot import show
from pyproj import Transformer
from rasterio.transform import Affine
import os
from shapely.geometry import Point

def load_and_print_npy(filename, slice_range=None):
    if os.path.exists(filename):
        array = np.load(filename)
        print(f"Array loaded from {filename}")
        print(f"Array shape: {array.shape}")
        
        if slice_range:
            sliced_array = array[slice_range]
            print(f"Sliced Array shape: {sliced_array.shape}")
            print(f"Sliced Array contents:\n{sliced_array}")
        else:
            print(f"Array contents:\n{array}")
    else:
        print(f"{filename} does not exist.")
        
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

# def get_random_point_within_polygon(shapefile_path, reference_shapefile_path, dem_shape):
#     # Load the shapefiles
#     gdf = gpd.read_file(shapefile_path)
#     reference_gdf = gpd.read_file(reference_shapefile_path)
    
#     # Assuming there's only one polygon in each shapefile
#     polygon = gdf.geometry[0]
#     reference_polygon = reference_gdf.geometry[0]
    
#     # Get the coordinate reference systems (CRS)
#     src_crs = gdf.crs
#     dst_crs = reference_gdf.crs
    
#     # Initialize the transformer
#     transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
#     # Get the bounding box of the polygon
#     minx, miny, maxx, maxy = polygon.bounds
    
#     while True:
#         # Generate random points within the bounding box
#         pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        
#         # Check if the point is within the polygon
#         if polygon.contains(pnt):
#             # Transform the point to the reference CRS
#             pnt_x, pnt_y = transformer.transform(pnt.x, pnt.y)
            
#             # Convert the coordinates to DEM array indices
#             dem_x = int((pnt_x - reference_polygon.bounds[0]) / (reference_polygon.bounds[2] - reference_polygon.bounds[0]) * dem_shape[0])
#             dem_y = int((pnt_y - reference_polygon.bounds[1]) / (reference_polygon.bounds[3] - reference_polygon.bounds[1]) * dem_shape[1])
            
#             # Check if the transformed point is within the DEM bounds
#             if 0 <= dem_x < dem_shape[0] and 0 <= dem_y < dem_shape[1]:
#                 return dem_x, dem_y

def get_random_point_within_polygon(shapefile_path, dem_shape):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Assuming there's only one polygon in the shapefile
    polygon = gdf.geometry[0]
    
    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    
    # Define the transform
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, dem_shape[1], dem_shape[0])
    # Ensure the transform is an Affine object
    if not isinstance(transform, Affine):
        transform = Affine(*transform)
    
    try:
        inverse_transform = ~transform
    except ValueError as ve:
        print(f"Transform matrix is degenerate: {ve}")
        return None, None

    while True:
        # Generate random points within the bounding box
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        
        # Check if the point is within the polygon
        if polygon.contains(pnt):
            # Convert point coordinates to the array indices
            row, col = inverse_transform * (pnt.x, pnt.y)
            if 0 <= row < dem_shape[0] and 0 <= col < dem_shape[1]:
                return int(row), int(col)
    
    # If no valid point is found within the polygon bounds
    print("Failed to find a valid point within the polygon.")
    return None, None

def visualize_watershed_array(npy_file):
    """
    Load and visualize a single watershed array from an npy file.
    
    Parameters:
    npy_file (str): Path to the npy file containing the watershed array.
    """
    # Load the array from the npy file
    watershed_array = np.load(npy_file)
    
    # Visualize the array
    plt.figure(figsize=(10, 10))
    plt.imshow(watershed_array, cmap='gray')
    plt.title('Watershed Array')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    filename = '/Users/heekim/Desktop/heekimjun/SAR_Project_Agent/test_area.npy'
    files = '/Users/heekim/Documents/GitHub/SAR_Project/featured_dem.npy'

   # 1. numpy 배열 로드
    test_area = np.load(filename)
    combined_array = np.load(filename)
    print(combined_array.shape)

    dem_array = combined_array[:, :, 0]
    rirsv_transformed = combined_array[:, :, 1]
    wkmstrm_transformed = combined_array[:, :, 2]
    climbpath_transformed = combined_array[:, :, 3]
    watershed_basins_transformed = combined_array[:, :, 4]
    channels_transformed = combined_array[:, :, 5]

    # 2. 배열의 기본 정보 출력
    print(f"Array shape: {test_area.shape}")
    print(f"Array contents:\n{test_area}")

    # 3. 값이 0이 아닌 요소 확인
    non_zero_indices = np.nonzero(test_area)
    non_zero_values = test_area[non_zero_indices]

    # 4. 값이 0이 아닌 요소들의 개수 확인
    non_zero_count = len(non_zero_values)
    print(f"Number of non-zero elements: {non_zero_count}")

    # 5. 값이 0이 아닌 요소들의 위치 확인
    print(f"Non-zero elements are located at indices:\n{non_zero_indices}")

    # 6. 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(rirsv_transformed, cmap='gray')
    if non_zero_count > 0:
        plt.scatter(non_zero_indices[1], non_zero_indices[0], color='red', s=1)  # 빨간 점으로 표시
    plt.title('Non-zero elements in test_area.npy')
    plt.show()