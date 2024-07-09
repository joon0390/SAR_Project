def array_2_plot(array):
    '''
    각 shapefile을 변환된 DEM 영역에 맞춰 변환한 array가 input
    '''
    array = pd.DataFrame(array)
    fig, ax = plt.subplots(figsize=(20, 20)) 
    ax.imshow(watershed_basins_transformed, cmap='gray', interpolation='none') 
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

def discretize_state(state, q_mean):
    # 상태를 디스크리트 상태로 변환
    x, y = state[:2]  # 상태에서 x, y 좌표 추출
    max_x, max_y = q_mean.shape[0] - 1, q_mean.shape[1] - 1  # Q-테이블의 최대 인덱스
    return min(x // 10, max_x), min(y // 10, max_y)  # 좌표를 10으로 나누고 최대값을 넘지 않도록 제한