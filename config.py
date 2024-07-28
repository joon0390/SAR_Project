dem_file= '/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/dem/dem.tif'
area_file = '/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/area/area.shp'
rirsv_shp_file = '/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/river/it_c_rirsv.shp'
wkmstrm_shp_file = '/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/river/lt_c_wkmstrm.shp'
climbpath_shp_file = '/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/road/lt_l_frstclimb.shp'
road_shp_file = '/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/road/road_polygon.shp'
watershed_basins_shp_file = '/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/watershed/basins_valid.shp'
channels_shp_file = '/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/watershed/channels.shp'
q_table_file = '/Users/heekim/Documents/GitHub/SAR_Project/q_table.pkl'
forestroad_shp_file = '/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/forestroad/clipped_31.shp' #추가
hiking_shp_file = '/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/climb/ulsan_climb_path.shp' #추가
area_difference_file = "/Users/kwonheedam/Desktop/data/GIS_데이터_20240704/area/area_difference.shp"
test_area_npy = '/Users/kwonheedam/Desktop/SAR_Project/test_area_result.npy'
# 하이퍼 파라미터 설정
alpha = 0.2  # 학습률
gamma = 0.9  # 할인 인자
epsilon = 0.9  # 탐험 vs 활용 비율
speed = 10  # 이동 속도

beta = 0.1  # 불확실성에 대한 가중치

max_steps = 10  # 최대 스텝 수
episodes = 10  # 에피소드 수
simulation_max_steps = 20 # 시뮬레이션 최대 스템

target_update = 10
batch_size = 16
memory_size=10000

# 연령, 성별, 건강 상태에 따른 설정들
agent_config = {
    'young_male_good': {'speed': 2, 'explore_rate': 0.8, 'stay_put_prob': 0.1},
    'young_male_bad': {'speed': 1, 'explore_rate': 0.7, 'stay_put_prob': 0.3},
    'young_female_good': {'speed': 2, 'explore_rate': 0.7, 'stay_put_prob': 0.2},
    'young_female_bad': {'speed': 1, 'explore_rate': 0.6, 'stay_put_prob': 0.4},
    'old_male_good': {'speed': 2, 'explore_rate': 0.6, 'stay_put_prob': 0.3},
    'old_male_bad': {'speed': 1, 'explore_rate': 0.4, 'stay_put_prob': 0.5},
    'old_female_good': {'speed': 2, 'explore_rate': 0.5, 'stay_put_prob': 0.3},
    'old_female_bad': {'speed': 1, 'explore_rate': 0.4, 'stay_put_prob': 0.6},
}
