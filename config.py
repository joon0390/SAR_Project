# 파일 경로 설정
dem_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/dem/dem.tif'
area_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/area/area.shp'
rirsv_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/river/it_c_rirsv.shp'
wkmstrm_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/river/lt_c_wkmstrm.shp'
climbpath_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/road/lt_l_frstclimb.shp'
road_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/road/road_polygon.shp'
watershed_basins_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/watershed/8/validBasins8.shp'
channels_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/watershed/8/channel8.shp'
q_table_file = '/Users/heekim/Documents/GitHub/SAR_Project/q_table.pkl'
area_difference_file = "/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/area/area_difference.shp"

# 하이퍼 파라미터 설정
alpha = 0.2  # 학습률
gamma = 0.9  # 할인 인자
epsilon = 0.9  # 탐험 vs 활용 비율
speed = 10  # 이동 속도

beta = 0.1  # 불확실성에 대한 가중치

max_steps = 1000  # 최대 스텝 수
episodes = 100  # 에피소드 수
simulation_max_steps = 1000 # 시뮬레이션 최대 스템

target_update = 10
batch_size = 16
memory_size=10000

# 연령, 성별, 건강 상태에 따른 설정들
agent_config = {
    'young_male_good': {'speed': 20, 'explore_rate': 0.8, 'stay_put_prob': 0.1},
    'young_male_bad': {'speed': 10, 'explore_rate': 0.7, 'stay_put_prob': 0.3},
    'young_female_good': {'speed': 10, 'explore_rate': 0.7, 'stay_put_prob': 0.2},
    'young_female_bad': {'speed': 10, 'explore_rate': 0.6, 'stay_put_prob': 0.4},
    'old_male_good': {'speed': 10, 'explore_rate': 0.6, 'stay_put_prob': 0.3},
    'old_male_bad': {'speed': 5, 'explore_rate': 0.4, 'stay_put_prob': 0.5},
    'old_female_good': {'speed': 10, 'explore_rate': 0.5, 'stay_put_prob': 0.3},
    'old_female_bad': {'speed': 5, 'explore_rate': 0.4, 'stay_put_prob': 0.6},
}
