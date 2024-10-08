# 파일 경로 설정
dem_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/dem/dem.tif'
area_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/area/area.shp'
rirsv_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/river/it_c_rirsv.shp'
wkmstrm_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/river/lt_c_wkmstrm.shp'
climbpath_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/road/lt_l_frstclimb.shp'
road_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/road/road_polygon.shp'
watershed_basins_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/watershed/3/validBasins3.shp'
channels_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/watershed/3/channel3.shp'
forestroad_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/forestroad/31.shp'
q_table_file = '/Users/heekim/Documents/GitHub/SAR_Project/q_table.pkl'
area_difference_file = "/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/area/area_difference.shp"
hiking_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/hiking/ulsan_climb_path.shp'
test_area_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/area/area_difference.shp'

test_area_npy='/Users/heekim/Desktop/heekimjun/SAR_Project_Agent/test_area_result.npy'

# 하이퍼 파라미터 설정
alpha = 0.2  # 학습률
gamma = 0.9  # 할인 인자
epsilon = 0.9  # 탐험 vs 활용 비율
speed = 10  # 이동 속도
beta = 0.1  # 불확실성에 대한 가중치

max_steps = 1000  # 최대 스텝 수
episodes = 50  # 에피소드 수
simulation_max_steps = 2000 # 시뮬레이션 최대 스템

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
