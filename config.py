# 파일 경로 설정
dem_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/dem/dem.tif'
area_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/area/area.shp'
rirsv_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/river/it_c_rirsv.shp'
wkmstrm_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/river/lt_c_wkmstrm.shp'
forestroad_shp_file= '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/road/lt_l_frstclimb.shp'
road_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/road/road_centerline.shp'
watershed_basins_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/watershed/8/validBasins8.shp'
channels_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/watershed/8/channel8.shp'
area_difference_file = "/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/area/area_difference.shp"
channels_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/hiking/ulsan_climb_path.shp'
test_area_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/area/area_difference.shp'

test_area_npy='/Users/heekim/Desktop/heekimjun/SAR_Project_Agent/test_area_result.npy'
featured_npy = '/Users/heekim/Desktop/heekimjun/SAR_Project_Agent/final_rl/featured_dem.npy'
# 하이퍼 파라미터 설정
gamma = 0.99  # 할인 인자
speed = 2  # 이동 속도

max_steps = 720  # 최대 스텝 수
episodes = 100  # 에피소드 수
simulation_max_steps = 240 # 시뮬레이션 최대 스텝

train_iter = 30
target_update_freq = 20
batch_size = 32
memory_size = 1000
buffer_size = 1000
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
