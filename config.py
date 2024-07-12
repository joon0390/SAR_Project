# 파일 경로 설정
dem_path = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/dem/dem.tif'
area_path = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/area/area.shp'
rirsv_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/river/it_c_rirsv.shp'
wkmstrm_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/river/lt_c_wkmstrm.shp'
road_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/road/lt_l_frstclimb.shp'
watershed_basins_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/watershed/basins_valid.shp'
channels_shp_file = '/Users/heekim/Desktop/heekimjun/WiSAR/data/GIS 데이터_20240704/watershed/channels.shp'
q_table_file = '/Users/heekim/Documents/GitHub/SAR_Project/q_table.pkl'

# 하이퍼 파라미터 설정
alpha = 0.3  # 학습률
gamma = 0.9  # 할인 인자
epsilon = 0.8  # 탐험 vs 활용 비율
beta = 0.01  # 불확실성에 대한 가중치
speed = 20  # 이동 속도
max_steps = 1000  # 최대 스텝 수
episodes = 1000  # 에피소드 수
simulation_max_steps = 1000 # 시뮬레이션 최대 스템

target_update = 10
batch_size = 16
memory_size=10000


#DQN 하이퍼 파라미터
# # 하이퍼 파라미터 설정
dqn_alpha = 0.001  # 학습률
dqn_gamma = 0.9  # 할인 인자
dqn_epsilon = 0.8  # 탐험 vs 활용 비율
dqn_beta = 0.01  # 불확실성에 대한 가중치
dqn_replay_buffer = deque(maxlen=10000)
dqn_batch_size = 64
dqn_input_dim = 9  # 상태의 차원
dqn_output_dim = 6  # 행동의 차원
