# GIS 기반 강화학습 프로젝트

> 이 프로젝트는 GIS 데이터를 활용하여 강화학습을 통해 조난자의 예상 이동경로 탐색을 수행하는 프로젝트입니다. 주어진 DEM(디지털 고도 모델) 및 여러 지형의 shapefile 데이터를 활용하여 조난자의 예상 이동경로를 모델링합니다.

## Files

- geo_processing.py 
    - GIS 데이터 처리 관련 함수 및 클래스
    - Feature들을 포함하는 Map을 생성
- method
    - q_learning.py 
        - Bayesian Q-Learning 
    - dqn.py
        - Deep Q-Network 
- main.py 
    - 메인 실행 파일
- README.md
    - 프로젝트 설명 파일
- requirements.txt 
    - 프로젝트 의존성 파일
- config.py
    - config
- data 
    - GIS 데이터 폴더 (DEM, shapefile 등)

## Algorithm

- Bayesian Q-Learning(BQN)
    https://cdn.aaai.org/AAAI/1998/AAAI98-108.pdf
- DQN (Deep Q-Network)
    https://arxiv.org/pdf/1312.5602

---
## How to use
1. Clone this repository
   ``` 
   git clone https://github.com/joon0390/SAR_Project.git
   ``` 

2. install dependencies
   ```
   pip install -r requirements.txt
   ``` 
