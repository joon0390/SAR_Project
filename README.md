# GIS 기반 강화학습 프로젝트

이 프로젝트는 GIS 데이터를 활용하여 강화학습을 통해 조난자의 예상 경로 탐색을 수행하는 프로젝트입니다. 주어진 DEM(디지털 고도 모델) 및 여러 지형 shapefile 데이터를 활용하여 조난자의 예상 경로를 학습합니다.

## 프로젝트 구조

- geo_processing.py 
    - GIS 데이터 처리 관련 함수 및 클래스
- q_learning.py 
    - 강화학습 알고리즘 구현
- main.py 
    - 메인 실행 파일
- README.md
    - 프로젝트 설명 파일
- requirements.txt 
    - 프로젝트 의존성 파일
- data 
    - GIS 데이터 폴더 (DEM, shapefile 등)
