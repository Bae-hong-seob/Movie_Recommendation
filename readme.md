# Movie Recommendation

## **Abstract**

- 추천시스템 연구 및 학습 용도로 가장 널리 사용되는 MovieLens 데이터 사용.
- 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측
- user-item interaction 정보를 기반으로 AutoEncoder 계열의 모델을 구현하면서 성능 고도화를 진행.

## **Introduction**

- timestamp를 고려한 사용자의 순차적인 이력을 고려, **Implicit feedback**을 사용한다는 점이 **explicit feedback** 기반의 행렬을 사용한 Collaborative Filtering 문제와 차별점. Implicit feedback 기반의 sequential recommendation 시나리오를 바탕으로 사용자의 time-ordred sequence에서 일부 item이 누락(dropout)된 상황을 상정한다. 이는 sequence를 바탕으로 마지막 item만을 예측하는 시나리오보다 복잡하며 실제와 비슷한 상황을 가정. 해당 프로젝트는 여러가지 아이템 (영화)과 관련된 content (side-information)가 존재하기 때문에, side-information 활용이 중요 포인트이다.
![image](https://github.com/Bae-hong-seob/Movie_Recommendation/assets/49437396/fda45a1a-45bf-400d-b4c7-616dbb0a3ba5)


## Dataset
<img width="289" alt="image" src="https://github.com/Bae-hong-seob/Movie_Recommendation/assets/49437396/ef5ec5a9-b161-4695-b972-4ceb49dfe3a8">\
<br>
[MovieLens](https://grouplens.org/datasets/movielens/)
- **input:** user의 implicit 데이터, item(movie)의 meta데이터 (tsv)
- **output:** user에게 추천하는 10개의 item 목록을 user, item이 ','로 구분된 파일(csv) 로 제출

## Team members
1. 김진용 : GRU4Rec 구현
2. 박치언 : SASRec, Bert4Rec 참조 및 구현
3. 배홍섭 : Data Engineering 및 AutoEncoder(AE, DAE, VAE) 구현
4. 안제준 : Sequential model 구현 및 하이퍼 파라미터 튜닝
5. 윤도현 : 데이터 EDA, RecVae 구현

## Experiment
Data Engineering
- rating.csv 파일을 통해 user-item interaction(binary) matrix 생성
- 사용자별 item interaction timestamp 기준으로 sequence 생성
- 전체 user-item interaction을 matrix로 구성하여 AutoEncoder input으로 제공

모델 선정 및 분석: public score기준 각 데이터에서 좋은 성능을 보이는 모델을 선정
- VAE(Variational AutoEncoder) : 0.1467 (DAE << AE << VAE)
- Loss function: MSE (Multinomial <<< MSE)

## Evaluation
<img width="791" alt="Untitled" src="https://github.com/Bae-hong-seob/Movie_Recommendation/assets/49437396/ff4baca1-216f-4cbd-ab3a-724f12d4427f">
