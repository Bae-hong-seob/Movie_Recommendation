# Movie Recommendation

## **Abstract**

- 추천시스템 연구 및 학습 용도로 가장 널리 사용되는 MovieLens 데이터 사용.
- 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측
- user-item interaction 정보를 기반으로 AutoEncoder 계열의 모델을 구현하면서 성능 고도화를 진행.

## **Introduction**

- timestamp를 고려한 사용자의 순차적인 이력을 고려, **Implicit feedback**을 사용한다는 점이 **explicit feedback** 기반의 행렬을 사용한 Collaborative Filtering 문제와 차별점. Implicit feedback 기반의 sequential recommendation 시나리오를 바탕으로 사용자의 time-ordred sequence에서 일부 item이 누락(dropout)된 상황을 상정한다. 이는 sequence를 바탕으로 마지막 item만을 예측하는 시나리오보다 복잡하며 실제와 비슷한 상황을 가정. 해당 프로젝트는 여러가지 아이템 (영화)과 관련된 content (side-information)가 존재하기 때문에, side-information 활용이 중요 포인트이다.
![image](https://github.com/Bae-hong-seob/Movie_Recommendation/assets/49437396/fda45a1a-45bf-400d-b4c7-616dbb0a3ba5)

[Dataset](https://grouplens.org/datasets/movielens/)
<img width="289" alt="image" src="https://github.com/Bae-hong-seob/Movie_Recommendation/assets/49437396/ef5ec5a9-b161-4695-b972-4ceb49dfe3a8">
- **input:** user의 implicit 데이터, item(movie)의 meta데이터 (tsv)
- **output:** user에게 추천하는 10개의 item 목록을 user, item이 ','로 구분된 파일(csv) 로 제출
