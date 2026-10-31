# KNN (K-Nearest Neighbor) — 붓꽃 분류

인공지능개론 실습 · 거리 기반 분류  
붓꽃(iris) 데이터로 KNN 분류기를 `fitcknn`으로 먼저 써보고 → **직접 구현**하고 → 3품종·4특징으로 일반화한 뒤 → 결정 경계까지 그려봤다.

> **KNN = 가장 가까운 k개 이웃의 다수결.** 새 데이터가 들어오면 학습 데이터와의 거리를 재서 가까운 k개를 뽑고, 그중 가장 많은 품종으로 분류한다.

```mermaid
flowchart LR
    A["새 데이터"] --> B["① 거리 계산<br>(유클리디안)"]
    B --> C["② 거리순 정렬<br>sort"]
    C --> D["③ 가까운 k개 선택"]
    D --> E["④ 다수결<br>mode"]
    E --> F["예측 품종"]
```

| 파일 | 내용 |
| --- | --- |
| `ex1_dataload_visualization.m` | 데이터 불러오기 → 품종을 숫자로 변환 → sepal / petal 산점도 → 학습·평가 데이터 분리 |
| `ex2_knn_using_matlab.m` | `fitcknn` + `predict` 두 줄로 KNN (versicolor vs virginica, sepal 2특징, k=3) |
| `ex3_knn_my_style.m` | 같은 문제를 거리 계산부터 **직접 구현**, 마지막에 `fitcknn` 결과와 비교 |
| `ex4_knn_generalization.m` | 거리 계산을 벡터화(`sum((a-b).^2)`)해서 **3품종·4특징**으로 확장 (k=5) |
| `ex5_knn_space.m` | `meshgrid` 격자점을 전부 분류해 **결정 경계** 시각화 (sepal 2특징, k=50) |

- 데이터 분할: ex2·ex3은 학습 60개 / 평가 40개, ex4·ex5는 품종별 21\~50번째를 학습(90개), ex4는 1\~20번째를 평가(60개)
- ex2·ex3 (sepal 2특징, k=3): 평가 40개 중 **24개 정답 (60.0%)** — versicolor와 virginica가 sepal에서 많이 겹쳐서 어렵다
- ex4 (특징 4개, k=5): 평가 60개 중 **59개 정답 (98.3%)**, 같은 분할에서 sepal 2개만 쓰면 45개 (75.0%)
- 직접 구현한 코드는 Python으로 같은 로직을 재현해 scikit-learn `KNeighborsClassifier`와 예측이 모두 일치하는 것을 확인했다. 위 정확도도 이 재현 값이다.

![scatter](images/knn_fig1_scatter.png)
![confusion](images/knn_fig2_ex4_confusion.png)
![k accuracy](images/knn_fig3_k_accuracy.png)
![space](images/knn_fig4_space.png)

**정리**
1. 직접 구현과 내장함수는 같은 4단계(거리 → 정렬 → 상위 k개 → 다수결)이고, 결과도 같다.
2. 벡터화하면 특징이 2개든 4개든, 품종이 2개든 3개든 같은 코드를 쓸 수 있다.
3. k는 모델 복잡도를 조절한다. 작으면 경계가 울퉁불퉁하고 점 하나에 민감하고, 크면 매끄럽지만 수가 적은 품종이 다수결에 묻힌다.
4. k를 바꿔도 정확도는 몇 % 안에서 움직이지만, 어떤 특징을 쓰느냐는 20%p 넘게 바꾼다.
