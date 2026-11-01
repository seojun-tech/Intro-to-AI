# Performance Evaluation (성능평가) — 혼동행렬과 평가지표

인공지능개론 7주차 실습 · 성능평가  
TP·FP·FN·TN을 입력받아 혼동행렬을 출력하고, 7가지 평가지표와 FP/FN 오류 해석을 계산하는 Python 프로그램.

> **성능평가 = 예측을 정답과 비교해 혼동행렬 네 칸을 세고, 그 칸들의 비율로 지표를 만든다.** 같은 민감도라도 클래스 비율이 바뀌면 정밀도는 크게 달라질 수 있어서, 지표 하나만 보면 안 된다.

| 파일 | 내용 |
| --- | --- |
| `binary_classification_eval.py` | 빈칸(정확도·민감도·특이도·정밀도·NPV) 완성 + TP·FP·FN·TN 사용자 입력으로 변경 |

| 지표 | 공식 |
| --- | --- |
| 정확도 Accuracy | (TP + TN) / N |
| 민감도 Sensitivity (Recall) | TP / (TP + FN) |
| 특이도 Specificity | TN / (TN + FP) |
| 정밀도 Precision (PPV) | TP / (TP + FP) |
| 음성예측도 NPV | TN / (TN + FN) |
| F1-score | 2 × Precision × Recall / (Precision + Recall) |
| 오분류율 Error Rate | (FP + FN) / N |

- 실행 결과: TP=450, FP=50, FN=50, TN=450 → 지표 6개 모두 90.00%, 오분류율 10.00%
- 강의 예제 TP=71, FP=33, FN=29, TN=304 → 정확도 85.81%, 민감도 71.00%, 특이도 90.21%, 정밀도 68.27%, NPV 91.29%, F1 69.61%
- FP는 불필요한 검사·처치로 이어지는 **비용**, FN은 질병·이상을 놓치는 **위험**
