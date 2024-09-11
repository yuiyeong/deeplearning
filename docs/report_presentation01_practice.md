# MNIST 데이터셋을 사용한 실습

- 이 문서는 [노트북](https://colab.research.google.com/drive/1dDrNBFxOdmVf9a7_Tw2tvlQMv7Nkyvka)의 내용에서 CUDA 대신에 MPS(Metal Performance Shader) 를 사용해서 실습해본 내용을 정리한 문서입니다.
- 이 [노트북](../presentation01_mnist_ex.ipynb)으로 실습을 진행했습니다.

## Epoch 이 50번 일 때, Batch Size 에 따른 학습

| Train | Batch Size | Learning Rate | 첫 Epoch 의 Loss 합 | 최종 Epoch 의 Loss 합 | Test Set 에 대한 평가의 평균 Loss |
|-------|------------|---------------|------------------|-------------------|---------------------------|
| 1     | 64         | 0.001         | 4647.29          | 247.11            | 0.5407                    |
| 2     | 128        | 0.001         | 2904.45          | 199.61            | 0.6239                    |
| 3     | 256        | 0.001         | 1929.81          | 158.67            | 0.7829                    |

## Epoch 이 100번일 때, Batch Size 와 lr 에 따른 학습

| Train | Batch Size | Learning Rate | 첫 Epoch 의 Loss 합 | 마지막 Epoch 의 Loss 합 | Test Set 에 대한 평가의 평균 Loss |
|-------|------------|---------------|------------------|--------------------|---------------------------|
| 1     | 64         | 0.001         | 4656.53          | 110.12             | 0.5549                    |
| 2     | 128        | 0.001         | 3150.85          | 116.50             | 0.5217                    |
| 3     | 256        | 0.001         | 2076.81          | 100.47             | 0.6320                    |
| 4     | 64         | 0.01          | 2085.68          | 4.42               | 0.3905                    |
| 5     | 128        | 0.01          | 1382.98          | 6.93               | 0.3971                    |
| 6     | 256        | 0.01          | 904.22           | 9.35               | 0.4158                    |

## 결론

### Epoch 수의 영향
- Epoch 수를 50에서 100으로 늘리면 훈련 Loss 가 더 낮아집니다.
- 그러나 Test Set 에 대한 평균 Loss 는 큰 변화가 없거나 오히려 약간 증가하는 경우도 있습니다.
- 이는 과적합(overfitting)의 가능성을 시사합니다.

### Batch Size 의 영향
- 큰 Batch Size(256)는 초기 Loss 를 낮추는 데 효과적이지만, Test Set 에 대한 성능은 오히려 작은 Batch Size 에서 더 좋은 경향을 보입니다.
- Batch Size 가 증가할수록 최종 훈련 Loss 는 감소하지만, Test Set 에 대한 평균 Loss 는 증가하는 경향이 있습니다. 
- 이는 일반화 성능의 저하를 나타낼 수 있습니다.

### Learning Rate 의 영향
- 높은 Learning Rate(0.01)는 훈련 Loss 를 크게 감소시키지만, Test Set 에 대한 성능도 개선됩니다.
- Learning Rate 0.01 에서 Batch Size 에 관계없이 Test Set 성능이 일관되게 향상되었습니다.

### 최적의 조합
- 위 표의 결과에 대해서만 보자면, Test Set 에 대한 평가시 성능을 기준으로 볼 때, Batch Size 64, Learning Rate 0.01, 100 Epochs 의 조합(Train 4)이 가장 좋은 결과(0.3905)를 보여줍니다.
