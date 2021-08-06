# Further Question

## Deep Learning
- 분류 문제에서 softmax 함수가 사용되는 이유는?
- softmax 함수의 결과값을 분류 모델의 학습에 어떤 식으로 사용할 수 있나요?

### Q1) 분류 문제에서 softmax 함수가 사용되는 이유는?
- 출력 logits값을 binary/multi class에 맞는 확률값으로 변환시켜주기 위해서

### Q2) softmax 함수의 결과값을 분류 모델의 학습에 어떤식으로 사용할 수 있나요?
```python
import torch
import torch.nn as nn

# 학습 데이터 생성
X = torch.randn(1000, 10)
# 3가지의 multi-class golden label 생성
y = torch.LongTensor(1000,).random_(0,3)

# 간단한 모델 구축
model = nn.Sequential(
    nn.Linear(10, 3),
    nn.ReLU()
)

# 예측
logits = model(X)
# 예측 logit값을 확률로 변환
torch.softmax(logits, dim=-1)
```
```
tensor([[0.3877, 0.2861, 0.3262],
        [0.5758, 0.2315, 0.1927],
        [0.3474, 0.2580, 0.3945],
        ...,
        [0.2983, 0.4568, 0.2449],
        [0.4557, 0.2722, 0.2722],
        [0.5690, 0.2155, 0.2155]], grad_fn=<SoftmaxBackward>)
```

### 추가 공부 log-softmax vs softmax
- https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
- https://github.com/pytorch/pytorch/issues/31829
- https://discuss.pytorch.org/t/logsoftmax-vs-softmax/21386/4
- https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
