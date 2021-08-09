# Further Question

## Multi-Layer Perceptron
- Regression Task, Classification Task, Probabilistic Task의 Loss 함수(or 클래스)는 Pytorch에서 어떻게 구현이 되어있을까요?

## Pytorch classes
- 우선 다음주 pytorch 시간에 자세히 다룰테지만, 토치의 기본 모듈은 `nn.Module`을 상속받아 구현된다.
- Loss도 동일하게 기본 구조는 아래처럼 상속받는다.
```python
import warnings

from torch.nn import functional as F
from torch.nn import _reduction as _Reduction
from torch.nn.modules.module import Module
from torch.nn.modules.distance import PairwiseDistance
from torch import Tensor
from typing import Callable, Optional


# tensorflow의 reduce_mean, reduce_sum과 동일함
class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
          self.reduction = reduction


# 단순히 weight만 추가된 loss
class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
```

- 토치 친구들은 기본 내장 모듈에 있는 모든 Loss들을 cpp 엔진에서 계산된 값을 돌려준다.
- 이번 예제에 있는 걸 확인해보자

```python
# MSE
class MSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)


# Cross Entropy Loss
class CrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean') -> None:
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

# Cross Entropy = Negative Log Likelihood(Log Softmax (input), target)
# torch.nn.functional.py
...
def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean'):
    ...
    return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
```

- 저기 위에 `F`가 뭐냐, `torch.nn.functional`이다.
- 모듈을 뒤져보면 나오지만, 결국엔 아래 둘 중 하나의 연산으로 들어가서 확인이 어렵다.
    - `torch._C._nn.nll_loss2d`
    - `from torch import _VF`
- 만약 cpp를 잘 안다면 들어가서 뜯어보면 확인이 쉽긴 하다!

## Cpp source code
- cpp를 잘 모르지만, 이번 기회에 한 번 뒤져보자.
- 여기서 찾았다.
    - https://discuss.pytorch.org/t/where-is-c-source-code/46850
    - https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Loss.cpp
- 그만 알아보도록 하자.

## `F`를 사용하지 않고 직접 구현
- 사실, 개념 파악을 위해서라면 직접 구현해보는 것이 가능하다.
- 재생산을 위해 seed는 아래처럼 고정하도록 하자.
- 간단한 실험이므로 분산 환경의 seed까진 고정하지 않는다.
```python
import random
import numpy as np
import torch


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
```

### MSELoss 구현
```python
import torch
import torch.nn as nn


# input & golden label 정의
x = torch.randn(32, 128)
y = torch.randn(32, 1)

# 1개의 hidden layer를 가지는 network 정의
network = nn.Sequential(
    nn.Linear(128, 64, bias=True),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# 예측값 얻어오기
y_pred = network(x)
```

#### 1. 기존 torch의 `MSELoss`
- torch에서 제공한 함수로 loss를 계산해보자.
```python
loss = nn.MSELoss()(y_pred, y)
loss
```
```python
tensor(1.1322, grad_fn=<MseLossBackward>)
```
- 이미 torch에서 MSE를 위한 backward 함수도 제공한다.
- `retain_graph`는 실험을 위해 grad 재생산이 잘 되는지 개인적으로 테스트하기 위해 넣었다.
- 굳이 안 넣어도 되지만, 이 경우 노드 초기화를 안해주면 다음 backward에서 에러가 발생할 수 있기 때문에 주의하도록 하자.
```python
# zero_grad test를 위해 그래프 연결 유지
# 굳이 안해줘도 되간 함!
loss.backward(retain_graph=True)
list(network.parameters())[0].grad
```
```python
tensor([[ 0.0052,  0.0063,  0.0052,  ...,  0.0081,  0.0004, -0.0005],
        [-0.0285, -0.0175,  0.0025,  ..., -0.0375, -0.0204, -0.0365],
        [-0.0274, -0.0085, -0.0343,  ..., -0.0173,  0.0118,  0.0187],
        ...,
        [ 0.0185,  0.0158, -0.0047,  ...,  0.0277,  0.0284,  0.0063],
        [-0.0017, -0.0043, -0.0316,  ..., -0.0493, -0.0128, -0.0232],
        [ 0.0282,  0.0338, -0.0016,  ...,  0.0586,  0.0120,  0.0352]])
```

#### 2. 새롭게 구현한 `MSELoss`
- 새롭게 구현한 loss는 간단하게 차원을 신경쓰지 않고 아래처럼 구현했다.
```python
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        return torch.mean((y_pred - y) ** 2)
```
- 계산해보면 완벽히 동일하다!
```python
# 위에서 계산된 gradient 초기화
network.zero_grad()
# 새롭게 loss 계산
loss = MSELoss()(y_pred, y)
loss
```
```python
tensor(1.1322, grad_fn=<MeanBackward0>)
```
- 사실, 완벽히 동일하진 않다. backward 함수가 다르다.
- 그래도 맞게 구현했다면 전파된 gradient 값도 동일해야 할 것 이다.
- 확인해보면 같은 것을 알 수 있다.

```python
loss.backward(retain_graph=True)
list(network.parameters())[0].grad
```
```python
tensor([[ 0.0052,  0.0063,  0.0052,  ...,  0.0081,  0.0004, -0.0005],
        [-0.0285, -0.0175,  0.0025,  ..., -0.0375, -0.0204, -0.0365],
        [-0.0274, -0.0085, -0.0343,  ..., -0.0173,  0.0118,  0.0187],
        ...,
        [ 0.0185,  0.0158, -0.0047,  ...,  0.0277,  0.0284,  0.0063],
        [-0.0017, -0.0043, -0.0316,  ..., -0.0493, -0.0128, -0.0232],
        [ 0.0282,  0.0338, -0.0016,  ...,  0.0586,  0.0120,  0.0352]])
```

### CrossEntropyLoss 구현
- notebook으로 진행한다면, session을 초기화하고 위의 seed로 동일하게 고정해주자.

#### 1. 기존 torch의 `CrossEntropyLoss`
```python
import torch
import torch.nn as nn


# input & golden label 정의
x = torch.randn(32, 128)
y = torch.LongTensor(32,).random_(0, 3) # 0~2의 label 생성

network = nn.Sequential(
    nn.Linear(128, 64, bias=True),
    nn.ReLU(),
    nn.Linear(64, 3)
)

y_pred = network(x)
```
- torch에서 제공한 함수로 loss를 계산해보자.
```python
loss = nn.CrossEntropyLoss()(y_pred, y)
loss
```
```python
tensor(1.0898, grad_fn=<NllLossBackward>)
```
- 이미 torch에서 Cross Entropy를 위한 backward 함수도 제공한다.
```python
loss.backward(retain_graph=True)
list(network.parameters())[0].grad
```
```python
tensor([[ 0.0081,  0.0075, -0.0024,  ...,  0.0043,  0.0005, -0.0008],
        [ 0.0299,  0.0068, -0.0086,  ..., -0.0026,  0.0013, -0.0051],
        [-0.0020, -0.0002, -0.0018,  ..., -0.0014, -0.0029, -0.0020],
        ...,
        [-0.0006,  0.0005,  0.0081,  ..., -0.0008, -0.0016,  0.0013],
        [ 0.0023,  0.0014,  0.0013,  ..., -0.0004, -0.0005, -0.0009],
        [-0.0149,  0.0042,  0.0098,  ...,  0.0071, -0.0059,  0.0099]])
```
#### 2. How to calculate Cross-Entropy?
- Cross Entropy는 log_softmax와 NLL로 계산 가능하다.
- log_softmax는 예측 logits에 씌워준다.
- 왜냐? softmax는 확률로 만들어주기 위해서!
- log는 nll을 위해서!
- 우선 개념이 맞는지 확인해보자.
```python
class CELoss1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        y_pred = torch.log_softmax(y_pred, dim=-1)
        return nn.NLLLoss()(y_pred, y)
```
- 예측값에 log_softmax를 취하고 NLL Loss를 계산한 값을 반환한다.
- 값은 아래처럼 완벽히 동일하게 반환된다.
```python
# 위에서 계산된 gradient 초기화
network.zero_grad()
# 새롭게 loss 계산
loss = CELoss1()(y_pred, y)
loss
```
```python
tensor(1.0898, grad_fn=<NllLossBackward>)
```
- gradient는 아래처럼 update된다.
```python
loss.backward(retain_graph=True)
list(network.parameters())[0].grad
```
```python
tensor([[ 0.0081,  0.0075, -0.0024,  ...,  0.0043,  0.0005, -0.0008],
        [ 0.0299,  0.0068, -0.0086,  ..., -0.0026,  0.0013, -0.0051],
        [-0.0020, -0.0002, -0.0018,  ..., -0.0014, -0.0029, -0.0020],
        ...,
        [-0.0006,  0.0005,  0.0081,  ..., -0.0008, -0.0016,  0.0013],
        [ 0.0023,  0.0014,  0.0013,  ..., -0.0004, -0.0005, -0.0009],
        [-0.0149,  0.0042,  0.0098,  ...,  0.0071, -0.0059,  0.0099]])
```
#### 3. 새롭게 구현한 `CrossEntropyLoss`
- 우선은 `log_softmax`를 구현해보자.
```python
def log_softmax(tensor, dim=-1):
    # only support 2-d tensor
    assert isinstance(dim, int) and dim in [0, 1, -1]
    sizes = (-1, 1) if dim != 0 else (1, -1)
    # overflow 방지 대책
    c = torch.max(tensor, dim=dim).values
    # calculate exponential
    v = torch.exp(tensor - c.view(*sizes))
    return torch.log(v / v.sum(dim=dim).view(*sizes))
```
- 맞게 구현했는지 확인하기 위해 `isclose` 함수를 torch를 활용하여 구현하자.
```python
def isclose(a, b, rtol=1e-05, atol=1e-08):
    isclose_tensor = torch.isclose(a, b, rtol=rtol, atol=atol)
    return torch.all(isclose_tensor).item()
```
- 맞게 구현된 것 같다.
```python
isclose(
    log_softmax(y_pred, dim=-1),
    torch.log_softmax(y_pred, dim=-1)
)
>>> True
```
- 그 다음은 `NLLLoss`를 구현해보자.
    - 개념만 확인하고 넘어갈 것이기 때문에
    - 2d말고 1d만 구현한다.
    - https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
    - https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
- torch 공홈의 수식대로 구현하면 아래와 같다.
    - reduction은 `mean`만 고려
```python
class NLLLoss(nn.Module):
    def __init__(self):
        # reduction은 항상 "mean"
        super().__init__()

    def forward(self, y_pred, y, batch_index=0):
        # input과 target shape 체크
        assert y_pred.ndim == 2 and y.ndim == 1
        # 같은 batch size인지 체크
        assert y_pred.size(batch_index) == y.size(batch_index)
        # n_classes가 맞는지 체크
        assert y_pred.size(-1) > max(y).item()
        # torch 공홈의 수식과 같은 값을 반환
        bsz = y_pred.size(batch_index)
        return torch.mean(y_pred[torch.arange(bsz), y])
```
- 이를 기반으로 Cross Entropy Loss를 구현하면 아래와 같다.
```python
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y, *args):
        nll_loss = NLLLoss(*args)
        y_pred = -log_softmax(y_pred, dim=-1)
        return nll_loss(y_pred, y)
```
- 맞게 구현되었는지 loss 값을 확인하자!
```python
# 위에서 계산된 gradient 초기화
network.zero_grad()
# 새롭게 loss 계산
loss = CrossEntropyLoss()(y_pred, y)
loss
```
```python
tensor(1.0898, grad_fn=<MeanBackward0>)
```
- gradient도 정확히 동일하다!
```python
loss.backward(retain_graph=True)
list(network.parameters())[0].grad
```
```python
tensor([[ 0.0081,  0.0075, -0.0024,  ...,  0.0043,  0.0005, -0.0008],
        [ 0.0299,  0.0068, -0.0086,  ..., -0.0026,  0.0013, -0.0051],
        [-0.0020, -0.0002, -0.0018,  ..., -0.0014, -0.0029, -0.0020],
        ...,
        [-0.0006,  0.0005,  0.0081,  ..., -0.0008, -0.0016,  0.0013],
        [ 0.0023,  0.0014,  0.0013,  ..., -0.0004, -0.0005, -0.0009],
        [-0.0149,  0.0042,  0.0098,  ...,  0.0071, -0.0059,  0.0099]])
```

## 다양한 loss function들
- 이 과정을 진행하며 정말 많은 것들을 배울텐데,
- https://neptune.ai/blog/pytorch-loss-functions
- 위 링크를 공부하며 추가 구현하는 것도 괜찮아보임!
