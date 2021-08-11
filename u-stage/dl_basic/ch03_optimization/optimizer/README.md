# Optimizer torch 구현체

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#optimizer-basic-class">Optimizer basic class</a>
    </li>
    <li>
      <a href="#gradient-direction">Gradient direction</a>
      <ul>
        <li><a href="#stochastic-gradient-descent">Stochastic Gradient Descent</a></li>
        <li><a href="#momentum">Momentum</a></li>
        <li><a href="#nesterov-accelerated-gradient">Nesterov Accelerated Gradient</a></li>
      </ul>
    </li>
    <li>
      <a href="#step-size">Step size</a>
      <ul>
        <li><a href="#adagrad">Adagrad</a></li>
        <li><a href="#adadelta">AdaDelta</a></li>
        <li><a href="#rmsprop">RMSprop</a></li>
      </ul>
    </li>
    <li>
      <a href="#adam-lines">Adam lines</a>
      <ul>
        <li><a href="#adam">Adam</a></li>
        <li><a href="#adamw">AdamW</a></li>
        <li><a href="#radam">RAdam</a></li>
        <li><a href="#adamp">AdamP</a></li>
      </ul>
    </li>
  </ol>
</details>

## Optimizer Basic Class
- `torch.optim.optimizer.py`

### `__init__` 생성자
- `_C`의 `log_api_usage_once`로 `python.optimizer` 로깅? 연결
- `defaults`는 attribute로 기록
- `params`가 `Iterable`이 아니라 `torch.Tensor`로 들어오면 Error 띄움
- 입력 parameter `params`를 `dict` 타입의 `param_groups`로 만듦

    ```python
    [{"params": param_groups}]
    ```
    - 따로 parameter를 아래처럼 넣어주는 것도 가능
    ```python
    [
        {"params": model.encoder.parameters(), "weight_decay": 0.01},
        {"params": model.decoder.parameters(), "weight_decay": 0.0},
    ]
    ```
- 위 param 하나하나 `.add_param_group` 메서드로 추가

### `add_param_group`
- optimizer에 model parameter를 등록하는 메서드
- 입력: dictionary
- 아래 과정을 거침
    - `params`키에 해당하는 value를 list로 묶어서 메모리에 올림
    - parameter가 `nn.Parameter`가 아니라 `torch.Tensor`이거나 `is_leaf`일 경우 에러를 반환
    - default value값을 `param_group`에 포함
    - param set을 만들어서 disjoint한지 검사 후
    - 문제가 없을 시 `param_group`을 `param_groups`에 추가

### `__getstate__` and `__setstate__`

```python
def __getstate__(self):
    return {
        'defaults': self.defaults,
        'state': self.state,
        'param_groups': self.param_groups,
    }

def __setstate__(self, state):
    self.__dict__.update(state)
```

### `state_dict` and `load_state_dict`
- `__getstate__`과 `__setstate__`를 사용하여 아래 상태를 반환 혹은 기록

```python
{
    'state': state,
    'param_group': param_group
}
```

### `zero_grad`
- `param_groups` attribute에 있는 모든 param에 대해 loop을 돌며 gradient를 아래 처리
    - None으로 처리하거나
    - detach하고 requires_grad를 False로 만들고 값을 0로

### `step`
- 모든 optimizer는 해당 메서드를 override해야함
- 인자로 `closure`를 받음. 대부분의 optimizer엔 필요없는 option
    - 모델을 재평가하고 loss를 반환



<br/>
<div align="right">
    <b><a href="#optimizer-torch-구현체">↥ back to top</a></b>
</div>
<br/>

## Gradient direction
- 아래 세 모델은 전부 `torch.optim.sgd.SGD` class를 사용한다.
- 생성자에선 아래 사항들을 검사한다
    - `lr`이 요구되고 0.0을 포함한 양수인지 (default: required)
    - `momentum`이 0.0을 포함한 양수인지
    - `weight_decay`가 0.0을 포함한 양수인지
    - `nesterov`를 사용할 때 `momentum`이 양수이고 `dampening`이 0인지
        - `nesterov`는 momentum과 zero dampening을 요구한다고 함
- 입력 인자 `params`와 `defaults`를 부모 생성자에 넘겨주며 맞게 입력됐는지 검사한다.

### Stochastic Gradient Descent


### Momentum

### Nesterov Accelerated Gradient

## Step size

### Adagrad
- torch엔 `F`로 구현

### Adadelta

### RMSprop

## Adam lines

### Adam
- torch엔 `F`로 구현

### Adamw

### RAdam

### AdamP
