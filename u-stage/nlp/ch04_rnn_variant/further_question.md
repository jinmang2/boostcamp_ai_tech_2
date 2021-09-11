# Further Question

## LSTM and GRU
- BPTT 이외에 RNN/LSTM/GRU의 구조를 유지하면서 gradient vanishing/exploding 문제를 완화할 수 있는 방법이 있을까요?
- RNN/LSTM/GRU 기반의 Language Model에서 초반 time step의 정보를 전달하기 어려운 점을 완화할 수 있는 방법이 있을까요?

### BPTT 이외에 Gradient Vanishing / Exploding 문제를 완화시킬 수 있는 방법
[On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.pdf)에서 그 답을 찾았음.
- Gradient Clipping
- `Regularization` L1 or L2 Penalty on $W_{hh}$
- `Teacher Forcing`
- Use Hessian-Free Optimizer in conjunction with structural damping
    - [Learning Recurrent Neural Networks with Hessian-Free Optimization](https://icml.cc/Conferences/2011/papers/532_icmlpaper.pdf)
- $W_{hh}$의 diagonal term을 1로, off-diagonal term은 small random value로 초기화
    - [A Simple Way to Initialize Recurrent Networks of Rectified Linear Units](https://arxiv.org/abs/1504.00941)
- Leaky Integration units or Recurrent Highway Network
    - a * x + (1-a) * f(x)

⇒  정리  :  **구조적인 변화** (lstm, gru) / **가중치에 직접 변화를 가하는 형식** (regularization, clipping, diagonal term)  / **학습 방식에서의 차이** (teacher forcing , *hessian-free optimizer* , leaky integration ...)

### 초기 Time Step의 정보를 전달하기 어려운 점을 완화할 수 있는 방법
Attention 모듈을 이용하면 출력 시퀀스가 입력 문장에서 참조하는 주요 부분에 가중치 파라미터를 두고 Loss 역전파가 추가로 이루어질 수 있도록 만들어서 초반 time step의 정보 전달 문제를 완화하는 방법이 될 수 있을것 같습니다.

5강에서 교수님께서 살짝 언급해주셨던게 앞부분의 정보를 잘 전달할 수 없으니까 아예 입력문장을 뒤집어서 뒤쪽부터 넣어주고 앞부분의 정보가 더 많이 포함되어 있도록 하는 방법도 사용할 수 있다고 하셨는데, Bidirectional 네트워크를 사용해서 순방향의 입력으로 만들어진 Context vector 와 역방향의 입력으로 만들어진 Context vector를 모두 이용하여 다음 Layer의 입력 또는 마지막 출력값으로 이용하면 역시 앞부분의 정보 전달 문제를 보완할 수 있을것 같습니다!
