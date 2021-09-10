# 3강 Basics of Recurrent Neural Networks (RNNs)

자연어 처리 분야에서 Recurrent Neural Network(RNN)를 활용하는 다양한 방법과 이를 이용한 Language Model을 학습합니다.

RNN은 단어간 순서를 가진 문장을 표현하기 위해 자주 사용되어 왔습니다. 이러한 RNN 구조를 활용해 다양한 NLP 문제를 정의하고 학습하는 방법을 소개합니다.

Language Model은 이전에 등장한 단어를 condition으로 다음에 등장할 단어를 예측하는 모델입니다. 이전에 등장한 단어는 이전에 학습했던 다양한 neural network 알고리즘을 이용해 표현될 수 있습니다. 이번 시간에는 RNN을 이용한 character-level의 language model에 대해서 알아봅니다.

RNN을 이용한 Language Model에서 생길 수 있는 초반 time step의 정보를 전달하기 어려운 점, gradient vanishing/exploding을 해결하기 위한 방법 등에 대해 다시 한번 복습할 수 있는 시간이 됐으면 합니다.

[back to super](https://github.com/jinmang2/boostcamp_ai_tech_2/tree/main/u-stage/nlp)

- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)


## Types of RNNs
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/

### Recurrent Neural Network
- 자세한 내용은 `implement_rnn.py`에 이해한 내용을 전부 정리했다.

**One-to-one**
- Standard Neural Networks

**One-to-many**
- Image Captioning

**Many-to-one**
- Sentiment Classification

**Many-to-many**
- Machine Translation

**Many-to-many-by-step**
- Video classification on frame level

## Discussion
**RNN에서  활성화함수로 Tanh를 사용하는 이유**: RNN에서는 이전 값을 중첩으로 사용하기 때문에 normalizing을 효과를 줄 수 있는 활성화함수가 필요. ReLU를 사용할 경우 이전 값이 커짐에 따라 전체적인 출력이 발산하는 문제가 생길 수 있다! Normalize 효과를 줄 수 있는 비선형 함수로 Sigmoid와 Tanh가 있는데 sigmoid는 중심값이 0이 아닌 점과 gradient vanishing이 더 잘 일어난다는 문제 때문에 tanh 함수가 더 좋은 성능을 보임. (하지만 tanh도 gradient vanishing 문제를 완전히 벗어나지는 못함.)

1. centre value가 0이냐 0.5냐
--> hidden state를 time step별로 공유하기 때문에 open boundary가 단방향으로 shift됨
2. BackProp시 sigmoid나 tanh나 문제가 되긴 함 ;; 이건 ReLU로도 해결 불가능
때문에 Residual Learning이나 Clipping 기법이 RNN에서 해결책으로 제시되어 옴
그러나 90년대 연구에선 그나마 tanh가 1번의 이유 + BackProp시 계산 효율이 좋았다는 결과가 있음 (LeCun의 Efficient Backprop)
[https://arxiv.org/pdf/1211.5063.pdf](https://arxiv.org/pdf/1211.5063.pdf) (clipping)
[https://arxiv.org/pdf/1607.03474.pdf](https://arxiv.org/pdf/1607.03474.pdf) (residual connection)
[http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (tanh가 그나마 나음)

- sigmoid 함수의 중심값이 0이 아닌 것의 단점
    - https://reniew.github.io/12/
    - https://stats.stackexchange.com/questions/237169/why-are-non-zero-centered-activation-functions-a-problem-in-backpropagation
    - https://rohanvarma.me/inputnormalization/
    - http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf
    - https://nittaku.tistory.com/267


## Code implementation
이번 강의는 최성준 교수님 강의 시간에 기초적인 이론 및 torch code wrap-up을 했습니다.

때문에 이번주 학습 기간에는 직접 내부 구현 및 역전파에 대해 확실하게 이해해보고자 코드를 작성했습니다.

[해당 파일](https://github.com/jinmang2/boostcamp_ai_tech_2/tree/main/u-stage/nlp/ch03_rnn/implement_rnn.py)에서는 RNN과 RNNLM에 대한 forward/backward에 대한 자세한 수식이 코드로 작성되어 있습니다. 추가적으로 Cross Entropy, Affine Transform, Embedding에 대한 역전파도 `torch.autograd.Function` 객체를 활용하여 직접 구현해봤습니다.

$\begin{aligned}
\cfrac{d \mathcal{L}}{d W_{hh}}&=\cfrac{1}{T}\sum_{1\leq t_2 \leq T}\cfrac{d \mathcal{L}^{[t_2]}}{d W_{hh}}\\
&=\cfrac{1}{T}\sum_{1\leq t_2 \leq T}\sum_{1\leq t_1 \leq t_2}\cfrac{\partial \mathcal{L}^{[t_2]}}{\partial \mathcal{o}^{[t_2]}}\cfrac{\partial \mathcal{o}^{[t_2]}}{\partial \mathcal{y}^{[t_2]}}\cfrac{\partial \mathcal{y}^{[t_2]}}{\partial \mathcal{y}^{[t_1]}}\cfrac{\partial \mathcal{y}^{[t_1]}}{\partial W_{hh}}
\end{aligned}$


$\begin{aligned}
\cfrac{\partial \mathcal{y}^{[t_2]}}{\partial \mathcal{y}^{[t_1]}}:=\prod_{t_1 < t \leq t_2}{\cfrac{\partial y^{[t]}}{\partial y^{[t-1]}}}=\prod_{t_1 < t \leq t_2}{W_{hh}^{\intercal}\;\text{diag}\big[\tanh^\prime(W_{hh}y^{[t-1]})\big]}
\end{aligned}$

$\text{Let }A=\cfrac{\partial y^{[t]}}{\partial y^{[t-1]}}$
