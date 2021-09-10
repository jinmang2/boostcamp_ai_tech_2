import math
import numbers
from typing import Optional, Tuple, Sequence, Union, Any

import torch
import torch.nn as nn


"""
Recurrent Neural Networks PyTorch Implementation
================================================
본 파일은 torch.nn.modules.rnn.py에 있는 파일을 _VF없이 RNN을 구현합니다.

@TODO 21.09.11
- Affine, Embedding backward 구현
- RNNLM backward test code 작성
- RNNCell bakcward 작성
- RNN class에 LSTMCell, GRUCell 추가 가능하게 작성
"""

_TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]


class RNNCell(nn.Module):
    """
    RNN Cell class
    torch.nn.modules.rnn.RNNCell 대비 훨씬 느림
    atol=1e-8, rtol=1e-4 안에서 기존 RNNCell의 연산과 동일
    학습 목적으로 보길 바랍니다!
    """
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size: int, hidden_size: int,
                 bias: bool = True, nonlinearity: str = 'tanh',
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        assert nonlinearity in ['tanh', 'relu']
        # Elman RNN, tanh는 기본 form
        # ReLU는 vanishing/exploding gradient를 해결하기 위해 제안되었음
        # 그러나 RNN의 내부는 계속 순환하기 때문에 relu를 사용하면 발산할 수 있음
        # Yann LeCun 교수님의 Efficient BackProp에 따르면 tanh를 사용한 Backprop이
        # RNN에선 더 좋다고 함. http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        self.act = getattr(torch, nonlinearity)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # Weight와 Bias 선언!
        # projected version이 아니라 전부 따로 구현!
        # parameter 초기화 부분을 제외하면 `nn.Linear`와 동일함
        # torch의 Linear는 weight를 kaiming uniform,
        # bias를 1 / math.sqrt(fan_in)의 bound를 가지는 단순 uniform으로 초기화
        # rnn의 초기화는 아래 `reset_parameters` 메서드를 참고.
        self.weight_ih = nn.Parameter(torch.empty((hidden_size, input_size), **factory_kwargs))
        self.weight_hh = nn.Parameter(torch.empty((hidden_size, hidden_size), **factory_kwargs))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
            self.bias_hh = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            # bias를 사용하지 않더라도 접근할 수 있도록 등록
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        # 파라미터 초기화
        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        .. math::

            h' = \tanh(W_{ih} x + b_{ih} + W_{hh} h + b_{hh})

        If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.
        """
        # initial hidden state가 없을 경우 zeros로 초기화
        # direction, layer별 전부 초기 은닉 상태가 있어야 한다.
        if hidden is None:
            hidden = torch.zeros(input.size(0), self.hidden_size,
                                 dtype=input.dtype, device=input.device)
        # input.shape == (batch_size, input_size)
        # weight_ih.shape == (hidden_size, input_size)
        # hidden.shape == (batch_size, hidden_size)
        # weight_hh.shape == (hidden_size, hidden_size)
        # wx.shape = (batch_size, hidden_size)
        wx = input @ self.weight_ih.T + hidden @ self.weight_hh.T
        # bias_ih.shape == (hidden_size,)
        # bias_hh.shape == (hidden_size,)
        # f_before_act.shape == (batch_size, hidden_size)
        f_before_act = wx + self.bias_ih + self.bias_hh if self.bias else wx
        next_hx = self.act(f_before_act)
        return next_hx # (batch_size, hidden_size)

    @torch.no_grad()
    def backward(self, grad_outputs: _TensorOrTensors):
        """
        Elman RNNCell의 backward pass
        torch.autograd.Function없이 naive하게 구현
        """
        return None

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)


class RNN(nn.Module):
    """
    ELMan RNN
    torch.nn.modules.rnn.RNN 대비 훨씬 느림
    atol=1e-7, rtol=1e-3 안에서 기존 RNN의 연산과 동일
    학습 목적으로 보길 바랍니다!
    batch_first : bool = True만 고려했습니다!
    GPU 최적화보단 학습 목적이기 때문에
    flatten_parameters 메서드는 구현하지 않았습니다.
    """
    __constants__ = ['input_size', 'hidden_size', 'num_layers', 'bias',
                     'dropout', 'bidirectional']
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    dropout: float
    bidirectional: bool

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True,
                 dropout: float = 0., bidirectional: bool = False,
                 device=None, dtype=None, nonlinearity: str = 'tanh',):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        assert nonlinearity in ['tanh', 'relu']
        assert isinstance(dropout, numbers.Number) and 0 <= dropout <= 1
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))
        self.act = getattr(torch, nonlinearity)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        gate_size = hidden_size

        # RNNCell을 submodule로 할당
        # 생성되는 RNNCell은 num_layers * num_directions개 만큼!
        rnn_cells = []
        # layer로 for loop
        for layer in range(num_layers):
            # forward/backward for loop
            for direction in range(num_directions):
                # 첫 layer에선 input_size로 받고
                # 이후 layer에선 hidden_size에 direction의 수를 곱한 input이 들어간다
                # 이는 forward / backward result를 concat해서 넘겨줘서 그렇다
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                # backward pass의 경우 name scope을 달리 해주기 위해서 suffix를 붙여준다
                suffix = "_reverse" if direction == 1 else ""
                cell_name = "cell_l{}{}".format(layer, suffix)
                # RNNCell을 submodule로 할당한다.
                setattr(self, cell_name, RNNCell(layer_input_size, hidden_size))
            # 최상위 층이 아니고 dropout을 줄 경우 할당
            if dropout and layer != num_layers - 1:
                setattr(self, "dropout_l{}".format(layer), nn.Dropout(dropout))

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Elman RNN의 forward pass
        _VF없이 naive하게 구현
        """
        # input.shape == (batch_size, sequence_length, input_size)
        num_directions = 2 if self.bidirectional else 1
        # hxs.shape == (num_layers*num_directions, batch_size, hidden_size)
        if hx is None:
            hxs = torch.zeros(self.num_layers * num_directions,
                             input.size(0), self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            hxs = hx
        # hidden state는 layer별 그리고 direction별 가장 마지막의 tensor만 기록
        # output은 최종 layer의 모든 sequence에 대한 출력 hidden state만 기록
        # 때문에 hidden만 for loop의 바깥에서 정의
        # output은 출력값을 그대로 가지고 나오면 된다.
        hidden = []
        # layer별 rnn 연산을 수행
        for layer in range(self.num_layers):
            # forward, backward의 concat된 결과를 다음 layer로 먹여줌
            # 때문에 이를 기록할 list 작성
            layer_output = []
            # forward, backward pass
            for direction in range(num_directions):
                # 현재 layer 층 그리고 forward vs backward에 맞는 hx 반환
                hx = hxs[layer * num_directions + direction]
                # RNNCell들의 Name Scope는 생성자 참고
                suffix = "_reverse" if direction == 1 else ""
                rnn_cell = getattr(self, "cell_l{}{}".format(layer, suffix))
                # 각 time step별 RNN Cell 연산을 기록할 list
                cell_outputs = []
                # Time step별 Cell 연산 실시
                for step in range(input.size(1)):
                    # forward일 경우 0부터 T-1로 slicing, backward일 경우 -1부터 -T로 slicing
                    step = -(1 + step) if direction == 1 else step
                    # tanh(x @ W_ih.T + hx @ W_hh.T + bias)
                    # input.shape == (batch_size, seq_len, layer_input_size)
                    # where if layer == 0: layer_input_size = input_size
                    #       else: layer_input_size = hidden_size * num_directions
                    hx = rnn_cell(input[:, step, :], hx)
                    # hx.unsqueeze(1).shape == (batch_size, 1, hidden_size)
                    cell_outputs.append(hx.unsqueeze(1))
                # hx.unsqueeze(0).shape == (1, batch_size, hidden_size)
                hidden.append(hx.unsqueeze(0))
                # Concatenate forward and backward path
                # torch.cat(cell_outputs, dim=1).shape == (batch_size, seq_len, hidden_size)
                if direction == 1:
                    cell_outputs = cell_outputs[::-1]
                layer_output.append(torch.cat(cell_outputs, dim=1))
            if layer != self.num_layers - 1:
                # new input.shape == (batch_size, seq_len, hidden_size * num_directions)
                input = torch.cat(layer_output, dim=-1)
                # 최상위 layer가 아니고 dropout option이 켜져있으면 dropout 실시
                if self.dropout:
                    input = getattr(self, "dropout_l{}".format(layer))(input)
            else:
                # output.shape == (batch_size, seq_len, hidden_size * num_directions)
                output = torch.cat(layer_output, dim=-1)
                # hidden.shape == (num_layers * num_directions, batch_size, hidden_size)
                hidden = torch.cat(hidden, dim=0)

        return output, hidden

    @torch.no_grad()
    def backward(self, grad_outputs: _TensorOrTensors) -> _TensorOrTensors:
        """
        Elman RNN의 backward pass
        torch.autograd.Function없이 naive하게 구현

        - many to many problem (language model)
        """
        return None

    def reset_parameters(self):
        for child in self.childrens:
            child.reset_parameters()


class LogSoftmax(torch.autograd.Function):
    """
    Calculate Log Softmax
    - 오답에 대해 더 큰 loss를 발생시킴
    - Numerical Stability
    LINK
    - https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
    - https://discuss.pytorch.org/t/logsoftmax-vs-softmax/21386/4
    - https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
    - https://github.com/pytorch/pytorch/issues/31829
    """

    @staticmethod
    def forward(ctx: Any, tensor: Any, dim: int = -1) -> Any:
        # overflow 대책
        # softmax(x) = softmax(x+c)
        # 즉, backward에 영향 X
        c = torch.amax(tensor, dim=dim, keepdims=True)
        s = tensor - c
        # Calculate softmax
        nominator = torch.exp(s)
        denominator = nominator.sum(dim=dim, keepdims=True)
        probs = nominator / denominator
        # Calculate log
        log_probs = torch.log(probs)
        ctx.save_for_backward(probs, torch.tensor(dim))
        return log_probs

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SoftMax.cpp#L219
        probs, dim, = ctx.saved_tensors
        grad_outputs -= probs * grad_outputs.sum(dim=dim.item(), keepdims=True)
        return grad_outputs, None


class NegativeLogLikelihoodLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, y_pred: Any, y: Any) -> Any:
        bsz, n_classes = torch.tensor(y_pred.size())
        ctx.save_for_backward(bsz, n_classes, y)
        # (1) Calculate Log Likelihood
        log_likelihood = y_pred[torch.arange(bsz), y]
        # (2) Calculate Negative Log Likelihood
        nll = -log_likelihood
        # (3) Calculate Loss
        return torch.mean(nll)

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        bsz, n_classes, y, = ctx.saved_tensors
        # (3) backward mean function
        mean_grad = grad_outputs.expand(bsz) / bsz
        # (2) backward negative
        negative_mean_grad = -mean_grad
        # (1) backward log likelihood (indexing)
        ll_grad = torch.zeros(bsz, n_classes)
        ll_grad[torch.arange(bsz), y] = 1.
        # 틀린 index만 update됩니다
        grad_outputs = torch.diag(negative_mean_grad) @ ll_grad
        return grad_outputs, None


log_softmax = LogSoftmax.apply
nll_loss = NegativeLogLikelihoodLoss.apply


class CrossEntropyLoss(nn.Module):
    """
    Naive Cross Entropy Loss
    - ignore index 구현 X
    - high dimension 구현 X
    - weighted loss 구현 X

    Cross Entropy의 계산 과정 및 역전파에 대한 정확한 이해를 위한 class
    """

    def forward(self, y_pred, y):
        log_probs = log_softmax(y_pred, dim=-1)
        ce_loss = nll_loss(log_probs, y)
        probs = torch.exp(log_probs) / log_probs.size(0)
        self.save_for_backward(probs, y, y_pred.size(-1))
        return ce_loss

    def save_for_backward(self, *args):
        self.saved_tensors = args

    @torch.no_grad()
    def backward(self, grad_outputs: _TensorOrTensors) -> _TensorOrTensors:
        probs, y, num_classes, = self.saved_tensors
        ce_grad = probs - torch.nn.functional.one_hot(y, num_classes=num_classes)
        return grad_outputs * ce_grad


class RNNLM(nn.Module):
    """
    RNN Language Model
    """
    vocab_size: int
    input_size: int
    hidden_size: int
    bidirectional: bool

    def __init__(self, vocab_size: int,
                 input_size: int, hidden_size: int, bidirectional: bool,
                 device=None, dtype=None, **kwargs):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        num_directions = 2 if bidirectional else 1
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.rnn = RNN(input_size=input_size, hidden_size=hidden_size,
                       bidirectional=bidirectional, **kwargs, **factory_kwargs)
        self.out_head = nn.Linear(hidden_size * num_directions, vocab_size)

    def forward(
        self,
        input: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        embeded = self.embedding(input)
        output, _ = self.rnn(embeded)
        logits = self.out_head(output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1,))

        return (logits, loss)

    @torch.no_grad()
    def backward(self, grad_outputs: _TensorOrTensors) -> _TensorOrTensors:
        return None


if __name__ == "__main__":
    # =========================================================================#
    # Settings                                                                 #
    # =========================================================================#
    batch_size = 3
    vocab_size = 30000
    input_size = 10 # is equal to embedding dimension
    hidden_size = 8
    seq_len = 7
    num_layers = 2
    num_classes = 3
    bidirectional = True
    atol = 1e-07 # absolute tolerance
    rtol = 1e-03 # relative tolerance

    # =========================================================================#
    # First, check Elman RNN forward pass                                      #
    # =========================================================================#
    my_rnn = RNN(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
    rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

    from collections import OrderedDict
    from functools import partial

    isclose = partial(torch.isclose, rtol=rtol, atol=atol)

    state_dict = rnn.state_dict()

    new_state_dict = OrderedDict()

    for p, (n, _) in zip(state_dict.values(), my_rnn.named_parameters()):
        new_state_dict[n] = p

    my_rnn.load_state_dict(new_state_dict)

    x = torch.randn(batch_size, seq_len, input_size)
    my_output, my_hidden = my_rnn(x)
    torch_output, torch_hidden = rnn(x)

    assert my_output.shape == torch_output.shape
    assert my_hidden.shape == torch_hidden.shape

    assert isclose(my_output, torch_output).all().item()
    assert isclose(my_hidden, torch_hidden).all().item()

    # =========================================================================#
    # Second, check Elman RNN backward pass                                    #
    # - check :math: \cfrac{\partial loss}{\partial W_{hh}}                    #
    # =========================================================================#
    num_directions = 2 if bidirectional else 1
    out_proj = nn.Linear(hidden_size * num_directions, num_classes)
    ground_truth = torch.LongTensor(batch_size,).random_(0, num_classes)

    my_logits = out_proj(my_output[:, -1, :])
    torch_logits = out_proj(torch_output[:, -1, :])

    loss_fct = nn.CrossEntropyLoss()
    my_loss = loss_fct(my_logits, ground_truth)
    torch_loss = loss_fct(torch_logits, ground_truth)
    # check loss
    assert isclose(my_loss, torch_loss)

    my_loss.backward()
    torch_loss.backward()

    for my_param, torch_param in zip(my_rnn.parameters(), rnn.parameters()):
        assert isclose(my_param.grad, torch_param.grad).all().item()

    # =========================================================================#
    # Third, check LogSoftmax and NegativeLogLikelihoodLoss
    # =========================================================================#
    y_pred = torch.randn(batch_size, vocab_size)
    y = torch.LongTensor(batch_size).random_(vocab_size)

    # forward check
    assert isclose(
        torch.log_softmax(y_pred, dim=-1),
        LogSoftmax.apply(y_pred, -1)
    ).all().item()

    assert isclose(
        nn.NLLLoss()(y_pred, y),
        NegativeLogLikelihoodLoss.apply(y_pred, y)
    ).all().item()

    # backward
    my_log_y_pred = LogSoftmax.apply(y_pred, -1)
    log_y_pred = torch.log_softmax(y_pred, dim=-1)

    my_log_softmax_grad = torch.autograd.grad(
        my_log_y_pred,
        y_pred,
        grad_outputs=torch.ones_like(my_log_y_pred),
        retain_graph=True)[0]
    log_softmax_grad = torch.autograd.grad(
        log_y_pred,
        y_pred,
        grad_outputs=torch.ones_like(log_y_pred),
        retain_graph=True)[0]

    assert isclose(my_log_softmax_grad, log_softmax_grad).all().item()

    my_ce_loss = NegativeLogLikelihoodLoss.apply(my_log_y_pred, y)
    ce_loss = nn.NLLLoss()(log_y_pred, y)

    my_ce_grad = torch.autograd.grad(
        my_ce_loss,
        y_pred,
        retain_graph=True)[0]
    ce_grad = torch.autograd.grad(
        loss,
        y_pred,
        retain_graph=True)[0]

    assert isclose(my_ce_grad, ce_grad).all().item()

    # =========================================================================#
    # Fourth, implement Elman RNN Language Model backward pass step by step    #
    # =========================================================================#
    rnnlm = RNNLM(
        vocab_size=vocab_size,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        bidirectional=bidirectional,
    )
    x = torch.LongTensor(batch_size, seq_len).random_(vocab_size)
    golden_truth = torch.LongTensor(batch_size,).random_(num_classes)
