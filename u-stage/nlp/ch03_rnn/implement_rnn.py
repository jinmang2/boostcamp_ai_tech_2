import math
import numbers
from typing import Optional

import torch
import torch.nn as nn


"""
Recurrent Neural Networks PyTorch Implementation
------------------------------------------------
본 파일은 torch.nn.modules.rnn.py에 있는 파일을 _VF없이 RNN을 구현합니다.
"""


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
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        # input.shape == (batch_size, input_size)
        # weight_ih.shape == (hidden_size, input_size)
        # hx.shape == (batch_size, hidden_size)
        # weight_hh.shape == (hidden_size, hidden_size)
        # wx.shape = (batch_size, hidden_size)
        wx = input @ self.weight_ih.T + hx @ self.weight_hh.T
        # bias_ih.shape == (hidden_size,)
        # bias_hh.shape == (hidden_size,)
        # f_before_act.shape == (batch_size, hidden_size)
        f_before_act = wx + self.bias_ih + self.bias_hh if self.bias else wx
        next_hx = self.act(f_before_act)
        return next_hx # (batch_size, hidden_size)

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
    """
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional']
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

        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
