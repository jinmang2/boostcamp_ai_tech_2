import math
import numbers
from typing import Optional, Tuple, Sequence, Union, Any

import torch
import torch.nn as nn


_TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]


"""
@TODO
- check code

(h_torch, c_torch) = _VF.lstm_cell(
    x, hx,
    weight_ih, weight_hh,
    bias_ih, bias_hh
)

h_torch = _VF.gru_cell(
    x, hx,
    weight_ih, weight_hh,
    bias_ih, bias_hh
)
"""


class LSTMCell(nn.Module):
    """
    LSTM Cell class
    torch.nn.modules.rnn.LSTMCell 대비 훨씬 느림
    atol=1e-8, rtol=1e-4 안에서 기존 LSTMCell의 연산과 동일
    학습 목적으로 보길 바랍니다!
    """
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # Weight와 Bias 선언!
        # projected version이 아니라 전부 따로 구현!
        # parameter 초기화 부분을 제외하면 `nn.Linear`와 동일함
        # torch의 Linear는 weight를 kaiming uniform,
        # bias를 1 / math.sqrt(fan_in)의 bound를 가지는 단순 uniform으로 초기화
        # rnn의 초기화는 아래 `reset_parameters` 메서드를 참고.
        num_chunks = 4
        self.weight_ih = nn.Parameter(torch.empty((num_chunks * hidden_size, input_size), **factory_kwargs))
        self.weight_hh = nn.Parameter(torch.empty((num_chunks * hidden_size, hidden_size), **factory_kwargs))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(num_chunks * hidden_size, **factory_kwargs))
            self.bias_hh = nn.Parameter(torch.empty(num_chunks * hidden_size, **factory_kwargs))
        else:
            # bias를 사용하지 않더라도 접근할 수 있도록 등록
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        # 파라미터 초기화
        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        .. math::
            \begin{array}{ll} \\
                i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
                f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
                g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
                o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
                c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
                h_t = o_t \odot \tanh(c_t) \\
            \end{array}

        where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
        state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
        is the hidden state of the layer at time `t-1` or the initial hidden
        state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
        :math:`o_t` are the input, forget, cell, and output gates, respectively.
        :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.
        """
        # initial hidden state가 없을 경우 zeros로 초기화
        # direction, layer별 전부 초기 은닉 상태가 있어야 한다.
        if hidden is None:
            zeros = torch.zeros(input.size(0), self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hidden = (zeros, zeros)
        i = torch.sigmoid(self.gate_wo_act(x, hx[0], 0)) # input gate
        f = torch.sigmoid(self.gate_wo_act(x, hx[0], 1)) # forget gate
        g = torch.tanh(self.gate_wo_act(x, hx[0], 2)) # gate gate
        o = torch.sigmoid(self.gate_wo_act(x, hx[0], 3)) # output gate
        next_cell = f * hx[1] + i * g
        next_hidden = o * torch.tanh(c)
        return (next_hidden, next_cell)

    def gate_wo_act(self, x: torch.Tensor, hx: torch.Tensor, i: int) -> torch.Tensor:
        w_ih = self.weight_ih[i*self.hidden_size:(i+1)*hidden_size, :]
        w_hh = self.weight_hh[i*self.hidden_size:(i+1)*hidden_size, :]
        b_ih, b_hh = torch.zeros(2, device=x.device, dtype=x.dtype)
        if self.bias:
            b_ih = self.bias_ih[i*self.hidden_size:(i+1)*hidden_size]
            b_hh = self.bias_hh[i*self.hidden_size:(i+1)*hidden_size]
        return x @ w_ih.T + hx @ w_hh.T + b_ih + b_hh

    @torch.no_grad()
    def backward(self, grad_outputs: _TensorOrTensors):
        """
        Elman LSTMCell의 backward pass
        torch.autograd.Function없이 naive하게 구현
        """
        return None

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)


class GRUCell(nn.Module):
    """
    GRU Cell class
    torch.nn.modules.rnn.GRUCell 대비 훨씬 느림
    atol=1e-8, rtol=1e-4 안에서 기존 LSTMCell의 연산과 동일
    학습 목적으로 보길 바랍니다!
    """
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # Weight와 Bias 선언!
        # projected version이 아니라 전부 따로 구현!
        # parameter 초기화 부분을 제외하면 `nn.Linear`와 동일함
        # torch의 Linear는 weight를 kaiming uniform,
        # bias를 1 / math.sqrt(fan_in)의 bound를 가지는 단순 uniform으로 초기화
        # rnn의 초기화는 아래 `reset_parameters` 메서드를 참고.
        num_chunks = 3
        self.weight_ih = nn.Parameter(torch.empty((num_chunks * hidden_size, input_size), **factory_kwargs))
        self.weight_hh = nn.Parameter(torch.empty((num_chunks * hidden_size, hidden_size), **factory_kwargs))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(num_chunks * hidden_size, **factory_kwargs))
            self.bias_hh = nn.Parameter(torch.empty(num_chunks * hidden_size, **factory_kwargs))
        else:
            # bias를 사용하지 않더라도 접근할 수 있도록 등록
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        # 파라미터 초기화
        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        .. math::
            \begin{array}{ll}
                r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
                z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
                n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
                h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
            \end{array}

        where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
        at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
        at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
        :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
        :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.
        """
        # initial hidden state가 없을 경우 zeros로 초기화
        # direction, layer별 전부 초기 은닉 상태가 있어야 한다.
        if hidden is None:
            zeros = torch.zeros(input.size(0), self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hidden = zeros
        r = torch.sigmoid(self.gate_wo_act(x, hx, 0)) # reset gate
        z = torch.sigmoid(self.gate_wo_act(x, hx, 1)) # update gate
        n = torch.tanh(self.gate_wo_act(x, hx, 2, reset=r)) # candidate activation vector
        next_hidden = (1 - z) * n + z * hx
        return next_hidden

    def gate_wo_act(
        self,
        x: torch.Tensor,
        hx: torch.Tensor,
        i: int,
        reset: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        w_ih = self.weight_ih[i*self.hidden_size:(i+1)*hidden_size, :]
        w_hh = self.weight_hh[i*self.hidden_size:(i+1)*hidden_size, :]
        b_ih, b_hh = torch.zeros(2, device=x.device, dtype=x.dtype)
        if self.bias:
            b_ih = self.bias_ih[i*self.hidden_size:(i+1)*hidden_size]
            b_hh = self.bias_hh[i*self.hidden_size:(i+1)*hidden_size]
        x_part = x @ w_ih.T + b_ih
        h_part = hx @ w_hh.T + b_hh
        if reset is None:
            reset = torch.ones_like(h_part, dtype=h_part.dtype, device=h_part.device)
        return x_part + reset * h_part

    @torch.no_grad()
    def backward(self, grad_outputs: _TensorOrTensors):
        """
        Elman LSTMCell의 backward pass
        torch.autograd.Function없이 naive하게 구현
        """
        return None

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)
