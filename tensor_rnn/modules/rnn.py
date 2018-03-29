import math
import torch

from torch.nn import Module
from torch.nn import Parameter
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from ..utils.helper import torchauto

##### WRAPPER #####
class StatefulBaseCell(Module) :
    def __init__(self) :
        super(StatefulBaseCell, self).__init__()
        self._state = None
        pass

    def reset(self) :
        self._state = None

    @property
    def state(self) :
        return self._state

    @state.setter
    def state(self, value) :
        self._state = value

class StatefulLSTMCell(StatefulBaseCell) :
    def __init__(self, input_size, hidden_size, bias=True) :
        super(StatefulLSTMCell, self).__init__()
        self.rnn_cell = nn.LSTMCell(input_size, hidden_size, bias)
        pass

    @property
    def weight_hh(self) :
        return self.rnn_cell.weight_hh.t()

    @property
    def weight_ih(self) :
        return self.rnn_cell.weight_ih.t()

    @property
    def bias_hh(self) :
        return self.rnn_cell.bias_hh

    @property
    def bias_ih(self) :
        return self.rnn_cell.bias_ih

    def forward(self, input) :
        batch = input.size(0)
        if self.state is None :
            h0 = Variable(torchauto(self).FloatTensor(batch, self.rnn_cell.hidden_size).zero_())
            c0 = Variable(torchauto(self).FloatTensor(batch, self.rnn_cell.hidden_size).zero_())
            # h0, c0 #
            self.state = (h0, c0)

        self.state = self.rnn_cell(input, self.state)
        return self.state

class StatefulGRUCell(StatefulBaseCell) :
    def __init__(self, input_size, hidden_size, bias=True) :
        super(StatefulGRUCell, self).__init__()
        self.rnn_cell = nn.GRUCell(input_size, hidden_size, bias)
        pass

    @property
    def weight_hh(self) :
        return self.rnn_cell.weight_hh.t()

    @property
    def weight_ih(self) :
        return self.rnn_cell.weight_ih.t()

    @property
    def bias_hh(self) :
        return self.rnn_cell.bias_hh

    @property
    def bias_ih(self) :
        return self.rnn_cell.bias_ih

    def forward(self, input) :
        batch = input.size(0)
        if self.state is None :
            h0 = Variable(torchauto(self).FloatTensor(batch, self.rnn_cell.hidden_size).zero_())
            # h0, c0 #
            self.state = h0

        self.state = self.rnn_cell(input, self.state)
        return self.state
###################
