import numpy as np

import torch
from torch import nn
from torch.nn import Module, Parameter, ParameterList
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable

from ..rnn import StatefulBaseCell
from ..candecomp import CPLinear
from ...utils.helper import torchauto, tensorauto

class StatefulCPGRUCell(StatefulBaseCell) :
    def __init__(self, in_modes, out_modes, order, bias=True, cache=True,
            compress_in=True, compress_out=True) :
        super().__init__()
        self.in_modes = in_modes
        self.out_modes = out_modes

        self.input_size = int(np.prod(in_modes))
        self.hidden_size = int(np.prod(out_modes))

        self.compress_in = compress_in
        self.compress_out = compress_out

        self.bias = bias
        self.order = order
        self.out_modes_Mx = list(out_modes)
        self.out_modes_Mx[-1] *= 3
        if compress_in :
            self.weight_ih = CPLinear(in_modes, self.out_modes_Mx, order, bias=self.bias, cache=cache)
        else :
            self.weight_ih = nn.Linear(self.input_size, self.hidden_size*3, bias=self.bias)

        if compress_out :
            self.weight_hh = CPLinear(out_modes, self.out_modes_Mx, order, bias=self.bias, cache=cache)
        else :
            self.weight_hh = nn.Linear(self.hidden_size, self.hidden_size*3, bias=self.bias)

        self.reset_parameters()
        pass

    def reset_parameters(self) :
        self.weight_hh.reset_parameters()
        self.weight_ih.reset_parameters()

    def reset(self) :
        super().reset()
        if self.compress_out :
            self.weight_hh.reset()
        if self.compress_in :
            self.weight_ih.reset()
    
    def forward(self, input) :
        batch = input.size(0)
        if self.state is None :
            h0 = Variable(torchauto(self).FloatTensor(batch, self.hidden_size).zero_())
        else :
            h0 = self.state
        pre_rih, pre_zih, pre_nih = torch.split(self.weight_ih(input), self.hidden_size, dim=1)
        pre_rhh, pre_zhh, pre_nhh = torch.split(self.weight_hh(h0), self.hidden_size, dim=1)
        r_t = F.sigmoid(pre_rih + pre_rhh)
        z_t = F.sigmoid(pre_zih + pre_zhh)
        c_t = F.tanh(pre_nih + r_t * (pre_nhh))
        h_t = (1-z_t) * c_t + (z_t * h0)
        self.state = h_t
        return h_t
    pass

class StatefulCPLSTMCell(StatefulBaseCell) :
    def __init__(self, in_modes, out_modes, order, bias=True, cache=True, 
            compress_in=True, compress_out=True) :
        super().__init__()
        self.in_modes = in_modes
        self.out_modes = out_modes

        self.input_size = int(np.prod(in_modes))
        self.hidden_size = int(np.prod(out_modes))

        self.compress_in = compress_in
        self.compress_out = compress_out

        self.bias = bias
        self.order = order
        self.out_modes_Mx = list(out_modes)
        self.out_modes_Mx[-1] *= 4 
        if compress_in :
            self.weight_ih = CPLinear(in_modes, self.out_modes_Mx, order, bias=self.bias, cache=cache)
        else :
            self.weight_ih = nn.Linear(self.hidden_size, self.hidden_size*4, bias=self.bias)

        if compress_out :
            self.weight_hh = CPLinear(out_modes, self.out_modes_Mx, order, bias=self.bias, cache=cache)
        else :
            self.weight_hh = nn.Linear(self.hidden_size, self.hidden_size*4, bias=self.bias)
        
        self.reset_parameters()
        pass

    def reset_parameters(self) :
        self.weight_hh.reset_parameters()
        self.weight_ih.reset_parameters()
    
    def reset(self) :
        super().reset()
        if self.compress_out :
            self.weight_hh.reset()
        if self.compress_in :
            self.weight_ih.reset()
    
    def forward(self, input) :
        batch = input.size(0)
        if self.state is None :
            h0 = Variable(torchauto(self).FloatTensor(batch, self.hidden_size).zero_())
            c0 = Variable(torchauto(self).FloatTensor(batch, self.hidden_size).zero_())
        else :
            h0, c0 = self.state
        pre_iih, pre_fih, pre_gih, pre_oih = torch.split(self.weight_ih(input), self.hidden_size, dim=1)
        pre_ihh, pre_fhh, pre_ghh, pre_ohh = torch.split(self.weight_hh(h0), self.hidden_size, dim=1)
        i_t = F.sigmoid(pre_iih + pre_ihh)
        f_t = F.sigmoid(pre_fih + pre_fhh)
        o_t = F.sigmoid(pre_oih + pre_ohh)
        g_t = F.tanh(pre_gih + pre_ghh)
        c_t = f_t * c0 + i_t * g_t
        h_t = o_t * F.tanh(c_t)
        self.state = (h_t, c_t)
        return (h_t, c_t)
