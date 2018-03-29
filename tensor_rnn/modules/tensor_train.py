import math
import numpy as np

import torch
from torch.nn import Module, Parameter, ParameterList
from torch.nn import functional as F
from torch.nn import init

def _create_tt_cores(in_modes, out_modes, ranks) :
    assert len(in_modes) == len(out_modes) == len(ranks)-1
    dim = len(in_modes)
    list_tt_cores = []
    for ii in range(dim) :
        list_tt_cores.append(Parameter(torch.Tensor(out_modes[ii] * ranks[ii+1], in_modes[ii] * ranks[ii])))
    weight = ParameterList(list_tt_cores)
    return weight 

def tt_dot(in_modes, out_modes, ranks, input, weight, bias=None) :
    assert len(in_modes) == len(out_modes) == len(ranks)-1
    assert input.shape[1] == np.prod(in_modes)
    res = input
    res = res.view(-1, int(np.prod(in_modes)))
    res = res.transpose(1, 0)
    res = res.contiguous()
    dim = len(in_modes)
    for ii in range(dim) :
        res = res.view(ranks[ii] * in_modes[ii], -1)
        res = torch.matmul(weight[ii], res)
        res = res.view(out_modes[ii], -1)
        res = res.transpose(1, 0)
        res = res.contiguous()
    res = res.view(-1, int(np.prod(out_modes)))

    if bias is not None :
        res += bias
    return res

class TTLinear(Module):

    def __init__(self, in_modes, out_modes, ranks, bias=True):
        super().__init__()
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.ranks = ranks
        dim = len(self.in_modes)

        assert len(self.in_modes) == len(self.out_modes) == len(self.ranks)-1
        
        self.weight = _create_tt_cores(self.in_modes, self.out_modes, self.ranks)

        if bias:
            self.bias = Parameter(torch.Tensor(int(np.prod(out_modes))))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_xavier(self) :
        for ii in range(len(self.weight)) :
            init.xavier_normal(self.weight[ii])

    def reset_normal(self) :
        CONST = ((((0.05**2)/np.prod(self.ranks)))**(1/(len(self.ranks)-1))) ** 0.5 
        for ii in range(len(self.weight)) :
            init.normal(self.weight[ii], 0, CONST)

    def reset_parameters(self) :
        self.reset_normal()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return tt_dot(self.in_modes, self.out_modes, self.ranks, input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + 'in: ' \
                + str(self.in_modes) + ' -> out:' \
            + str(self.out_modes) + ' | ' \
            + 'rank: {}'.format(str(self.ranks))

