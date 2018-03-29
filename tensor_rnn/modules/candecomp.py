import math
import numpy as np

import torch
from torch.nn import Module, Parameter, ParameterList
from torch.nn import functional as F
from torch.nn import init

def _create_candecomp_cores(in_modes, out_modes, order) :
    assert len(in_modes) == len(out_modes)
    assert order > 0
    list_cores = []
    modes = in_modes + out_modes # extend list
    for mm in modes :
        list_cores.append(Parameter(torch.Tensor(mm, order).zero_()))
    list_cores = ParameterList(list_cores)
    return list_cores

def _tensor_to_matrix(in_modes, out_modes, tensor) :
    return tensor.view(int(np.prod(in_modes)), int(np.prod(out_modes)))

def _cpcores_to_tensor(list_factors) :
    assert len(list_factors) > 2
    tensor_out = None
    list_tensor_shape = [list_factors[ii].shape[0] for ii in range(len(list_factors))]
    for ii in range(len(list_factors)) :
        if ii == 0 :
            tensor_out = list_factors[ii]
        else :
            t_r, t_c = tensor_out.shape
            f_r, f_c = list_factors[ii].shape
            assert t_c == f_c, "tensor core order should be same"
            tensor_out = tensor_out.view(t_r, 1, t_c) * list_factors[ii].view(1, f_r, f_c)
            tensor_out = tensor_out.view(t_r * f_r, f_c)
    # sum across all order #
    tensor_out = tensor_out.sum(-1)
    tensor_out = tensor_out.view(*list_tensor_shape)
    return tensor_out

class CPLinear(Module) :
    def __init__(self, in_modes, out_modes, order, bias=True, cache=True) :
        """
        cache: if cache is True, pre calculated W_tsr until user reset the variable
        """
        super().__init__()
        self.in_modes = in_modes
        self.out_modes = out_modes
        self.order = order
        self.cache = cache
        self._W_linear = None
        
        self.factors = _create_candecomp_cores(in_modes, out_modes, order)

        if bias :
            self.bias = Parameter(torch.Tensor(int(np.prod(out_modes))))
        else :
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) :
        CONST = (0.05 / (self.order**0.5)) ** (1.0/(len(self.in_modes)+len(self.out_modes)))
        for ii in range(len(self.factors)) :
            init.normal(self.factors[ii], 0, CONST)
            pass
        if self.bias is not None :
            self.bias.data.zero_()

    def reset(self) :
        self._W_linear = None
    
    @property
    def W_linear(self) : 
        if not self.cache :
            return _tensor_to_matrix(self.in_modes, self.out_modes, _cpcores_to_tensor(list(self.factors)))
        if self._W_linear is None :
            self._W_linear = _tensor_to_matrix(self.in_modes, self.out_modes, _cpcores_to_tensor(list(self.factors)))
        else :
            pass
        return self._W_linear

    def forward(self, input) :
        return F.linear(input, self.W_linear.t(), self.bias)
