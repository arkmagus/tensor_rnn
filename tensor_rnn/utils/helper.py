import torch
from torch.autograd import Variable

def is_cuda_module(module) :
    return next(module.parameters()).is_cuda

def _auto_detect_cuda(module) :
    if isinstance(module, torch.nn.Module) :
        return is_cuda_module(module) 
    if isinstance(module, bool) :
        return module
    if isinstance(module, int) :
        return module >= 0
    if isinstance(module, torch.autograd.Variable) :
        return module.data.is_cuda
    if isinstance(module, torch.tensor._TensorBase) :
        return module.is_cuda
    raise NotImplementedError()

def torchauto(module) :
    return torch.cuda if _auto_detect_cuda(module) else torch

def tensorauto(module, tensor) :
    return tensor.cuda() if _auto_detect_cuda(module) else tensor.cpu()
