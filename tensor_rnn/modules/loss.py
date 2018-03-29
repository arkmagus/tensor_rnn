import torch
from torch.nn import Module

def elementwise_bce(input, target) :
    return ElementwiseBCE()(input, target)

class ElementwiseBCEWithLogits(Module) :
    def __init__(self) :
        super().__init__()
        pass

    def forward(self, input, target) :
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        return loss

def elementwise_bce_with_logits(input, target) :
    return ElementwiseBCEWithLogits()(input, target)
