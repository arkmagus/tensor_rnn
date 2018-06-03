import math
import numpy as np

import torch
from torch import tensor
from tensor_rnn.modules import candecomp

if __name__ == '__main__':
    MAIN_DEVICE = torch.device('cuda')
    M = [100, 100, 100]
    mu = [0.6, -0.3, 2,0]
    std = [1, 1, 1]
    EPOCHS = 10000
    INTERVAL = 10
    ORDER = 10
    # real data #
    true_factors = candecomp._create_candecomp_cores_unconstrained(M, order=ORDER)
    for ii, factor in enumerate(true_factors) :
        factor.data.normal_(mu[ii], std[ii])
    true_tensors = candecomp._cpcores_to_tensor(true_factors)
    true_tensors = true_tensors.detach().to(MAIN_DEVICE)
    
    # pred data #
    pred_factors = candecomp._create_candecomp_cores_unconstrained(M, order=ORDER)
    for ii, factor in enumerate(pred_factors) :
        # Eq. refer here -> Tensor Decomposition for Compressing Recurrent Neural Network
        # initialization variance is 0.5 here
        factor.data.normal_(0.0, (0.5/(ORDER**0.5))**(1.0/len(M))) 
    pred_factors.to(MAIN_DEVICE)
    opt = torch.optim.Adam(pred_factors, lr=5e-3, amsgrad=True)
    for ee in range(1, EPOCHS+1) :
        pred_tensors = candecomp._cpcores_to_tensor(pred_factors)
        loss = (true_tensors - pred_tensors).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        if ee % INTERVAL == 0 :
            print('Epoch {}: MSE {:g}'.format(ee, loss.item()))
    pass
