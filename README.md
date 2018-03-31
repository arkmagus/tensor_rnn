# Tensor RNN 
An implementation of various tensor-based decomposition for NN & RNN parameters

## Quick Start
1. Install `python >= 3.0`
2. Install `pytorch >= 3.0`
3. `pip install -e .` or `python setup.py install`

### Run example scripts
1. Go to example folder
    
    `cd example/polymusic`

2. Go to data folder, download the pickled dataset and return.
    
    `cd data && ./download_data.sh && cd ..`
3. Run any example script
    
    `python run_ttgru.py`

For the usage, see the code inside `example/polymusic/poly_allrnn.py`
## Modules
* `TuckerLinear `
* `CPLinear`
* `TTLinear`
* `StatefulCPLSTMCell`
* `StatefulCPGRUCell`
* `StatefulTuckerLSTMCell`
* `StatefulTuckerGRUCell`
* `StatefulTTLSTMCell`
* `StatefulTTGRUCell`

## Reference
If you find this package is useful, please kindly cite: 
```
@article{tjandra2018tensor,
  title={Tensor Decomposition for Compressing Recurrent Neural Network},
  author={Tjandra, Andros and Sakti, Sakriani and Nakamura, Satoshi},
  journal={arXiv preprint arXiv:1802.10410},
  year={2018}
}

@inproceedings{tjandra2017compressing,
  title={Compressing recurrent neural network with tensor train},
  author={Tjandra, Andros and Sakti, Sakriani and Nakamura, Satoshi},
  booktitle={Neural Networks (IJCNN), 2017 International Joint Conference on},
  pages={4451--4458},
  year={2017},
  organization={IEEE}
}
```
