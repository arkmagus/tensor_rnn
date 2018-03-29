import numpy as np

def iter_minibatches(indices, batchsize, shuffle=True, pad=False, excludes=None):
    """
    Args:
        datasize : total number of data or list of indices
        batchsize : mini-batchsize
        shuffle :
        use_padding : pad the dataset if dataset can't divided by batchsize equally

    Return :
        list of index for current epoch (randomized or not depends on shuffle)
    """
    if isinstance(indices, list) :
        indices = indices
    elif isinstance(indices, int) :
        indices = list(range(indices))
    if excludes is not None :
        indices = [x for x in indices if x not in excludes]
    if shuffle:
        np.random.shuffle(indices)

    if pad :
        indices = pad_idx(indices, batchsize)

    for ii in range(0, len(indices), batchsize):
        yield indices[ii:ii + batchsize]
    pass
