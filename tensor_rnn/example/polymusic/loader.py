import numpy as np
import pickle
from scipy.io import loadmat

LIST_DATASET = ['jsb', 'musedata', 'nottingham', 'pianomidi']

def parse_midi_pkl(path) :
    OFFSET = 21
    MAXDIM = 88
    data = pickle.load(open(path, 'rb'))
    def convert_int_to_onehot(items) :
        mat_onehot = np.zeros((len(items), MAXDIM), dtype='float32')
        for ii, item in enumerate(items) :
            item = [jj-OFFSET for jj in item]
            mat_onehot[ii, item] = 1.0
        return mat_onehot
    mat = {'train':[convert_int_to_onehot(item) for item in data['train']],
           'valid':[convert_int_to_onehot(item) for item in data['valid']],
           'test':[convert_int_to_onehot(item) for item in data['test']]}
    return mat

def load_dataset_pickle(dataset) :
    assert dataset in LIST_DATASET 
    if dataset == 'jsb' :
        mat = parse_midi_pkl('data/JSB Chorales.pickle')
    elif dataset == 'musedata' :
        mat = parse_midi_pkl('data/MuseData.pickle')
    elif dataset == 'nottingham' :
        mat = parse_midi_pkl('data/Nottingham.pickle')
    elif dataset == 'pianomidi' :
        mat = parse_midi_pkl('data/Piano-midi.de.pickle')
    train_all = mat['train']
    val_all = mat['valid']
    test_all = mat['test']
    return train_all, val_all, test_all

def load_dataset(dataset) :
    assert dataset in LIST_DATASET
    if dataset == 'jsb' :
        mat = loadmat('data/JSB_Chorales.mat')
    elif dataset == 'musedata' :
        mat = loadmat('data/MuseData.mat')
    elif dataset == 'nottingham' :
        mat = loadmat('data/Nottingham.mat')
    elif dataset == 'pianomidi' :
        mat = loadmat('data/Piano_midi.mat')
    else :
        raise ValueError('dataset not available')
    train_all = mat['traindata'][0]
    val_all = mat['validdata'][0]
    test_all = mat['testdata'][0]
    return train_all, val_all, test_all

def batch_data(list_data) :
    batch = len(list_data)
    seq_len = [len(x)-1 for x in list_data]
    ndim = list_data[0].shape[1]
    input = np.zeros((batch, max(seq_len), ndim), dtype='float32')
    target = np.zeros_like(input)
    mask = np.zeros((batch, max(seq_len)), dtype='float32')
    for ii in range(batch) :
        input[ii, 0:seq_len[ii]] = list_data[ii][0:-1]
        target[ii, 0:seq_len[ii]] = list_data[ii][1:]
        mask[ii, 0:seq_len[ii]] = 1 
    return input, target, mask


# TODO #
def acc_polymusic(pred, label, mask) :
    """ Ref : http://web.eecs.umich.edu/~honglak/ismir2011-PolyphonicTranscription.pdf
    ACC = TP/(FP+FN+TP)
    TP = number of note correctly predicted
    FP = number of note-off predicted as note-on
    FN = number of note-on predicted as note-off
    """
    pred = np.round(pred) #sigmoid threshold 0.5

    TP = np.float((np.logical_and(pred==1, label==1) * mask[:, :, np.newaxis]).sum())
    FP = np.float((np.logical_and(pred==1, label==0) * mask[:, :, np.newaxis]).sum())
    FN = np.float((np.logical_and(pred==0, label==1) * mask[:, :, np.newaxis]).sum())

    TN = np.float((np.logical_and(pred==0, label==0) * mask[:, :, np.newaxis]).sum())
    denom = TP+FP+FN
    nom = TP
    return nom, denom # nominator, denominator
