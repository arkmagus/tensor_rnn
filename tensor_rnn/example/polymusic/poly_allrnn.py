from __future__ import print_function
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from tensor_rnn.modules import StatefulLSTMCell, StatefulGRUCell
from tensor_rnn.modules.composite import StatefulCPLSTMCell, StatefulCPGRUCell,\
        StatefulTuckerLSTMCell, StatefulTuckerGRUCell, \
        StatefulTTLSTMCell, StatefulTTGRUCell
from tensor_rnn.modules import elementwise_bce, elementwise_bce_with_logits

from tensor_rnn.utils import tensorauto, torchauto
from tensor_rnn.utils.data_util import iter_minibatches

from loader import load_dataset_pickle, batch_data, LIST_DATASET, acc_polymusic

# rename
load_dataset = load_dataset_pickle

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--data', type=str, choices=LIST_DATASET, help='dataset {}'.format(str(LIST_DATASET)))
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--inmodes', type=int, nargs='+', default=None)
parser.add_argument('--outmodes', type=int, nargs='+', default=None)
parser.add_argument('--ranks', type=int, nargs='+', default=None)
parser.add_argument('--order', type=int, default=5, help='order for CP decomposition')
parser.add_argument('--clip', type=float, default=5, help='clip grad norm')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--do', type=float, default=0.2, help='dropout rnn layer')
parser.add_argument('--rnntype', type=str, choices=['gru', 'lstm', 
    'ttgru', 'ttlstm', 
    'tuckergru', 'tuckerlstm', 
    'cpgru', 'cplstm'])
parser.add_argument('--opt', type=str, default='Adam')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_data, val_data, test_data = load_dataset(args.data)

in_modes = args.inmodes
out_modes = args.outmodes
in_sizes = int(np.prod(in_modes))
out_sizes = int(np.prod(out_modes))
order = args.order
ranks = args.ranks
rnn_type = args.rnntype
nlayers = args.nlayers
dropout = args.do

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.prenet = nn.Linear(88, in_sizes)
        self.nlayers = nlayers
        self.rnn = nn.ModuleList()
        for ii in range(nlayers) :
            if rnn_type == 'gru' :
                self.rnn.append(StatefulGRUCell(in_sizes if ii == 0 else out_sizes, out_sizes))
            elif rnn_type == 'lstm' :
                self.rnn.append(StatefulLSTMCell(in_sizes if ii == 0 else out_sizes, out_sizes))
            elif rnn_type == 'ttlstm' :
                self.rnn.append(StatefulTTLSTMCell(in_modes if ii == 0 else out_modes, out_modes, ranks))
            elif rnn_type == 'ttgru' :
                self.rnn.append(StatefulTTGRUCell(in_modes if ii == 0 else out_modes, out_modes, ranks))
            elif rnn_type == 'cplstm' :
                self.rnn.append(StatefulCPLSTMCell(in_modes if ii == 0 else out_modes, out_modes, order))
            elif rnn_type == 'cpgru' :
                self.rnn.append(StatefulCPGRUCell(in_modes if ii == 0 else out_modes, out_modes, order))
            elif rnn_type == 'tuckerlstm' :
                self.rnn.append(StatefulTuckerLSTMCell(in_modes if ii == 0 else out_modes, out_modes, ranks))
            elif rnn_type == 'tuckergru' :
                self.rnn.append(StatefulTuckerGRUCell(in_modes if ii == 0 else out_modes, out_modes, ranks))
            else :
                raise ValueError()
        self.postnet = nn.Linear(out_sizes, 88)
   
    def reset(self) :
        for rnn in self.rnn :
            rnn.reset()

    def forward(self, x):
        # x = [batch, max_seq_len, 88] #
        batch, max_seq_len, _ = x.shape
        res = F.leaky_relu(self.prenet(x.view(-1, 88)).view(batch, max_seq_len, -1), 0.1)
        list_res = []
        for ii in range(max_seq_len) : # seq_len #
            hidden = res[:, ii].contiguous()
            for jj in range(len(self.rnn)) :
                hidden = self.rnn[jj](hidden)
                if isinstance(hidden, (list, tuple)) :
                    hidden = hidden[0]
                if dropout > 0 :
                    hidden = F.dropout(hidden, p=dropout, training=self.training)
            list_res.append(hidden)
        res = torch.stack(list_res, dim=1)
        res = self.postnet(res.view(batch*max_seq_len, -1)).view(batch, max_seq_len, -1) # use last h_t #
        # res = F.sigmoid(res)
        return res

model = Net()
if args.cuda:
    model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = getattr(torch.optim, args.opt)(model.parameters(), lr=args.lr)

total_params = sum([np.prod(x.size()) for x in model.rnn.parameters()])
for rnn in model.rnn :
    # minus 1 bias #
    if isinstance(rnn, (StatefulGRUCell, StatefulLSTMCell)) :
        total_params -= np.prod(rnn.bias_hh.size())
    else :
        total_params -= np.prod(rnn.weight_hh.bias.size())
print(vars(args))
print('RNN parameters: {}'.format(total_params))

def train(epoch, data):
    model.train()
    data_size = len(data)
    train_loss = 0
    acc_nom = 0
    acc_denom = 0
    count = 0
    for rr in iter_minibatches(data_size, args.batch_size, shuffle=True, pad=False) :
        curr_input, curr_target, curr_mask = batch_data([data[rrii] for rrii in rr])
        curr_input = Variable(tensorauto(model, torch.from_numpy(curr_input)))
        curr_target = Variable(tensorauto(model, torch.from_numpy(curr_target)))
        curr_mask = Variable(tensorauto(model, torch.from_numpy(curr_mask)))
        curr_count = curr_mask.data.sum()
        model.reset()
        optimizer.zero_grad()
        output = model(curr_input)
        loss = elementwise_bce_with_logits(output, curr_target) * curr_mask.unsqueeze(-1)
        loss = loss.sum() / curr_count
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss.data.sum() * curr_count
        curr_acc_nom, curr_acc_denom = acc_polymusic(F.sigmoid(output).data.cpu().numpy(), curr_target.data.cpu().numpy(), curr_mask.data.cpu().numpy())
        acc_nom += curr_acc_nom
        acc_denom += curr_acc_denom
        count += curr_count
        pass

    train_loss /= count
    acc = acc_nom / acc_denom
    return train_loss, acc * 100

def test(data):
    model.eval()
    model.reset()
    data_size = len(data)
    test_loss = 0
    acc_nom = 0
    acc_denom = 0
    count = 0

    for rr in iter_minibatches(data_size, args.batch_size, shuffle=False, pad=False) :
        curr_input, curr_target, curr_mask = batch_data([data[rrii] for rrii in rr])
        curr_input = Variable(tensorauto(model, torch.from_numpy(curr_input)))
        curr_target = Variable(tensorauto(model, torch.from_numpy(curr_target)))
        curr_mask = Variable(tensorauto(model, torch.from_numpy(curr_mask)))
        curr_count = curr_mask.data.sum()
        model.reset()
        output = model(curr_input)
        loss = elementwise_bce_with_logits(output, curr_target) * curr_mask.unsqueeze(-1)
        loss = loss.sum() / curr_count
        test_loss += loss.data.sum() * curr_count
        curr_acc_nom, curr_acc_denom = acc_polymusic(F.sigmoid(output).data.cpu().numpy(), curr_target.data.cpu().numpy(), curr_mask.data.cpu().numpy())
        acc_nom += curr_acc_nom
        acc_denom += curr_acc_denom
        count += curr_count

    test_loss /= count
    acc = acc_nom / acc_denom
    return test_loss, acc * 100

INF = 2**32
best_val_loss, best_val_loss_idx = INF, 0
best_val_acc, best_val_acc_idx = -INF, 0

hist_loss = {'train':[], 'val':[], 'test':[]}
hist_acc = {'train':[], 'val':[], 'test':[]}

for epoch in range(1, args.epochs + 1):
    start = time.time()
    train_loss, train_acc = train(epoch, train_data)
    end = time.time() - start
    print('Epoch {} -- time {:.1f} s'.format(epoch, end))
    print('\tTrain set: loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc))
    val_loss, val_acc = test(val_data)
    print('\tVal set: loss: {:.4f}, acc: {:.2f}'.format(val_loss, val_acc))
    test_loss, test_acc = test(test_data)
    print('\tTest set: loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc))

    hist_loss['train'].append(train_loss)
    hist_loss['val'].append(val_loss)
    hist_loss['test'].append(test_loss)
    hist_acc['train'].append(train_acc)
    hist_acc['val'].append(val_acc)
    hist_acc['test'].append(test_acc)

    if best_val_loss > val_loss :
        best_val_loss = val_loss
        best_val_loss_idx = epoch-1
    if best_val_acc < val_acc :
        best_val_acc = val_acc
        best_val_acc_idx = epoch-1

print('Best val loss: {:.4f}, acc: {:.2f}'.format(hist_loss['val'][best_val_loss_idx], hist_acc['val'][best_val_acc_idx]))
print('Best test loss: {:.4f}, acc: {:.2f}'.format(hist_loss['test'][best_val_loss_idx], hist_acc['test'][best_val_acc_idx]))
