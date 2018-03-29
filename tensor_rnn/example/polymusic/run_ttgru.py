import subprocess
from sklearn.model_selection import ParameterGrid

_cmd = 'CUDA_VISIBLE_DEVICES=0 python -u poly_allrnn.py --data {data} --inmodes {inmodes} --outmodes {outmodes} --ranks {ranks} --rnntype {rnntype} --lr {lr} --do {do} --batch-size {batchsize}  | tee {log}'

hparams = [
        {
        'data': ['pianomidi', 'musedata'],
        'inmodes': ["4 4 4 4"],
        'outmodes': ["8 4 4 4"],
        'rnntype': ['ttgru'],
        'ranks':["1 11 11 11 1"],
        'lr': [5e-3, 1e-2],
        'do':[0.2, 0.5],
        'batchsize':[8]}
        ]

if __name__ == '__main__' :
    list_param = list(ParameterGrid(hparams))
    for item in list_param :
        _log = 'log/{data}-{rnntype}-inmodes_{inmodes}-outmodes_{outmodes}-ranks_{ranks}-lr_{lr}-do_{do}-bsize_{batchsize}.log'.format(data=item['data'], rnntype=item['rnntype'], inmodes=item['inmodes'].replace(' ', '_'), outmodes=item['outmodes'].replace(' ', '_'), lr=item['lr'], do=item['do'], ranks=item['ranks'].replace(' ', '_'), batchsize=item['batchsize'])
        print('CMD : {}'.format(_cmd.format(log=_log, **item)))
        subprocess.run(_cmd.format(log=_log, **item), shell=True)
