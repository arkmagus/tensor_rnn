import subprocess
from sklearn.model_selection import ParameterGrid

_cmd = 'CUDA_VISIBLE_DEVICES=0 python -u poly_allrnn.py --data {data} --inmodes {inmodes} --outmodes {outmodes} --rnntype {rnntype} --lr {lr} --do {do} --epochs {epochs} | tee {log}'

hparams = [{
        'data': ['jsb', 'nottingham'],
        'inmodes': [256],
        'outmodes': [512],
        'rnntype': ['gru'],
        'lr': [2.5e-3],
        'do':[0.3],
        'epochs':[100]
        }]

if __name__ == '__main__' :
    list_param = list(ParameterGrid(hparams))
    for item in list_param :
        _log = 'log/{data}-{rnntype}-inmodes_{inmodes}-outmodes_{outmodes}-lr_{lr}-do_{do}-ep_{epochs}.log'.format(data=item['data'], rnntype=item['rnntype'], inmodes=item['inmodes'], outmodes=item['outmodes'], lr=item['lr'], do=item['do'], epochs=item['epochs'])
        subprocess.run(_cmd.format(log=_log, **item), shell=True)
        pass
    pass
