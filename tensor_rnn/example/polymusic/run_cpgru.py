import subprocess
from sklearn.model_selection import ParameterGrid

_cmd = 'CUDA_VISIBLE_DEVICES=0 python -u poly_allrnn.py --data {data} --inmodes {inmodes} --outmodes {outmodes} --order {order} --rnntype {rnntype} --lr {lr} --do {do} | tee {log}'

hparams = [{
        'data': ['jsb', 'nottingham', 'pianomidi', 'musedata'],
        'inmodes': ["4 4 4 4"],
        'outmodes': ["8 4 4 4"],
        'rnntype': ['cpgru'],
        'order':[50, 80, 110],
        'lr': [5e-3, 1e-2],
        'do':[0.2, 0.5]
        }]

if __name__ == '__main__' :
    list_param = list(ParameterGrid(hparams))
    for item in list_param :
        _log = 'log/{data}-{rnntype}-inmodes_{inmodes}-outmodes_{outmodes}-order_{order}-lr_{lr}-do_{do}.log'.format(data=item['data'], rnntype=item['rnntype'], inmodes=item['inmodes'].replace(' ', '_'), outmodes=item['outmodes'].replace(' ', '_'), lr=item['lr'], do=item['do'], order=item['order'])
        subprocess.run(_cmd.format(log=_log, **item), shell=True)
