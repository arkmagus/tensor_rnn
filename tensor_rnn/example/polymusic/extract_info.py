import sys
import re
INF = 2**64

files = [line.strip() for line in sys.stdin]

param = None
best_nll =  INF
best_acc = -INF
for ff in files :
    txt = open(ff).read()
    _param = re.findall('RNN parameters: ([0-9]+)', txt)
    if _param == [] :
        print('[WARN] {} crashed!'.format(ff))
        continue
    _param = int(_param[0])
    if param is not None :
        assert _param == param 
    else :
        param = _param
    _nll_acc = re.findall('Best test loss: ([0-9\.]+), acc: ([0-9\.]+)', txt)
    if _nll_acc != [] :
        _nll, _acc = _nll_acc[0]
        _nll, _acc = float(_nll), float(_acc)
    else :
        print('[WARN] {} crashed!'.format(ff))
        continue
    best_nll = min(best_nll, _nll)
    best_acc = max(best_acc, _acc)

print('PARAMS\tNLL\tACC')
print('{}\t{}\t{}'.format(param, best_nll, best_acc))
