import torch as tc
from Library.DataFun import load_mnist
from Algorithms.MPS_algo import GMPS_classification


dataset = 'mnist'
category = [0, 1]
para = {
    'lr': 0.1,
    'sweepTime': 5,
    'feature_map': 'cossin',
    'dtype': tc.float64}

paraMPS = {
        'd': 2,
        'chi': 3}

train_dataset, test_dataset = load_mnist(
    dataset, process={'classes': category})
GMPS_classification(train_dataset, test_dataset,
                    para=para, paraMPS=paraMPS)

