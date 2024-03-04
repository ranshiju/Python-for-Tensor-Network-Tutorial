import torch as tc
from Library.DataFun import load_mnist
from Algorithms.MPS_algo import GMPS_train
from Library.DataFun import dataset2tensors
from Library.BasicFun import plot

dataset = 'fmnist'
category = [0]
para = {
    'lr': 0.1,
    'sweepTime': 10,
    'feature_map': 'cossin',
    'dtype': tc.float32
}

paraMPS = {
        'length': 784,
        'd': 2,
        'chi': 64
    }

train_dataset, test_dataset = load_mnist(
    dataset, process={'classes': category})
train_img, _ = dataset2tensors(train_dataset)
test_img, _ = dataset2tensors(test_dataset)
train_img = train_img.reshape(train_img.shape[0], -1)
test_img = test_img.reshape(test_img.shape[0], -1)

mps, para, info = GMPS_train(
    train_img, para=para, paraMPS=paraMPS)
nll_test = mps.evaluate_nll(test_img, average=True)


plot(tc.arange(len(info['nll'])), info['nll'],
     tc.ones(len(info['nll']))*nll_test.item(),
     marker=['s', ''], xlabel='epoch', ylabel='NLL',
     legend=['train', 'test'], linestyle=['-', '--'])
