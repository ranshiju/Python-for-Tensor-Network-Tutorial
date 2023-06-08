import torch as tc
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Library.DataFun import load_mnist
from Algorithms.MPS_algo import ResMPS_classifier


dataset = 'fmnist'
batch_size = 1000
para = {
    'ResMPS': 'simple',  # ResMPS种类
    'lr': 1e-4,  # 学习率
    'feature_map': 'linear',
    'it_time': 30  # 优化次数
}
paraMPS = {
    'd': 2,  # 物理指标维数
    'chi': 100,  # 虚拟维数
    'pos_c': 'mid',  # 类别指标位置
    'bias': False,  # 偏置项
    'last_fc': True,  # 是否在最后加一层FC层
    'bias_fc': True,  # 线性层是否加bias
    'dropout': 0.6  # dropout概率（None为不dropout）
}

train_dataset, test_dataset = load_mnist(dataset)
train_loader = DataLoader(train_dataset, batch_size, True)
test_loader = DataLoader(test_dataset, batch_size, False)

infor_s, _ = ResMPS_classifier(
    train_loader, test_loader, para=para, paraMPS=paraMPS)

para['ResMPS'] = 'activated'
paraMPS['activation'] = 'ReLU'
infor_a, _ = ResMPS_classifier(
    train_loader, test_loader, para=para, paraMPS=paraMPS)

x = tc.arange(infor_s['train_acc'].numel())
l1, = plt.plot(x, infor_s['train_acc'], color='blue')
l2, = plt.plot(x, infor_s['test_acc'], color='red')
l3, = plt.plot(x, infor_a['train_acc'],
               linestyle='--', color='green')
l4, = plt.plot(x, infor_a['test_acc'],
               linestyle='--', color='black')
plt.legend([l1, l2, l3, l4],
           ['train acc (simple)', 'test acc (simple)',
            'train acc ('+paraMPS['activation']+')',
            'test acc ('+paraMPS['activation']+')'])
plt.xlabel('training epoch')
plt.show()
