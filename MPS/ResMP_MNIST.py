import torch as tc
from torch import nn, optim
from torch.utils.data import DataLoader
from Library import DataFun as df
from Library.BasicFun import print_dict
from Library.MatrixProductState import ResMPS_basic


feature_map = 'reshape'  # 特征映射
it_time = 2000  # 迭代总次数
batch_size = 2000  # batch尺寸
lr = 1e-3  # 学习率
# ResMPS参数
para = {
    'num_c': 10,  # 类别数
    'length': 784,  # MPS长度（特征数）
    'd': 1,  # 特征映射维数
    'chi': 40,  # 虚拟维数
    'bias': False,  # 是否加偏置项
    'eps': 1e-6,  # 扰动大小
    'dropout': 0.2  # dropout概率（None为不dropout）
}
if feature_map == 'reshape':
    para['d'] = 1

train_dataset, test_dataset = df.load_mnist(
    'MNIST', process={'normalize': [0.5, 0.5]})
train_dataset = DataLoader(train_dataset, batch_size, True)
test_dataset = DataLoader(test_dataset, batch_size, False)

mps = ResMPS_basic(para=para)
print('ResMPS的超参数为：')
print_dict(mps.para)
optimizer = optim.Adam(mps.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for t in range(it_time):
    print('\n训练epoch: %d' % t)
    mps.train()
    loss_rec, num_c, total = 0, 0, 0
    for nb, (img, lb) in enumerate(train_dataset):
        vecs = df.feature_map(img.to(
            device=mps.device, dtype=mps.dtype),
            which=feature_map, para={'d': para['d']})
        lb = lb.to(device=mps.device)
        y = mps(vecs)
        loss = criterion(y, lb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_rec += loss.item()
        predicted = y.argmax(dim=1)
        total += lb.shape[0]
        num_c += predicted.eq(lb).sum().item()
        if (nb % 20 == 0) or (nb == len(train_dataset)-1):
            print('%i个batch已训练; loss: %g, acc: %g' %
                  ((nb+1), loss_rec/(nb+1), num_c/total))

    test_loss, num_c, total = 0, 0, 0
    mps.eval()
    print('测试集:')
    with tc.no_grad():
        for nt, (imgt, lbt) in enumerate(test_dataset):
            vecs = df.feature_map(imgt.to(
                device=mps.device, dtype=mps.dtype),
                which=feature_map, para={'d': para['d']})
            lbt = lbt.to(device=mps.device)
            yt = mps(vecs)
            loss = criterion(yt, lbt)
            test_loss += loss.item()
            predicted = yt.argmax(dim=1)
            total += lbt.shape[0]
            num_c += predicted.eq(lbt).sum().item()
        print('loss: %g, acc: %g%% '
              % (test_loss / (nt+1), 100 * num_c / total))
