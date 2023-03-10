import torch as tc
from torch import nn, optim
from torch.utils.data import DataLoader
from Library import DataFun as df
from Library.BasicFun import choose_device


dim = 100  # 隐藏层维数
it_time = 2000  # 迭代总次数
batch_size = 300  # batch尺寸
lr = 1e-4  # 学习率
device = choose_device()  # 自动选择设备（cuda优先）
print('训练FCNN的设备：', device)


class fcnn(nn.Module):

    def __init__(self, dim_hidden):
        super(fcnn, self).__init__()
        self.layer1 = nn.Linear(784, dim_hidden)
        self.layer2 = nn.Linear(dim_hidden, 10)

    def forward(self, x):
        x1 = self.layer1(x.reshape(x.shape[0], -1))
        x1 = nn.ReLU()(x1)
        x1 = self.layer2(x1)
        return nn.Sigmoid()(x1)


train_dataset, test_dataset = df.load_mnist('MNIST')
train_dataset = DataLoader(train_dataset, batch_size, True)
test_dataset = DataLoader(test_dataset, batch_size, False)

net = fcnn(dim).to(device=device)
optimizer = optim.Adam(net.parameters(), lr=lr)
print('神经网络各个变分参数的形状与类型为：')
for x in net.parameters():
    print(x.shape, type(x))
criterion = nn.CrossEntropyLoss()

for t in range(it_time):
    print('\n训练epoch: %d' % t)
    loss_rec, num_c, total = 0, 0, 0
    for nb, (img, lb) in enumerate(train_dataset):
        img = img.to(device=device)
        lb = lb.to(device=device)
        y = net(img)
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
    print('测试集:')
    with tc.no_grad():
        for nt, (imgt, lbt) in enumerate(test_dataset):
            imgt = imgt.to(device=device)
            lbt = lbt.to(device=device)
            yt = net(imgt.to(device=device))
            loss = criterion(yt, lbt)
            test_loss += loss.item()
            predicted = yt.argmax(dim=1)
            total += lbt.shape[0]
            num_c += predicted.eq(lbt).sum().item()
        print('loss: %g, acc: %g%% '
              % (test_loss / (nt+1), 100 * num_c / total))
