import torch as tc
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from Library.BasicFun import choose_device
from Library.ADQC import \
    position_one_layer, FCNN_ADQC_latent
from AD_Circuits.wheels import \
    probabilities_adqc_classifier
from Library.DataFun import load_mnist

# 注：本代码考虑的是二分类问题
dims_nn = [784, 100, 5]  # NN维数 [输入，中间，输出]
nn_depth = 1  # NN层数
qc_depth = 2  # QC层数

it_time = 2000  # 迭代总次数
batch_size = 300  # batch尺寸
lr = 1e-4  # 学习率
device = choose_device()  # 自动选择设备（cuda优先）
print('训练NN-ADQC的设备：', device)

train_dataset, test_dataset = load_mnist(
    'MNIST', process={'classes': [0, 1]})
train_dataset = DataLoader(
    train_dataset, batch_size, True)
test_dataset = DataLoader(
    test_dataset, batch_size, False)
pos = position_one_layer('brick', dims_nn[2])

# 建立FCNN_ADQC_latent实例
net = FCNN_ADQC_latent(
    pos, dims_nn[0], dims_nn[1], dims_nn[2],
    nn_depth, qc_depth, device)
optimizer = Adam(net.parameters(), lr=lr)

print('各个变分参数的形状与类型为：')
for x in net.parameters():
    print(x.shape, type(x))
criterion = nn.CrossEntropyLoss()

for t in range(it_time):
    print('\n训练epoch: %d' % t)
    loss_rec, num_c, total = 0, 0, 0
    for nb, (img, lb) in enumerate(train_dataset):
        img = img.to(device=device)
        lb = lb.to(device=device)
        psi1 = net(img)
        y = probabilities_adqc_classifier(psi1, 1, 2)
        loss = criterion(y, lb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_rec += loss.item()
        _, predicted = y.max(1)
        total += lb.shape[0]
        num_c += predicted.eq(lb).sum().item()
        if (nb % 20 == 0) or (
                nb == len(train_dataset)-1):
            print('%i个batch已训练; loss: %g, '
                  'acc: %g%%' % ((nb+1), loss_rec/(
                    nb+1), num_c/total*100))

    test_loss, num_c, total = 0, 0, 0
    print('测试集:')
    with tc.no_grad():
        for nt, (imgt, lbt) in enumerate(test_dataset):
            imgt = imgt.to(device=device)
            lbt = lbt.to(device=device)
            psi1 = net(imgt.to(device=device))
            yt = probabilities_adqc_classifier(psi1, 1, 2)
            loss = criterion(yt, lbt)
            test_loss += loss.item()
            _, predicted = yt.max(1)
            total += lbt.shape[0]
            num_c += predicted.eq(lbt).sum().item()
        print('loss: %g, acc: %g%% '
              % (test_loss / (
                nt+1), 100 * num_c / total))



