import os
import torch as tc
from torch import nn


num = 200
epoch_train = 5
lr = 1e-3  # 学习率
tol = lr ** 2  # 收敛性判断标准
path = "./L2_linear_classifier/"
if not os.path.exists(path):
    os.mkdir(path)

print('随机生成500个二维散点，横轴大于纵轴的点标签为1，否则标签为-1')
points = tc.randn(num, 2)
labels = (points[:, 0] > points[:, 1]).to(dtype=tc.int64) * 2 - 1

print('利用nn.Linear建立线性分类器，区分横轴大于纵轴或横轴小于纵轴的点')
net = nn.Linear(2, 1)

optimizer = tc.optim.Adam(net.parameters(), lr=lr)
loss_fun = nn.MSELoss()
labels = labels.to(points.dtype)
for t in range(epoch_train):  # 循环优化（训练）
    print('=' * 30)
    print('Epoch： %i' % t)
    out = net(points).squeeze()
    loss = loss_fun(out, labels)
    loss.backward()
    print('自动微分得到的关于weight梯度为：')
    print(net.weight.grad.data)

    grad_w0 = 2 * (points[:, 0] * (out.data - labels)).mean().item()
    grad_w1 = 2 * (points[:, 1] * (out.data - labels)).mean().item()
    print('由链式公式计算得到的关于weight梯度为：')
    print(grad_w0, grad_w1)

    optimizer.step()
    optimizer.zero_grad()

'''
练习：
利用微分的链式公式计算上述代码中bias的梯度，并于自动微分所得的梯度比较。
'''
