import os, time
import torch as tc
from torch import nn
from matplotlib import pyplot as plt
from Library.BasicFun import gif_from_folder
from wheels_L2 import plot_points_and_line


print('设置参数')
num = 200
epoch_train = 2500
epoch_check = 20
lr = 1e-3  # 学习率
tol = lr ** 2  # 收敛性判断标准
path = "./L2_linear_classifier/"
if not os.path.exists(path):
    os.mkdir(path)

print('随机生成%i个二维散点，横轴大于纵轴的点标签为1，否则标签为-1' % num)
points = tc.randn(num, 2)
labels = (points[:, 0] > points[:, 1]).to(dtype=points.dtype) * 2 - 1
print('前5个样本：')
print(points[:5, :])
print('前5个样本对应的标签：')
print(labels[:5])

print('作图：横轴大于纵轴的点用红色，否则用蓝色')
colors = ['red' if labels[i] > 0
          else 'blue' for i in range(labels.numel())]
plot_points_and_line(points, colors, 1, 0,
                     'Ground-truth classification', show=True)

print('利用nn.Linear建立线性分类器，区分横轴大于纵轴或横轴小于纵轴的点')
net = nn.Linear(2, 1)

print('绘制初始（随机）Linear分类器对应的分类边界')
slp = -(net.state_dict()['weight'][0, 0] /
        net.state_dict()['weight'][0, 1]).item()
intcpt = -(net.state_dict()['bias'] /
           net.state_dict()['weight'][0, 1]).item()
plot_points_and_line(points, colors, slp, intcpt,
                     'Initial classification')

optimizer = tc.optim.Adam(net.parameters(), lr=lr)
loss_fun = nn.MSELoss()
loss_data = list()
bias_data = list()
labels = labels.to(points.dtype)
for t in range(epoch_train):  # 循环优化（训练）
    out = net(points).squeeze()
    loss = loss_fun(out, labels.to(out.dtype))
    # loss = ((out - labels) ** 2).sum() / labels.numel()
    loss_data.append(loss.item())
    bias_data.append(net.state_dict()['bias'].item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # for x in net.parameters():
    #     x.data = x.data - lr * x.grad.data
    #     x.grad.zero_()
    if (t+1) % epoch_check == 0:
        acc = ((out.data.sign() * labels) > 0).to(dtype=tc.float32).mean()
        title = 'Epoch: %i, loss: %g, acc: %g' % (t + 1, loss.item(), acc)
        slp = -(net.state_dict()['weight'][0, 0] /
                net.state_dict()['weight'][0, 1]).item()
        intcpt = -(net.state_dict()['bias'] /
                   net.state_dict()['weight'][0, 1]).item()
        plot_points_and_line(points, colors, slp, intcpt, title,
                             path, file_name=str(time.time()).replace('.', ''))
        if abs((loss_data[-1] - loss_data[-2])) < tol:
            break
plt.show()
gif_from_folder(path, duration=100)

plt.plot(tc.arange(len(loss_data)), loss_data)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(os.path.join(path, 'loss'))
plt.show()

plt.plot(tc.arange(len(bias_data)), bias_data)
plt.xlabel('epoch')
plt.ylabel('bias')
plt.savefig(os.path.join(path, 'bias'))
plt.show()


'''
练习：
1. 利用tc.save函数储存上述例子中训练好的Linear层所有参数，编程写函数实现储存参数的
读取，并利用einsum等张量运算函数，实现Linear层的前馈映射（不使用nn.Linear）；
2. 仿照上述例子，利用一个线性层判断3维空间中的散点是否位于x+y+z=0平面的
上方或下方（可考虑作3维图展示）；
3. 计算比较使用简单的梯度下降与Adam优化器时，收敛速度的差别。
'''



