import os, time
import torch as tc
from torch import nn
from matplotlib import pyplot as plt
from Library.BasicFun import gif_from_folder
from wheels_L2 import plot_points_and_line, plot_points_and_circle


print('设置参数')
r0 = 1.5
num = 500
epoch_train = 2000
epoch_check = 20
lr = 1e-3  # 学习率
tol = lr ** 2  # 收敛性判断标准
path = "./L2_nonlinear_problem/"
if not os.path.exists(path):
    os.mkdir(path)

print('随机生成%i个二维散点，半径大于%g时标签为1，否则标签为-1' % (r0, num))
points = tc.randn(num, 2)
r = points.norm(dim=1)
labels = (r>r0).to(dtype=points.dtype) * 2 - 1
print('前5个样本：')
print(points[:5, :])
print('前5个样本对应的标签：')
print(labels[:5])

print('作图：圆外的点用红色，否则用蓝色')
colors = ['red' if labels[i] > 0
          else 'blue' for i in range(labels.numel())]
plot_points_and_circle(points, colors, r0,
                       'Ground-truth classification', show=True)

print('利用nn.Linear建立分类器')
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
for t in range(epoch_train):  # 循环优化（训练）
    out = net(points).squeeze()
    loss = loss_fun(out, labels)
    loss_data.append(loss.item())
    bias_data.append(net.state_dict()['bias'].item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (t + 1) % epoch_check == 0:
        acc = ((out.data.sign() * labels) > 0).to(dtype=tc.float32).mean()
        title = 'Epoch: %i, loss: %g, acc: %g' % (t + 1, loss.item(), acc)
        slp = -(net.state_dict()['weight'][0, 0] /
                net.state_dict()['weight'][0, 1]).item()
        intcpt = -(net.state_dict()['bias'] /
                   net.state_dict()['weight'][0, 1]).item()
        plot_points_and_line(points, colors, slp, intcpt, title,
                             path, file_name=str(time.time()).replace('.', ''))
        print(net.weight.data)
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

'''



