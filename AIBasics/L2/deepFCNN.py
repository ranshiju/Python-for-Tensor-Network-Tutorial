import os, time
import numpy as np
import torch as tc
from torch import nn
from matplotlib import pyplot as plt
from Library.BasicFun import gif_from_folder
from wheels_L2 import plot_points_and_circle


class FCNN_2layer(nn.Module):

    def __init__(self, d_in, d_hidden, d_out):
        super(FCNN_2layer, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = tc.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.Sigmoid()(x)


def visualize_2dNN_classification_boundary(
        nn_model, x_min=-1, x_max=1, y_min=-1, y_max=1, num=100, data=None, colours=None,
        title='Decision Boundary of the Neural Network', save_path='.'):
    if data is not None:
        xx = data.abs().max()
        x_min = -xx
        x_max = xx
        y_min = -xx
        y_max = xx
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num),
                         np.linspace(y_min, y_max, num))
    # 将网格点转换为 PyTorch 张量
    dtype = next(nn_model.parameters()).dtype
    grid_points = tc.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=dtype)
    # 使用训练好的模型预测网格点的类别
    nn_model.eval()  # 进入评估模式
    with tc.no_grad():
        logits = nn_model(grid_points)
        preds = (logits > 0.5).to(dtype=dtype)  # 将输出转换为类别 0 或 1
        preds = preds.numpy().reshape(xx.shape)
        # 绘制分类边界
        plt.figure(figsize=(6, 6))
        plt.contourf(xx, yy, preds, alpha=0.8, cmap='coolwarm')
        if data is not None:
            plt.scatter(data[:, 0], data[:, 1], c=colours)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.savefig(os.path.join(save_path, str(time.time()).replace('.', '')))
        plt.cla()
        plt.close()


print('设置参数')
r0 = 1.5
num = 500
epoch_train = 2000
epoch_check = 50
lr = 1e-3  # 学习率
tol = lr ** 2  # 收敛性判断标准
path = "./L2_FCNN_2layer/"
if not os.path.exists(path):
    os.mkdir(path)

print('随机生成%i个二维散点，半径大于%g时标签为1，否则标签为0' % (r0, num))
points = tc.randn(num, 2)
r = points.norm(dim=1)
labels = (r>r0).to(dtype=points.dtype)

print('作图：圆外的点用红色，否则用蓝色')
colors = ['red' if labels[i] > 0
          else 'blue' for i in range(labels.numel())]
plot_points_and_circle(points, colors, r0,
                       'Ground-truth classification', show=True)

print('利用nn.Linear建立分类器')
net = FCNN_2layer(2, 20, 1)

optimizer = tc.optim.Adam(net.parameters(), lr=lr)
loss_fun = nn.BCELoss()
loss_data = list()
for t in range(epoch_train):  # 循环优化（训练）
    out = net(points).squeeze()
    loss = loss_fun(out, labels)
    loss_data.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (t + 1) % epoch_check == 0:
        pred = (out.data > 0.5).to(dtype=tc.float32)
        acc = (pred == labels).to(dtype=tc.float32).mean()
        title = 'Epoch: %i, loss: %g, acc: %g' % (t + 1, loss.item(), acc)
        print(title)
        visualize_2dNN_classification_boundary(
            net, -2, 2, -2, 2, data=points, colours=colors, title=title, save_path=path)
        if abs((loss_data[-1] - loss_data[-2])) < tol:
            break
gif_from_folder(path, duration=100)

plt.plot(tc.arange(len(loss_data)), loss_data)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(os.path.join(path, 'loss'))
plt.show()


'''
练习：
1. 编写函数，计算神经网络模型的总复杂度；
2. 修改神经网络模型代码（第10-20行），将网络层数作为输入参数，实现对应层数的全连接神经网络；
3. 通过数值计算比较层数（深度）与维数（宽度）的作用：尝试利用不同层数与不同维数的神经网络实现
  本代码考虑的非线性分类问题，在保持总参数大致不变的情况下，针对该问题，研究神经网络性能随层数
  与维数的变化趋势。

'''



