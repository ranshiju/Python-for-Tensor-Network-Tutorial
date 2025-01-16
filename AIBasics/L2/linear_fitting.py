import os, time
import torch as tc
from torch import nn
from matplotlib import pyplot as plt
from Library.BasicFun import gif_from_folder
from wheels_L2 import plot_points_and_line, calculate_distance_to_line


print('设置参数')
r0 = 1.5
num = 500
epoch_train = 2000
epoch_check = 20
lr = 1e-3  # 学习率
tol = lr ** 2  # 收敛性判断标准
path = "./L2_LF_nonlinear_problem/"
if not os.path.exists(path):
    os.mkdir(path)

print('随机生成%i个二维散点' % r0)
points = tc.randn(num, 2)
points[:, 1] = points[:, 0] * 0.2 + points[:, 1] * 0.1 # 分布变为椭圆

print('建立斜率与截距，作为自动微分变量')
k = tc.randn(1, requires_grad=True)
intc = tc.randn(1, requires_grad=True)

print('绘制初始（随机）Linear分类器对应的分类边界')
colors = ['red'] * points.shape[0]
plot_points_and_line(points, colors, k.item(), intc.item(),
                     'Initial classification')

optimizer = tc.optim.Adam([k, intc], lr=lr)
loss_fun = nn.MSELoss()
loss_data = list()
for t in range(epoch_train):  # 循环优化（训练）
    loss = calculate_distance_to_line(k, intc, points).mean()
    loss_data.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (t + 1) % epoch_check == 0:
        title = 'Epoch: %i, loss: %g' % (t + 1, loss.item())
        plot_points_and_line(points, colors, k.item(), intc.item(), title,
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


'''
练习：
考虑并改写“linear_classifier.py”：尝试不要使用nn.Linear，而是直接建立可自动微分的权重与偏执
（仿照本代码第25-26行），实现线性分类。
'''



