import torch as tc
import matplotlib.pyplot as plt


# 全零张量的建立
print('全零向量（1阶张量）')
x = tc.zeros(3)
print(x)
print(x.shape)
print(x.dtype, x.device)

print('全零矩阵（2阶张量）')
y = tc.zeros((3, 3))
print(y)
print(y.shape, y.numel(), y.ndimension())

print('全零张量（4阶张量）')
z = tc.zeros((2, 3, 2, 3))
print(z.shape, z.numel(), z.ndimension())

print('随机张量（标准高斯分布）')
print(tc.randn(4))
print('随机张量（0到1均匀分布）')
print(tc.rand(4))
print('等差数列')
print(tc.arange(2, 10, 2))

print('绘制x*sinx函数（0到10pi区间，500个数据点）')
x = tc.arange(500) * tc.pi / 50
y = x*tc.sin(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('x sin x')
plt.show()

"""
练习：
1. 定义python函数，命名为plt_sln，实现sin(ln(x))的绘制，要求plt_sln函数的
    输入为：自变量起点，自变量终点，数据点个数；在程序内部判断自变量的取值需为大
    于零的正数；运行函数后绘制出图像，无需返回值。
2. 调用上述程序，在自变量取0.01到10的区间绘制图像，相邻数据点横坐标间隔0.01；同时
    添加横坐标标题为x，纵坐标标题为sin(ln(x))，借助网络资料，尝试更改图像中的实现为
    虚线，线粗为2，颜色为红色；设置横纵坐标标题字体为新罗马字体（Times New Roman），
    字号为15号。
"""
