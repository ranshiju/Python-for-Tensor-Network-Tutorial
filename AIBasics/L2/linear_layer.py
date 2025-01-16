import torch as tc
from torch import nn


print('建立一个4*2的线性层')
f = nn.Linear(4, 2)
print('神经网络层的类型：', type(f))

print('通过parameters()查看一个线性层中的参数')
print(type(f.parameters()))
for x in f.parameters():
    print(x)

print('=' * 30)  # 分割线 =================
print('通过state_dict()查看一个线性层中的参数')
print(type(f.state_dict()))
for x in f.state_dict():
    print(x)
    print(f.state_dict()[x])

data_in = tc.randn(3, 4)
print('输入：')
print(data_in)
data_out = f(data_in)
print('输出：')
print(data_out)

print('=' * 30)  # 分割线 ====================
print('利用einsum实现f映射：')
weight = f.state_dict()['weight']
bias = f.state_dict()['bias']
data_out1 = tc.einsum('na,ba->nb', data_in, weight) + bias
print('输出：')
print(data_out1)

print('=' * 30)  # 分割线 ====================
print('(m*n)的矩阵 + n维向量 = 矩阵的每行加上该向量')
print(tc.eye(3) + tc.tensor([0, 1, 2]))

'''
练习：
建立10个随机二维向量，以及一个（2*1）的nn.Linear层，将这些向量映射成10个
标量，要求每个标量给出对应二维向量的平均值。
（提示：线性层的元素为0.5；可考虑实用load_state_dict函数手动赋值线性层的
权重与偏置，该函数的具体用法可上网查询；可能用到collections.OrderedDict
函数）
'''



