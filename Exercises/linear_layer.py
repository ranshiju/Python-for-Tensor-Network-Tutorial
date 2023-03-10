import torch as tc
from torch import nn

layer = nn.Linear(20, 8)
ReLU = nn.ReLU()
data = tc.rand(2, 20)
w = layer.weight
b = layer.bias

out = ReLU(layer(data))
print(out)

'''
利用data、w和b间的矩阵乘法、加法等运算，
计算出线性层的输出结果out
'''
