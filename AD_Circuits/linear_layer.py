import torch as tc
from torch import nn

layer = nn.Linear(20, 10)
print('线性层中的参数包括：')
print('权重weight: 形状为', layer.weight.shape)
print('偏置bias: 形状为', layer.bias.shape)

data = tc.randn((2, 20), dtype=tc.float32)
out1 = layer(data)
out2 = data.mm(layer.weight.t()) + layer.bias

print('两种方式得到的结果之差为：%g'
      % tc.norm(out1 - out2))





