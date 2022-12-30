import torch as tc

x = tc.tensor([1, 3, 5, 7])
x1 = tc.reshape(x, [2, 2])
x1_ = x.reshape(2, 2)
print(x1)
print(x1_)

print('测试张量元"1"的指标', x[0], x1[0, 0])
print('测试张量元"5"的指标', x[2], x1[1, 0])

print('--------------------- 分割线 ---------------------')
x2 = x1.permute(1, 0)
print('测试permute前，张量元"5"的指标', x1[1, 0])
print('测试permute后，张量元"5"的指标', x2[0, 1])

x2_ = x1.t()
print('x2 = ', x2)
print('x2_ = ', x2_)

print('--------------------- 分割线 ---------------------')
T = tc.randn(2, 3, 4)
print('张量T的形状为', T.shape)
V = T.flatten()
print('其向量化后(flatten)的形状为', V.shape)
print('flatten与reshape向量化所得结果的差别',
      (V - T.reshape(-1)).norm().item())
print('矩阵化T[0,1]的形状为', T.reshape(-1, 4).shape)
