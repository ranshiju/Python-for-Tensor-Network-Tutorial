import torch as tc


# 计算exp(x)
x = 0.8
# 将x转化为tc.Tensor类型的变量之后，才能调用tc的函数
x1 = tc.tensor(x, device='cpu', dtype=tc.float64)
print('x1的类型为：', type(x1))
print('x1的阶数（ndimension）：', x1.ndimension())
print('x1的形状（shape）：', x1.shape)

y = tc.exp(x1)
print('exp(x)的计算结果为：', y)
print('y的类型为：', type(y))
print('y的阶数（ndimension）：', y.ndimension())
print('y的形状（shape）：', y.shape)

# Tensor标量的阶数为0，没有维数（可认为是空元组）

