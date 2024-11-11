import torch as tc


x = tc.arange(12).reshape(2, 2, 3)
print('元素4的索引为：')
print(tc.nonzero(tc.eq(x, 4)))
x1 = x.permute(1, 0, 2)
print('交换指标后，元素4的索引为：')
print(tc.nonzero(tc.eq(x1, 4)))

print('变形为矩阵：')
x2 = x.reshape(4, 3)
print(x2)
print('矩阵转置的两种方式：')
print(x2.t())
print(x2.permute(1, 0))


