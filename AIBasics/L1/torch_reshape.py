import torch as tc


# 变形reshape
x = tc.arange(8)
print(x)
print('元素4的索引为：')
print(tc.nonzero(tc.eq(x, 4)))
# 查询等于某个值的元素的指标：tc.nonzero(tc.eq(tensor, value))

x1 = x.reshape(4, 2)
print('变形为矩阵：')
print(x1)
print('元素4的索引为：')
print(tc.nonzero(tc.eq(x1, 4)))

x2 = x1.reshape(2, 2, 2)
print('变形为3阶张量：')
print(x2)
print('元素4的索引为：')
print(tc.nonzero(tc.eq(x2, 4)))

# 变形顺序不影响结果!
print('两种不同变形方式所得结果之差：')
print(x.reshape(2, 2, 2).reshape(2, 4)
      - x.reshape(2, 4))


