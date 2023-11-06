import torch as tc

u = tc.randn(4)
v = tc.randn(4)

z1 = u.dot(v)
z2 = u.inner(v)
z3 = tc.einsum('digit,digit->', u, v)
print('三种方法计算的向量内积结果：', z1, z2, z3)

z4 = 0
for n in range(u.numel()):
    z4 += u[n] * v[n]
print('循环方法计算的向量内积结果：', z4)

z5 = (u * v).sum()
print('Hadamard积与求和方法计算的向量内积结果：', z5)