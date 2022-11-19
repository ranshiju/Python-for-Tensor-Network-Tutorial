import torch as tc

P = tc.randn((2, 2))
Q = tc.randn((2, 2))
x = P.mm(Q)
x1 = P.matmul(Q)
x2 = tc.einsum('ab,bc->ac', P, Q)

print('三种内置函数方法计算的矩阵乘结果：')
print('x = \n', x)
print('x1 = \n', x1)
print('x2 = \n', x2)

print('--------------------- 分割线 ---------------------')
x3 = tc.zeros((2, 2))
x4 = tc.zeros((2, 2))
for n1 in range(P.shape[0]):
    for n2 in range(Q.shape[1]):
        for n3 in range(P.shape[1]):
            x3[n1, n2] += P[n1, n3] * Q[n3, n2]
        x4[n1, n2] += P[n1, :].dot(Q[:, n2])

print('两种循环方法计算的矩阵乘结果：')
print('x3 = \n', x3)
print('x5 = \n', x4)

