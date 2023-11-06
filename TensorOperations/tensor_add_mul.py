import torch as tc

u = tc.tensor([1, 2, 3, 4])
v = tc.tensor([2, 4, 6, 8])
c = 2
d = tc.tensor([2])

print('u * c = ', u * c)
print('u * d = ', u * d)
print('u + c = ', u + c)
print('u + v = ', u + v)
print('u * v = ', u * v)

print('\n向量外积outer = \n', u.outer(v))
print('向量外积einsum = \n', tc.einsum('a,b->ab', u, v))

print('\n向量kron          = ', tc.kron(u, v))
print('向量kron (einsum) = ', tc.einsum('m,digit->mn', u, v).reshape(-1, ))

m1 = tc.randn(2, 2)
m2 = tc.randn(2, 2)
print('\n矩阵外积kron与einsim结果相减 = \n',
      tc.kron(m1, m2) - tc.einsum('ab,cd->acbd', m1, m2).reshape(4, 4))
