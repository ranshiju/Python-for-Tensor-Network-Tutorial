import torch as tc
from scipy.sparse.linalg import eigsh
from Library.ExampleFun import eigs_power


dim = 4
H = tc.randn((dim, dim), dtype=tc.float64)
H = H + H.t()

print('For the eigenvalue and eigenvector with '
      'the largest magnitude:')
lm1, u1 = eigsh(H.numpy(), k=1, which='SM')
lm2, u2 = eigs_power(H, which='SM')
print('Results from scipy.sparse.linalg.eigsh:')
print('eigenvalue = ', lm1[0])
print('eigenvector = \n', u1.reshape(-1))

print('Results from eigs_power:')
print('eigenvalue = ', lm2.item())
print('eigenvector = \n', u2.numpy())

print('--------------------- 分割线 ---------------------')

print('For the eigenvalue and eigenvector with '
      'the smallest algebraic:')
lm1, u1 = eigsh(H.numpy(), k=1, which='SA')
lm2, u2 = eigs_power(H, which='SA')
print('Results from scipy.sparse.linalg.eigsh:')
print('eigenvalue = ', lm1[0])
print('eigenvector = \n', u1.reshape(-1))

print('Results from eigs_power:')
print('eigenvalue = ', lm2.item())
print('eigenvector = \n', u2.numpy())


