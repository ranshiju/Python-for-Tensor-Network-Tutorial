import torch as tc
from scipy.sparse.linalg import eigsh, LinearOperator


def linear_fun(mat, v):
    return mat.dot(v)


dim = 8  # matrix dimension
M = tc.randn((dim, dim), dtype=tc.float64)
M = (M + M.t()) / 2

lm0, v0 = tc.linalg.eigh(M)
lm_min = tc.min(lm0)

M1 = M.numpy()  # transfer pytorch tensor to numpy ndarray
lm1, v1 = eigsh(M1, k=1, which='SA')
print('M的最小本征值\n by torch: ', lm_min.item(),
      '\n by scipy: ', lm1[0])

M2 = LinearOperator(M1.shape, lambda x: linear_fun(M1, x))
lm2, v2 = eigsh(M2, k=1, which='SA')
print(' by LinearOperator: ', lm2[0])



