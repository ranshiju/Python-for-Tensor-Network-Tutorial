import torch as tc


d = 3
H = tc.randn((d, d), dtype=tc.complex128)
H = H + H.conj().t()
print('随机厄密矩阵 H = \n', H)

lm, u = tc.linalg.eigh(H)
print('本征值 lm = ', lm)
print('变换矩阵 u = \n', u)

lm = lm.to(dtype=tc.complex128)
epsilon = (u.mm(lm.diag()).mm(u.conj().t()) - H).norm()
print('分解误差 = ', epsilon.item())

print('--------------------- 分割线 ---------------------')
print('测试本征值方程：')
print('lm[1]*u[:, 1] = \n', lm[1]*u[:, 1])
print('H.matmul(u[:, 1]) = \n', H.matmul(u[:, 1]))

print('--------------------- 分割线 ---------------------')
lm_ = tc.linalg.eigvalsh(H)
print('eigvalsh函数所得的本征值 = ', lm_)

print('--------------------- 分割线 ---------------------')
M = tc.tensor([[0, 0], [1, 0]], dtype=tc.float64)
lm, u = tc.linalg.eig(M)
epsilon = (u.mm(lm.diag()).mm(u.conj().t()) - M).norm()
print('分解误差 = ', epsilon.item())



