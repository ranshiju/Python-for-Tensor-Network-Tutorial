import torch as tc


M = tc.tensor([[0, 0], [1, 0]], dtype=tc.float64)
lm, u = tc.linalg.eig(M)
epsilon = (u.mm(lm.diag()).mm(u.conj().t()) - M).norm()
print('epsilon = ', epsilon.item())

print('--------------------- 分割线 ---------------------')

d = 3
H = tc.randn((d, d), dtype=tc.complex128)
H = H + H.conj().t()
print('The matrix H = \n', H)

lm, u = tc.linalg.eigh(H)
print('eigenvalues lm = ', lm)
print('transformation matrix u = \n', u)

lm = lm.to(dtype=tc.complex128)
epsilon = (u.mm(lm.diag()).mm(u.conj().t()) - H).norm()
print('epsilon = ', epsilon.item())

print('--------------------- 分割线 ---------------------')

lm_ = tc.linalg.eigvalsh(H)
print('Eigenvalues from eigvalsh = ', lm_)

print('--------------------- 分割线 ---------------------')

print('lm[1]*u[:, 1] = \n', lm[1]*u[:, 1])
print('H.matmul(u[:, 1]) = \n', H.matmul(u[:, 1]))


