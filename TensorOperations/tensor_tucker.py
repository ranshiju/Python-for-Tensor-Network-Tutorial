import torch as tc
from Library.MathFun import hosvd, tucker_product, \
      reduced_matrix, tucker_rank

x = tc.randn((2, 3, 4), dtype=tc.complex128)
rank = tucker_rank(x)
print('Tucker rank of x is ', rank)

g, u = hosvd(x)

print('The bond-0 reduced matrix of the core tensor g '
      '(real part): \n',
      reduced_matrix(g, 0).numpy().real)
print('The bond-0 reduced matrix of the core tensor g '
      '(imaginary part): \n',
      reduced_matrix(g, 0).numpy().imag)

g1, u1 = hosvd(x, (2, 3, 3))
x1 = tucker_product(g1, u1)
err = (x1 - x).norm() / x.norm()
print('Relative error of truncating x to (2, 3, 3) = ',
      err.item())

g1, u1 = hosvd(x, (2, 2, 2))
x1 = tucker_product(g1, u1)
err = (x1 - x).norm() / x.norm()
print('Relative error of truncating x to (2, 2, 2) = ',
      err.item())
