import torch as tc


x1 = tc.tensor(1, dtype=tc.float64, requires_grad=True)
g1 = tc.autograd.grad(x1 ** 2, x1)[0]
print('The grad of x**2 at x=1: ', g1)

(x1 ** 2).backward()
print('The grad of x**2 at x=1: ', x1.grad)

x3 = tc.arange(3, dtype=tc.float64, requires_grad=True)
g3 = tc.autograd.grad(x3 ** 2, x3, grad_outputs=tc.ones_like(x3))[0]
print('The grad of x**2 at x=', x3.data, ':', g3)





