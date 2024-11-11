import torch as tc


x = tc.arange(8)
print('x[3] = ', x[3])
print('x[1:4] = ', x[1:4])

y = [n*tc.ones(4, 1) for n in range(6)]
y = tc.cat(y, dim=1)
print('y = \n', y)
print('y[0] = ', y[0])
print('y[0, 3] = ', y[0, 3])
print('y[0, 3] = ', y[0][3])
print('y[0:2, 1:5] = \n', y[0:2, 1:5])


