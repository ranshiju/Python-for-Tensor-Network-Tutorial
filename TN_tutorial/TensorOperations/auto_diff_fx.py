import torch as tc
import matplotlib.pyplot as plt


lr = 0.01
it_time = 100
x = tc.tensor([1], dtype=tc.float64, requires_grad=True)

coordinates = tc.zeros((it_time, 2), dtype=tc.float64)
for t in range(it_time):
    y = x ** 2
    coordinates[t, 0] = x.data
    coordinates[t, 1] = y.tensor
    y.backward()
    x.data = x.data - lr * x.grad
    x.grad.tensor.zero_()


x = tc.tensor([1], dtype=tc.float64, requires_grad=True)
optimizer = tc.optim.Adam([x], lr=lr)
coordinates1 = tc.zeros((it_time, 2), dtype=tc.float64)
for t in range(it_time):
    y = x ** 2
    coordinates1[t, 0] = x.data
    coordinates1[t, 1] = y.tensor
    y.backward()
    optimizer.step()
    optimizer.zero_grad()


line1, = plt.plot(list(range(it_time)), coordinates[:, 1])
line2, = plt.plot(list(range(it_time)), coordinates1[:, 1],
                  linestyle='--', color='orange')
plt.legend([line1, line2], ['fixed lr', 'Adam'])
plt.text(80, 0.15, 'lr = %g' % lr, fontsize=16)
plt.show()
