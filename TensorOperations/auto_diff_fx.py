import torch as tc
import matplotlib.pyplot as plt


lr = 1e-2
it_time = 100
x = tc.tensor([1], dtype=tc.float64, requires_grad=True)

coordinates = tc.zeros((it_time, 2), dtype=tc.float64)
for t in range(it_time):
    y = x ** 2
    coordinates[t, 0] = x.data
    coordinates[t, 1] = y.data
    y.backward()
    x.data = x.data - lr * x.grad
    x.grad.data.zero_()


x = tc.tensor([1], dtype=tc.float64, requires_grad=True)
optimizer = tc.optim.Adam([x], lr=lr)
coordinates1 = tc.zeros((it_time, 2), dtype=tc.float64)
for t in range(it_time):
    y = x ** 2
    coordinates1[t, 0] = x.data
    coordinates1[t, 1] = y.data
    y.backward()
    optimizer.step()
    optimizer.zero_grad()


plt.plot(list(range(it_time)), coordinates[:, 1])
plt.plot(list(range(it_time)), coordinates1[:, 1],
         linestyle='--')
plt.show()
