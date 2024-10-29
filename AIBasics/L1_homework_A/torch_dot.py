import torch as tc
import math
import matplotlib.pyplot as plt


def rotate_2d(a):
    return tc.tensor(
        [[math.cos(a), -math.sin(a)],
         [math.sin(a), math.cos(a)]],
        dtype=tc.float64)


def rotate_plot(vec, theta, n):
    u = rotate_2d(theta)
    _, ax = plt.subplots()
    ax.scatter(vec[0], vec[1])
    data = list()
    for t in range(1000):
        vec = u.matmul(vec)
        data.append(vec.reshape(2, 1))
    data = tc.cat(data, dim=1)
    ax.scatter(data[0], data[1])
    plt.show()


vec1 = tc.tensor([1, 0], dtype=tc.float64)
rotate_plot(vec1, math.pi/20, 1000)

vec2 = tc.tensor([1, 0], dtype=tc.float64)
rotate_plot(vec2, 0.1, 1000)
