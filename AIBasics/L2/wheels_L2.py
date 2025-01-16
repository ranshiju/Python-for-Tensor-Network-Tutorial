import os
import torch as tc
from matplotlib import pyplot as plt


def plot_points_and_line(data, colours, slop, intercept, title='',
                         save_path=None, file_name='fig', show=False):
    plt.scatter(data[:, 0], data[:, 1], c=colours)
    xm = data.abs().max()
    plt.xlim(-xm, xm)  # 设置横坐标范围
    plt.ylim(-xm, xm)  # 设置纵坐标范围
    plt.axline((0, intercept), slope=slop, color="gray",
               linestyle="--", linewidth=1)
    plt.title(title)
    if save_path:
        plt.savefig(os.path.join(save_path, file_name))
    if show:
        plt.show()


def calculate_distance_to_line(k, b, points):
    """
    计算二维空间中N个点到直线的距离

    :param k: 直线的斜率
    :param b: 直线的截距
    :param points: 一个N*2的矩阵，表示N个点的坐标
    :return: 返回一个N维的张量，表示每个点到直线的距离
    """
    # 将输入的点转换为torch张量
    if type(points) is not tc.Tensor:
        points = tc.tensor(points, dtype=tc.float32)

    # 提取每个点的x和y坐标
    x = points[:, 0]
    y = points[:, 1]

    # 计算点到直线的距离
    numerator = tc.abs(k * x - y + b)  # 分子 |kx - y + b|
    denominator = tc.sqrt(k**2 + 1)    # 分母 sqrt(k^2 + 1)

    # 返回每个点到直线的距离
    return numerator / denominator


def plot_points_and_circle(data, colours, radius, title='',
                           save_path=None, file_name='fig', show=False):
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=colours)
    xm = data.abs().max()
    plt.xlim(-xm, xm)  # 设置横坐标范围
    plt.ylim(-xm, xm)  # 设置纵坐标范围
    circle = plt.Circle((0, 0), radius, fill=False, color="gray",
               linestyle="--", linewidth=1)
    ax.add_artist(circle)
    plt.title(title)
    ax.set_aspect("equal")
    if save_path:
        plt.savefig(os.path.join(save_path, file_name))
    if show:
        plt.show()


def nn_total_parameter(net):
    return sum(p.numel() for p in net.parameters())

