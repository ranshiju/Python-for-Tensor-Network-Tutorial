import torch as tc
import math


# 向量内积
x = tc.tensor([1, 0, 0], dtype=tc.float64)
y = tc.tensor([1/math.sqrt(2), 0, 1/math.sqrt(2)], dtype=tc.float64)
print('两个向量的范数分别为：')
print(x.norm(), y.norm())
print('两个向量的内积为：')
print(x.dot(y))
print(x.matmul(y))
print(tc.einsum('n,n', x, y))

# 矩阵乘向量
a = 1/math.sqrt(2)
x = tc.tensor([a, a], dtype=tc.float64)
mat = tc.tensor([[a, -a], [a, a]], dtype=tc.float64)
print(mat.matmul(x))
print(tc.einsum('ab,b->a', mat, x))

"""
练习：二维向量空间中，转角为a的旋转矩阵定义为
[[cos a, -sin a]
 [sin a, cos a]]
1. 定义python函数，函数输入为转角，输出为torch.tensor类型的旋转矩阵，
    数据类型为float64；
2. 定义python函数，函数输入为二维向量、旋转角度、旋转次数，函数的内容为以输
    入的二维向量为初始，循环地将对应旋转角度的旋转矩阵作用到向量上，每次作用
    后，画出以向量两个元素为x、y轴坐标的点。
    （提示：可调用matplotlib.pyplot中的scatter函数画点，一共有循环次数
    个点，不要旋转一次调用一次scatter函数，而是计算好全部旋转后储存好数据，
    调用scatter函数一次性画出所有点）
"""
