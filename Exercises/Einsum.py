import torch as tc

# 已知如下三个张量
x = tc.randn(2, 2, 3)
y = tc.randn(2, 2)
z = tc.randn(3, 2)

# 利用reshape等函数实现某种缩并计算
tmp = y.mm(x.reshape(2, -1)).reshape(2, 2, 3)
tmp = tmp.permute(1, 0, 2).reshape(2, -1)
out = tmp.matmul(z.t().reshape(-1))
print(out)

"""
请使用einsum函数完成上述运算，打印出所得向量，须与第10行所得向量一致。
（提示：建议尝试给出图形表示）
"""
