import torch as tc
from Library.MathFun import rank1, rank1_product


# 建立(4,4,4,4)维rank-1张量
v = list()
for _ in range(4):
    v.append(tc.randn(4, dtype=tc.complex128))
T0 = rank1_product(v)

v1, c1 = rank1(T0)  # Rank-1分解
T01 = rank1_product(v1, c1)
err = (T0 - T01).norm().item() / T0.norm().item()
print('Rank-1 approximation error of a random rank-1 tensor = ', err)

print('--------------------- 分割线 ---------------------')
# 建立(4,4,4,4)维随机张量
T = tc.randn((4, 4, 4, 4), dtype=tc.float64)
v2, c2 = rank1(T)  # Rank-1分解
T1 = rank1_product(v2, c2)
err = (T - T1).norm().item() / T1.norm().item()
print('Rank-1 approximation error of a random tensor = ', err)



