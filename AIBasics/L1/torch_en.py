import torch as tc


print('考虑8个行向量：')
vecs = tc.randn(8, 3)
print(vecs)

print('每个行向量的L2范数为：')
norms = vecs.norm(dim=1)
print(norms)

print('利用einsum对每个行向量归一化：')
vecs1 = tc.einsum('na,n->na', vecs, 1/norms)
print('归一结果为：')
print(vecs1)
print('检查L2范数：')
print(vecs1.norm(dim=1))

"""
练习：
将本代码第5行的矩阵看作是3个8维列向量，利用einsum实现每个列向量的归一化，并验证结果。
"""

