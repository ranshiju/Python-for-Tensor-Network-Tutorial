from random import choices
from torch.linalg import svdvals, eigvalsh
from torch import sqrt
from collections import Counter
from Library.QuantumState import TensorPureState
from Library.BasicFun import binary_strings


print('求[1，2]位置对应的约化密度矩阵')
psi = TensorPureState(nq=4)  # 4-qubit随机量子态
rmd1 = psi.reduced_density_matrix([1, 2])

tmp = psi.tensor.permute(1, 2, 0, 3).reshape(4, 4)
rmd2 = tmp.mm(tmp.t())
print('两种方法结果的差 = ', (rmd1 - rmd2).norm().item())

print('--------------------- 分割线 ---------------------')
print('对量子位[3]的状态进行采样（泡利z本征态为基底，采样2048次）')
rho0 = psi.reduced_density_matrix([3])
probabilities = rho0.diag()
population = binary_strings(2)
res = Counter(choices(population, probabilities, k=2048))
print('量子位构型：', population)
print('对应的概率：', list(probabilities.to('cpu').numpy()))
print('采样结果：', res)

res1 = psi.sampling(2048, [3])
print('利用成员函数的采样结果：', res1)

print('--------------------- 分割线 ---------------------')
s0 = svdvals(psi.tensor.reshape(4, 4))
print('SVD获得的奇异值：\n', s0)
rho = psi.reduced_density_matrix([2, 3])
s1 = eigvalsh(rho)
print('约化矩阵rho[2,3]的本征值（开方）：\n',
      sqrt(s1.sort(descending=True)[0]))
print('调用成员函数获得的本征谱：\n',
      psi.bipartite_ent([0, 1]))


