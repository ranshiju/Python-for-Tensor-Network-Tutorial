import torch as tc
import Library.MatrixProductState as mps
from Library.MathFun import reduced_matrix


para = {'length': 8,
        'd': 3,
        'chi': 4,
        'dtype': tc.complex128}
psi = mps.MPS_basic(para=para)
print('随机建立MPS，其不具备正交性')
print('self.center = %d' % psi.center)
x1 = psi.full_tensor()

psi.center_orthogonalization(c=2)
print('中心正交化，使其中心位于2号张量')
print('self.center = %d' % psi.center)
x2 = psi.full_tensor()

print('中心正交化前后MPS全局张量之差为：')
print((x1 - x2).norm().item())

print('检查各个张量的正交性：')
psi.check_center_orthogonality(prt=True)

print('--------------------- 分割线 ---------------------')
print('全局张量的模 = %g' % x1.norm())
print('中心张量的模 = %g' % psi.tensors[2].norm())

print(x1.shape, psi.tensors[2].shape)
s0 = tc.linalg.svdvals(x1.reshape(para['d']**2, -1))
print('全局张量奇异谱：', s0)
s1 = tc.linalg.svdvals(psi.tensors[2].reshape(psi.tensors[2].shape[0], -1))
print('中心张量奇异谱：', s1)
s2 = psi.bipartite_entanglement(nt=1)
print('成员函数所得奇异谱：', s2)

print('--------------------- 分割线 ---------------------')
nt = 1
rho0 = psi.one_body_RDM(nt)
rho0_ = reduced_matrix(x1, nt)
print('MPS的%d号位的约化密度矩阵：' % nt)
print(rho0)
print('全局张量指标%d的约化密度矩阵：' % nt)
print(rho0_ / rho0_.trace())
