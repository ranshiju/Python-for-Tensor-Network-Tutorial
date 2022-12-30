import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from Library.Hamiltonians import heisenberg
from Library.MathFun import kron
from Library.ED import ED_ground_state


delta = 1
print('各向异性参数 = %g' % delta)

print('测试三种方式计算长度为4的海森堡链基态能量')
hamilt = heisenberg(jz=delta).numpy()
identity = np.eye(2)
heisenberg = kron([hamilt, identity, identity])
heisenberg += kron([identity, hamilt, identity])
heisenberg += kron([identity, identity, hamilt])
ge1, gs1 = eigsh(heisenberg, k=1, which='SA')
print('\t 获取完整哈密顿量后调用eigsh, \t 基态能量 = %g ' % ge1[0])


# ====================================================
def heisenberg_hamilt_on_psi(psi, h):
    psi = psi.reshape(2, 2, 2, 2)
    psi1 = np.einsum('abcd,abij->ijcd', psi, h)
    psi1 += np.einsum('abcd,bcij->aijd', psi, h)
    psi1 += np.einsum('abcd,cdij->abij', psi, h)
    return psi1.flatten()


linear_fun = LinearOperator((16, 16), lambda x: heisenberg_hamilt_on_psi(
    x, hamilt.reshape(2, 2, 2, 2)))
ge2, gs2 = eigsh(linear_fun, k=1, which='SA')
print('\t 匿名函数 + LinearOperator, \t 基态能量 = %g ' % ge2[0])
# ====================================================

ge3, gs3 = ED_ground_state(hamilt.reshape(2, 2, 2, 2),
                           pos=[[0, 1], [1, 2], [2, 3]])
print('\t 直接调用Library.ED.ED_ground_state, \t 基态能量 = %g '
      % ge1[0])
