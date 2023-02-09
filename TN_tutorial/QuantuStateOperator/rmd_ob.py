import torch as tc
from Library.QuantumState import TensorPureState
from Library.MathFun import pauli_operators
from Library.Hamiltonians import heisenberg


print('求量子位[3]处的x方向的磁矩')
psi = TensorPureState(nq=4)  # 4-qubit random state

sigma_x = pauli_operators('x', device=psi.device)
mx = psi.observation(sigma_x, [3])

rho3 = psi.reduced_density_matrix([3])
mx_ = tc.trace(rho3.mm(sigma_x))
print('两种方法求得的磁矩 = %g, %g' % (mx.item(), mx_.item()))

print('--------------------- 分割线 ---------------------')
print('求量子位[1, 2]处的海森堡哈密顿量的能量')
hamilt = heisenberg().to(device=psi.device)
energy = psi.observation(hamilt, [1, 2])

rho12 = psi.reduced_density_matrix([1, 2])
energy_ = tc.trace(rho12.mm(hamilt))
print('两种方法求得的能量 = %g, %g' % (energy.item(), energy_.item()))

