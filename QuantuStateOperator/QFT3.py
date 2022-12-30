import torch as tc
import Library.MathFun as mf
from numpy import pi, sqrt
from Library.QuantumState import TensorPureState
from Library.QuantumTools import act_N_qubit_QFT


print('初态为|000>')
up = tc.tensor([1.0, 0.0]).to(dtype=tc.float64)
psi = TensorPureState(tc.einsum('a,b,c->abc', up, up, up))

psi.act_single_gate(mf.hadamard(), [0])
psi.act_single_gate(mf.phase_shift(pi/2), [0], [1])
psi.act_single_gate(mf.phase_shift(pi/4), [0], [2])
psi.act_single_gate(mf.hadamard(), [1])
psi.act_single_gate(mf.phase_shift(pi/2), [1], [2])
psi.act_single_gate(mf.hadamard(), [2])
print('3比特QFT变换后的末态为：')
print(psi.tensor)

psi1 = TensorPureState(tc.einsum('a,b,c->abc', up, up, up))
psi1 = act_N_qubit_QFT(psi1)
print('使用函数\'act_N_qubit_QFT\'的结果为：')
print(psi1.tensor)

up_down = tc.tensor([1/sqrt(2), 1/sqrt(2)])
psi2 = tc.einsum('a,b,c->abc', up_down, up_down, up_down)
print('理论结果：')
print(psi2)
