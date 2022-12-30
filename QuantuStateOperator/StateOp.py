import torch as tc
import Library.MathFun as mf
from Library.QuantumState import TensorPureState, state_ghz


x = mf.pauli_operators('x')
cnot = tc.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                  [0, 0, 0, 1], [0, 0, 1, 0]]
                 ).to(dtype=tc.float64)

ghz = state_ghz(4)

psi1 = TensorPureState(tensor=ghz.clone())
print('作用门之前为GHZ态：')
print(psi1.tensor.flatten())

print('在量子位1（控制）与2（目标）上作用完整CNOT后得：')
psi1.act_single_gate(cnot, [1, 2])
print(psi1.tensor.flatten())

print('--------------------- 分割线 ---------------------')
psi2 = TensorPureState(tensor=ghz.clone())
print('作用门之前为GHZ态：')
print(psi2.tensor.flatten())

print('以1为控制比特，2为目标比特，输入函数并作用泡利x得：')
psi2.act_single_gate(x, [2], [1])
print(psi1.tensor.flatten())
