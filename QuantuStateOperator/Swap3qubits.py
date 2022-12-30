import torch as tc
import Library.MathFun as mf
from Library.QuantumState import TensorPureState


up = tc.tensor([1.0, 0.0]).to(dtype=tc.float64)
psi = TensorPureState(tc.einsum('a,b,c->abc', up, up, up))
print('初态|000>的系数张量为：\n', psi.tensor)

psi.act_single_gate(mf.hadamard(), [0])
psi.act_single_gate(mf.swap(), [0, 1])
psi.act_single_gate(mf.swap(), [1, 2])

samples = psi.sampling(n_shots=2000, position=[2])
print('末态对量子位3进行2000次采样的结果为：', samples)

v, lm = mf.rank1(psi.tensor)
err = (mf.rank1_product(v, lm) - psi.tensor).norm()
print('对末态进行Rank1分解的误差为：%g' % err.item())
print('末态的系数张量可分解为如下三个向量的直积：')
for x in v:
    print(x)

