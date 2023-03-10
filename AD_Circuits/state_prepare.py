import torch as tc
import matplotlib.pylab as plt
import Library.QuantumState as qs
import Library.MathFun as mf
from torch.optim import Adam


lr, it_time = 1e-3, 5000  # 初始学习率与迭代总次数

print('随机生成目标态')
num_qubits = 3
psi_target = tc.randn((2**num_qubits, ), dtype=tc.complex128)
psi_target /= psi_target.norm()

print('初始化变分参数并建立优化器')
paras = tc.randn((num_qubits*2, 4), dtype=tc.float64,
                 requires_grad=True)
optimizer = Adam([paras], lr=lr)

print('建立非参数门')
sigma_x = mf.pauli_operators('x')
loss_rec = tc.zeros(it_time, )

print('开始优化')
for t in range(it_time):
    gates = list()
    for n in range(2*num_qubits):  # 将变分参数传输给旋转门
        gates.append(mf.rotate(paras[n, :]))
    # 建立初态|000>
    psi = qs.state_all_up(n_qubit=num_qubits, d=2)
    psi = qs.TensorPureState(psi)
    # 作用各个量子门
    psi.act_single_gate(gates[0], [0])
    psi.act_single_gate(sigma_x, [1], [0])
    psi.act_single_gate(gates[1], [0])
    psi.act_single_gate(gates[2], [1])
    psi.act_single_gate(sigma_x, [2], [1])
    psi.act_single_gate(gates[3], [1])
    psi.act_single_gate(gates[4], [2])
    psi.act_single_gate(sigma_x, [0], [2])
    psi.act_single_gate(gates[5], [2])
    # 计算loss并作反向传播与梯度优化
    loss = 1 - psi.tensor.conj().flatten().dot(
        psi_target).norm()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_rec[t] = loss.item()
    if t % 20 == 19:
        print('第%d次迭代后，loss = %g' % (t+1, loss.item()))

plt.plot(loss_rec)
plt.xlabel('iteration time')
plt.ylabel('loss')
plt.show()


