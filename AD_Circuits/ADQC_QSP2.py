import torch as tc
import matplotlib.pylab as plt
from torch.optim import Adam
from Library import ADQC
from Library.BasicFun import choose_device
from Library.QuantumState import state_all_up


lr, it_time = 1e-3, 5000  # 初始学习率与迭代总次数
device = choose_device()

print('随机生成目标态')
num_qubits = 6
psi_target = tc.randn((2**num_qubits, ),
                      dtype=tc.complex128, device=device)
psi_target /= psi_target.norm()

print('建立ADQC_basic实例')
qc = ADQC.ADQC_LatentGates(
    lattice='brick',
    num_q=num_qubits,
    depth=2)

print('每层二体门作用的位置为：')
print(qc.pos)
print('ADQC的变分参数维数为：')
for x in qc.state_dict():
    print(qc.state_dict()[x].shape)

optimizer = Adam(qc.parameters(), lr=lr)  # 建立优化器
loss_rec = tc.zeros(it_time, )
print('开始优化')
for t in range(it_time):
    # 建立初态|000>
    psi = state_all_up(n_qubit=num_qubits, d=2).to(
        device=device, dtype=psi_target.dtype)
    psi = qc(psi)
    loss = 1 - psi.flatten().dot(psi_target.conj()).norm()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_rec[t] = loss.item()
    if t % 20 == 19:
        print('第%d次迭代，loss = %g' % (t, loss.item()))

plt.plot(loss_rec)
plt.xlabel('iteration time')
plt.ylabel('loss')
plt.show()

