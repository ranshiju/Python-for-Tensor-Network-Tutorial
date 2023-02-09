import torch as tc
import matplotlib.pylab as plt
from torch.optim import Adam
from Library import ADQC
from Library.BasicFun import choose_device
from Library.QuantumState import state_all_up
from Library.Hamiltonians import heisenberg
from Library.ED import ED_ground_state


print('设置必要参数')
num_qubits = 6  # 自旋个数
tau = 0.02  # Trotter切片宽度
num_slice = 20  # 切片次数
J_target = [1, 1, 1]  # 制备该哈密顿量（海森堡模型）基态
J_evolve = [1, 1, 0]  # 时间演化海森堡模型耦合常数（取XY部分）
lr, it_time = 1e-2, 2000  # 初始学习率与迭代总次数
device = choose_device()

print('获得目标态：海森堡模型基态')
hamilt_t = heisenberg(J_target[0], J_target[1], J_target[2])
pos = [[n, n+1] for n in range(num_qubits-1)]
psi_target = ED_ground_state(
    hamilt_t.reshape(2, 2, 2, 2), pos)[1]
psi_target = tc.from_numpy(psi_target).flatten().to(
    device=device, dtype=tc.complex128)

print('采用XY相互作用')
hamilt = heisenberg(J_evolve[0], J_evolve[1], J_evolve[2])

print('建立ADQC_time_evolution_chain实例')
qc = ADQC.ADQC_time_evolution_chain(
    hamilt=hamilt, length=num_qubits,
    tau=tau, num_slice=num_slice)

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
        print('第%d次迭代，loss = %g' % (t+1, loss.item()))

print('绘制结果')
plt.plot(loss_rec)
plt.xlabel('iteration time')
plt.ylabel('loss')
plt.show()

fields_optimal = qc.cat_fields()
plt.imshow(fields_optimal.cpu().numpy())
plt.show()


