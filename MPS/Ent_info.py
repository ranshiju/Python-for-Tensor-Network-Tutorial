import torch as tc
import numpy as np
import matplotlib.pyplot as plt


mesure_time = 600  # 随机次数
dtype = tc.complex128

print('Generate a random 2-qubit state...')
psi = tc.randn((2, 2), dtype=dtype)
psi /= psi.norm()

print('Calculate entanglement entropy...')
rho = psi.conj().mm(psi.t())
lm = tc.linalg.eigvalsh(rho)
ent = -lm.inner(tc.log(lm))

print('Implement random measurement on qubit-0...')
s = tc.zeros(mesure_time, dtype=lm.dtype)
for t in range(mesure_time):
    # 随机生成测量基
    h = tc.randn((2, 2), dtype=lm.dtype)
    proj = tc.matrix_exp(1j * (h+h.t()))
    # 计算概率分布
    p = [proj[:, s].conj().inner(rho).inner(
        proj[:, s]) for s in [0, 1]]
    # 计算纠缠熵
    s[t] = -(p[0].real * np.log(p[0].real) + p[
        1].real * np.log(p[1].real))


print('On qubit-0:\n Maximum of entropy = %g \n '
      'Minimum of entropy = %g \n '
      'Entanglement entropy = %g'
      % (max(s), min(s), ent))

print('Implement random measurement on qubit-1...')
rho = psi.t().mm(psi.conj())
s1 = tc.zeros(mesure_time, dtype=lm.dtype)
for t in range(mesure_time):
    # 随机生成测量基
    h = tc.randn((2, 2), dtype=lm.dtype)
    proj = tc.matrix_exp(1j * (h+h.t()))
    # 计算概率分布
    p = [proj[:, s].conj().inner(rho).inner(
        proj[:, s]) for s in [0, 1]]
    # 计算冯诺伊曼熵
    s1[t] = -(p[0].real * np.log(p[0].real) + p[
        1].real * np.log(p[1].real))

print('On qubit-1:\n Maximum of entropy = %g \n '
      'Minimum of entropy = %g \n '
      'Entanglement entropy = %g'
      % (max(s1), min(s1), ent))

x = np.arange(mesure_time)
fig = plt.figure()
fig.add_subplot(1, 2, 1)
ent_m = plt.scatter(x, s)
ent_f, = plt.plot(x, np.ones((mesure_time,)) * ent.item(),
                  color='r', linestyle='--')
plt.xlabel('n-th random measurement on qubit-0')
plt.ylabel('entropy')
plt.legend([ent_m, ent_f],
           ['entropy by measurement',
            'entanglement entropy'])

fig.add_subplot(1, 2, 2)
ent_m = plt.scatter(x, s1)
ent_f, = plt.plot(x, np.ones((mesure_time,)) * ent.item(),
                  color='r', linestyle='--')
plt.xlabel('n-th random measurement on qubit-1')
plt.legend([ent_m, ent_f],
           ['entropy by measurement',
            'entanglement entropy'])
plt.show()




