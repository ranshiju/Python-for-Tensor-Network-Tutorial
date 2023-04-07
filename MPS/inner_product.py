import numpy as np
import Library.MatrixProductState as mps
import matplotlib.pyplot as plt


results = list()
for n in range(4, 80, 4):  # MPS长度变化范围
    para = {'length': n, 'd': 2, 'dc': 4}
    tmp = list()
    for m in range(50):
        # 随机建立两个归一化MPS
        mps_a = mps.MPS_basic(para=para)
        mps_a.normalize()

        mps_b = mps.MPS_basic(para=para)
        mps_b.normalize()

        # 计算MPS内积的绝对值
        tmp.append(mps_a.inner(mps_b, form='inner').abs().item())
    results.append(sum(tmp) / len(tmp))
    print('length = %d, inner (50-time average) = %e' % (
        para['length'], results[-1]))

results = np.array(results)
plt.plot(list(range(4, 80, 4)), np.log10(results), marker='o')
plt.xlabel('MPS length')
plt.ylabel('log10(Inner product)')
plt.show()


