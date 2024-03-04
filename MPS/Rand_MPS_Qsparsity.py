import torch as tc
import statistics
import matplotlib.pyplot as plt
from Library import MatrixProductState as MPS


length = 20
av_time = 10
chi = tc.arange(2, 26, 2)

para = {'length': length}
qs_num_av = list()
std = list()
for n in range(chi.numel()):
    print('With chi = %i' % chi[n])
    qs_num = list()
    for t in range(av_time):
        mps = MPS.MPS_basic(
            para={'length': length, 'chi': chi[n]})
        mps.center_orthogonalization(
            0, normalize=True)
        mps.Q_sparsity(clone=False)
        qs_num.append(mps.qs_number.item())
    std.append(statistics.stdev(qs_num))
    qs_num_av.append(sum(qs_num) / av_time)
    print('QS number = %g' % (qs_num_av[-1]))
plt.errorbar(chi, qs_num_av, yerr=std, capsize=3)
plt.xlabel('chi')
plt.ylabel('QS number')
plt.show()





