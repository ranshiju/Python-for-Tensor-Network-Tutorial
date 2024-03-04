import torch as tc
import numpy as np
from Library.DataFun import feature_map
from Library.QuantumTools import vecs2product_state
from Library.QuantumState import TensorPureState


def dOEE_analyze(samples, img_n=0):
    samples = tc.tensor(samples).to(dtype=tc.float64)
    vecs = feature_map(samples)
    psi = vecs2product_state(vecs)
    print('psi形状为：直积态个数 * (直积态维数) = ', psi.shape)

    print('转为TensorPureState类...')
    psi = TensorPureState(psi.sum(dim=0)/np.sqrt(psi.shape[0]))
    OEE = psi.onsite_ent_entropy()
    print('OEE = ', OEE.numpy())

    OEEsum = OEE.sum()
    for n in range(vecs.shape[2]):
        print('- 根据第%i个样本的特征取值 (x=%g)，投影测量第%i个量子位'
              % (img_n, samples[img_n, n], n))
        psi1 = psi.project(vecs[img_n, :, n], n,
                           update_state=False)
        OEE1 = psi1.onsite_ent_entropy()
        print('\t 测量后，其余量子位OEE = ', OEE1.numpy())
        dOEE = OEE1.sum() - OEEsum
        print('\t 测量前后OEE变化量 = ', dOEE.item())


print('Case 1: ')
dOEE_analyze([[0, 0, 1], [0, 1, 0]], 1)
print('\nCase 2: ')
dOEE_analyze([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], 1)
