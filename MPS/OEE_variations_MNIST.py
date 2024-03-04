import os, time
import Library.BasicFun as bf
import Library.MatrixProductState as mps
from Library.DataFun import load_mnist, feature_map
import Algorithms.wheels_gmps as wg


gmps_path = r'../data/GMPS_MNIST/'
gmps_file = 'GMPS_chi64_theta1.0_FM_cossin_digit_0'
sample_index = 2

sample, _ = load_mnist(
    'mnist', test=False,
    process={'classes': [0], 'return_tensor': True})[0]
sample = sample[sample_index].reshape(1, -1)

tensors, paraMPS = bf.load(os.path.join(gmps_path, gmps_file),
                           ['tensors', 'paraMPS'])
gmps = mps.generative_MPS(tensors, paraMPS)
gmps.correct_device()
img_vecs = feature_map(
    sample.to(device=gmps.device, dtype=gmps.dtype),
    paraMPS['feature_map'],
    {'d': paraMPS['d'], 'theta': paraMPS['theta']})

t0 = time.time()
dOEE = wg.OEE_variation_one_qubit_measurement_simple(
    gmps, img_vecs.squeeze())
print('Time cost: %i' % (time.time() - t0))
t0 = time.time()
dOEE1 = wg.OEE_variation_one_qubit_measurement(
    gmps, img_vecs.squeeze(), OEE_eps=1e-2)
print('Time cost: %i' % (time.time() - t0))
bf.show_multiple_images(
    [sample.reshape(28, 28), dOEE.reshape(28, 28),
     dOEE1.reshape(28, 28)], lxy=[1, 3])
