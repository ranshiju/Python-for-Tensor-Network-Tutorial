import copy
import torch as tc
import Library.MathFun as mf
from Library.MatrixProductState \
    import MPS_basic, full_tensor


d = 3
dc = 2
x = tc.randn((d, d, d, d), dtype=tc.complex128)

tensors = mf.ttd(x)[0]
for n in range(len(tensors)):
    print('第%d个张量的维数为：' % n,
          tensors[n].shape)

psi = MPS_basic(tensors=copy.deepcopy(tensors))
psi.center = 3
psi.check_center_orthogonality(True)

tensors_2 = mf.ttd(x, dc=dc)[0]
x2 = full_tensor(tensors_2)
err = (x - x2).norm() / x.norm()
print('全局张量TTD（指标正序）误差 = %g' % err)

psi.center_orthogonalization(0)
psi.center_orthogonalization(-1, dc=dc)
y2 = psi.full_tensor()
err = (x - y2).norm() / x.norm()
print('中心正交化（从左至右裁剪）误差 = %g' % err)

tensors_2 = mf.ttd(x.permute(3, 2, 1, 0), dc=dc)[0]
x2 = full_tensor(tensors_2)
err = (x.permute(3, 2, 1, 0) - x2).norm() / x.norm()
print('全局张量TTD（指标倒序）误差 = %g' % err)

psi1 = MPS_basic(tensors=copy.deepcopy(tensors))
psi1.center_orthogonalization(0, dc=dc)
z2 = psi1.full_tensor()
err = (x - z2).norm() / x.norm()
print('中心正交化（从右至左裁剪）误差 = %g' % err)
