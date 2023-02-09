import torch as tc

A = tc.randn((2, 2, 2), dtype=tc.float64)
B = tc.randn((2, 2, 2), dtype=tc.float64)
C = tc.randn((2, 2, 2), dtype=tc.float64)

X = tc.einsum('iaj,jbk,kci->abc', A, B, C)

tmp_a = A.reshape(4, 2)
tmp_b = B.reshape(2, 4)
tmp = tmp_a.mm(tmp_b).reshape(2, 2, 2, 2)  # 指标为i,a,b,k
tmp = tmp.permute(1, 2, 3, 0).reshape(4, 4)
tmp_c = C.permute(0, 2, 1).reshape(4, 2)
X1 = tmp.mm(tmp_c).reshape(2, 2, 2)

print('X - X1 = \n', X - X1)

print('--------------------- 分割线 ---------------------')
X2 = tc.zeros((2, 2, 2), dtype=tc.float64)
for n1 in range(2):
    for n2 in range(2):
        for n3 in range(2):
            X2[n1, n2, n3] += (A[:, n1, :].mm(B[:, n2, :]).mm(
                C[:, n3, :])).trace()
print('X - X2 = \n', X - X2)
