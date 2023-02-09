import torch as tc


M1 = tc.randn((3, 2), dtype=tc.complex128)
u, s, v = tc.linalg.svd(M1)
print('The shapes of u, s, and v_dagger with '
      'full_matrices=True (default):')
print(u.shape, s.shape, v.shape)

M1 = tc.randn((3, 2), dtype=tc.complex128)
u, s, v = tc.linalg.svd(M1, full_matrices=False)
print('The shapes of u, s, and v_dagger with '
      'full_matrices=False:')
print(u.shape, s.shape, v.shape)

print('--------------------- 分割线 ---------------------')
d1, d2 = 3, 4
M2 = tc.einsum('a,b->ab',
               tc.randn(d1, dtype=tc.complex128),
               tc.randn(d2, dtype=tc.complex128))
s = tc.linalg.svdvals(M2)
print('The singular values of a matrix from the outer product '
      'of two vectors:')
print(s)
