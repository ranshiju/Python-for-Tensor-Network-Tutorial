import Library.MathFun as mf
import torch as tc
import numpy as np
import Library.BasicFun as bf
import math, time


d = 20000
dtype = tc.float32

dev = bf.choose_device()
print(dev)

total = time.time()
for n in range(30):
    t = time.time()
    for m in range(10):
        x = tc.randn((d, d), device=dev, dtype=dtype)
        y = x.mm(x).mm(x).mm(x) / x.norm()
    print('time cost = %9g' % (time.time() - t))
print('Total time = %.9g' % (time.time() - total))
