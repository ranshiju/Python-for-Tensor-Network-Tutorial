import Library.MathFun as mf
import torch as tc
import numpy as np
import Library.BasicFun as bf


x = tc.randn(4, 4, dtype=tc.float64)
print(x)
print(x.to(dtype=tc.complex128))






