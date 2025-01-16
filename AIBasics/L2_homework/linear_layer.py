import torch as tc
from torch import nn
from collections import OrderedDict


vecs = tc.randn(10, 2)
f = nn.Linear(2, 1, bias=False)
para = {'weight': tc.tensor([[0.5, 0.5]])}
f.load_state_dict(OrderedDict(para))
out = f(vecs)

print(vecs)
print(out.data.view(-1))
print(vecs.sum(dim=1) / 2)

