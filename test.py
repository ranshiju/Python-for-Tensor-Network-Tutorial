import Library.MathFun as mf
import torch as tc
import numpy as np
from matplotlib import pyplot
import Library.BasicFun as bf
import math, time
from random import choices


from matplotlib.font_manager import FontManager
import subprocess

mpl_fonts = set(f.name for f in FontManager().ttflist)

print('all font list get from matplotlib.font_manager:')
for f in sorted(mpl_fonts):
    print('\t' + f)