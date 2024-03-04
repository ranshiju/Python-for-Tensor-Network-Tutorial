import torch as tc
import Library.BasicFun as bf
from os.path import join
from Algorithms.MPS_algo import input_GMPS

'''
下述示例用到的GMPS数据可从百度云盘下载：
https://pan.baidu.com/s/1C__CSfXIs_8yJU0dYyKq1g 
提取码: TNML
'''

# GMPS储存的路径与文件名
path = '../data/GMPS_fMNIST/'
file = 'GMPS_chi64_theta1.0_FM_' \
       'cossin_class_3'

gmps = input_GMPS(join(path, file))
oee_av = gmps.EOSP_average_OEEs(clone=True)

bf.plot(tc.log10(oee_av), marker='o',
        markerfacecolor='white',
        xlabel='Measurement number',
        ylabel='log10(average OEE)',
        markersize=4,
        markeredgewidth=0.3)


