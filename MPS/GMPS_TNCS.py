import torch as tc
import Library.BasicFun as bf
import Library.DataFun as df
from os.path import join
from Algorithms.MPS_algo import \
    generate_sample_by_gmps, input_GMPS

'''
下述示例用到的GMPS数据可从百度云盘下载：
https://pan.baidu.com/s/1C__CSfXIs_8yJU0dYyKq1g 
提取码: TNML
'''

category = 3  # 选择某一类
which_img = 13  # 选取特定图片进行测试
# GMPS储存的路径与文件名
path = '../data/GMPS_fMNIST/'
file = 'GMPS_chi64_theta1.0_FM_' \
       'cossin_class_' + str(category)
num_sp = 1  # 生成每个像素时的采样次数
# 已知特征数量
nf = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 100]

print('Load fashion-MNIST dataset and image '
      '(#%i)...' % which_img)
imgList0, imgList1 = list(), list()
imgList2, imgList3 = list(), list()

img0 = df.load_samples_mnist(
    'fmnist', num=which_img, pos='pos',
    process={'classes': [category]})
img0 = img0.reshape(-1, )

print('Load GMPS...')
gmps = input_GMPS(join(path, file))

print('Calculate OEE and ordering...')
OEE = gmps.entanglement_ent_onsite()
oee_order = tc.argsort(OEE, descending=True)

print('Calculate ESOP ordering...')
gmps.EOSP(clone=True)
gmps.save_properties(path, file)
print('Q-sparsity number = %g' % gmps.qs_number)

print('Generate a new image...')
imgRC0 = generate_sample_by_gmps(
    gmps.clone_gmps(),
    paraG={'way': 'inverse', 'num_s': num_sp})

bf.show_multiple_images(
    [img0.reshape(28, 28),
     tc.argsort(oee_order).reshape(28, 28).cpu(),
     tc.argsort(gmps.eosp_ordering).reshape(28, 28).cpu(),
     imgRC0.cpu().reshape(28, 28)], lxy=(1, 4))

for num_f in nf:
    print('Recover with %i known pixel(s)...' % num_f)
    print('Selecting by OEE...')

    img1 = img0.clone()
    oee_order1 = bf.supplementary(
        tc.arange(784), oee_order[:num_f])
    oee_order1 = sorted(list(oee_order1))
    img1[oee_order1] = 0.0
    imgList0.append(img1.reshape(28, 28))
    img_RC = generate_sample_by_gmps(
        gmps.clone_gmps(), img1, oee_order1,
        paraG={'way': 'inverse', 'num_s': num_sp})
    imgList1.append(img_RC.reshape(28, 28))

    print('Selecting by ESOP...')
    img1 = img0.clone()
    eosp_order1 = bf.supplementary(
        tc.arange(784), gmps.eosp_ordering[:num_f])
    eosp_order1 = list(eosp_order1)
    img1[eosp_order1] = 0.0
    imgList2.append(img1.reshape(28, 28))
    img_RC = generate_sample_by_gmps(
        gmps.clone_gmps(), img1, eosp_order1,
        paraG={'way': 'inverse', 'num_s': num_sp})
    imgList3.append(img_RC.reshape(28, 28))

bf.show_multiple_images(
    imgList0+imgList1+imgList2+imgList3, lxy=(4, -1))




