import Library.BasicFun as bf
import Library.DataFun as df
from Algorithms.MPS_algo import generate_sample_by_gmps


'''
下述示例用到的GMPS数据可从百度云盘下载：
https://pan.baidu.com/s/1xGZsHy9plpL6-3AZsBGqZA?pwd=TNML 
提取码: TNML
'''

print('读取GMPS')
tensors, para = bf.load(
    r'../data/GMPS_MNIST/GMPS_chi64'
    r'_theta1.0_FM_cossin_digit_0',
    ['tensors', 'paraMPS'])

print('生成全新的图片')
img_g = generate_sample_by_gmps([tensors, para])
img_g = img_g.reshape(28, 28)

print('读取MNIST图片')
img0 = df.load_samples_mnist(num=1, process={'classes': [0]})
print('抹去图片中一半像素信息')
img1 = img0.reshape(-1, ).clone()
img1[392:] = 0
order = list(range(392, 784, 1))

print('单次采样生成抹去的像素')
img2 = generate_sample_by_gmps([tensors, para], img1, order)
print('多次采样生成抹去的像素（5次平均）')
img3 = generate_sample_by_gmps([tensors, para], img1, order,
                               paraG={'num_s': 5})
print('多次采样生成抹去的像素（20次平均）')
img4 = generate_sample_by_gmps([tensors, para], img1, order,
                               paraG={'num_s': 20})

print('绘图...')
imgs = [img_g, img0.squeeze(), img1.reshape(28, 28),
        img2.reshape(28, 28), img3.reshape(28, 28),
        img4.reshape(28, 28)]
titles = ['直接生成', 'MNIST原始图',  '损坏图',
          '修复图(1次平均)', '修复图(5次平均)', '修复图(20次平均)']
bf.show_multiple_images(imgs, titles=titles)
