import Library.BasicFun as bf
import Library.DataFun as df
from Algorithms.MPS_algo import generate_sample_by_gmps


tensors, para = bf.load(r'../data/GMPS_MNIST/GMPS_chi64'
                        r'_theta1.0_FM_cossin_digit_0',
                        ['tensors', 'paraMPS'])
img_g = generate_sample_by_gmps([tensors, para])
img_g = img_g.reshape(28, 28)

img0 = df.load_samples_mnist(num=1, process={'classes': [0]})
img1 = img0.reshape(-1, ).clone()
img1[392:] = 0
order = list(range(392, 784, 1))
img2 = generate_sample_by_gmps([tensors, para], img1, order)
img3 = generate_sample_by_gmps([tensors, para], img1, order,
                               paraG={'num_s': 5})
img4 = generate_sample_by_gmps([tensors, para], img1, order,
                               paraG={'num_s': 20})

imgs = [img_g, img0.squeeze(), img1.reshape(28, 28), img2.reshape(28, 28),
        img3.reshape(28, 28), img4.reshape(28, 28)]
titles = ['直接生成', 'MNIST原始图',  '损坏图',
          '修复图(1次平均)', '修复图(5次平均)', '修复图(20次平均)']
bf.show_multiple_images(imgs, titles=titles)
