import torch as tc
from torch.utils.data import TensorDataset, DataLoader
from Library import DataFun as df
from Library.BasicFun import choose_device
from Algorithms.ADQC_algo import ADQC_classifier
from Algorithms.NN_algo import FC2_classifier


print('设置参数')
# 通用参数
para = {'lr': 5e-4,  # 初始学习率
        'n_img': 2000,  # batch大小
        'it_time': 2000,  # 总迭代次数
        'print_time': 50,  # 打印间隔
        'dtype': tc.float64,  # 数据精度
        'device': choose_device()}  # 计算设备（cuda优先）
# FC2参数
para_nn = {'dim_h': 100}
# ADQC参数
para_qc = {'depth': 4,  # ADQC线路层数
           'lattice': 'brick',  # ADQC结构
           'feature_map': 'cossin'}  # 特征映射
para_nn = dict(para, **para_nn)
para_qc = dict(para, **para_qc)

print('读取、预处理Iris数据集')
samples, targets = df.load_iris(
    device=para['device'], dtype=para['dtype'])
samples = df.rescale_max_min_simple(samples)

print('随机划分训练、测试数据集')
length = samples.shape[1]
train_samples, train_labels, test_samples, test_labels \
    = df.split_dataset_train_test(samples, targets)
trainset = TensorDataset(train_samples, train_labels)
testset = TensorDataset(test_samples, test_labels)
trainset = DataLoader(
    trainset, para['n_img'], shuffle=True)
testset = DataLoader(
    testset, para['n_img'], shuffle=False)

print('训练FC2分类器')
_, out1, _ = FC2_classifier(trainset, testset, num_classes=3,
                            length=length, para=para_nn)
print('训练ADQC分类器')
_, out2, _ = ADQC_classifier(trainset, testset, num_classes=3,
                             length=length, para=para_qc)

print('训练完成\n 训练集精度：FC2 = %g,\t ADQC = %g' % (
    out1['train_acc'][-1], out2['train_acc'][-1]))
print(' 测试集精度：FC2 = %g,\t ADQC = %g' % (
    out1['test_acc'][-1], out2['test_acc'][-1]))
