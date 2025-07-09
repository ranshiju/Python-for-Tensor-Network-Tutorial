import os
import cv2
import math
import numpy as np
import torch as tc
from torch.utils.data import TensorDataset, DataLoader
from Library.BasicFun import choose_device, combine_dicts


def adjust_lr(optimizer, factor, cond=True):
    if cond:
        for group in optimizer.param_groups:
            group['lr'] = group['lr'] * factor
    return optimizer


def binarize_samples(samples):
    return tc.round(samples)


def choose_classes(data, labels, classes):
    assert type(classes) in [tuple, list]
    shape = list(data.shape)
    data_ = list()
    labels_ = list()
    data = data.reshape(shape[0], -1)
    for n in range(len(classes)):
        data_.append(data[labels == classes[n]])
        labels_.append(tc.ones((
            data_[-1].shape[0],), device=labels.device, dtype=labels.dtype) * n)
    data_ = tc.cat(data_, dim=0)
    labels_ = tc.cat(labels_, dim=0)
    shape[0] = labels_.numel()
    return data_.reshape(shape), labels_


def choose_classes_dataset(dataset, classes, re_tensor=False):
    train_samples, train_lbs = dataset2tensors(dataset)
    train_samples, train_lbs = choose_classes(train_samples, train_lbs, classes)
    if re_tensor:
        return train_samples, train_lbs
    else:
        return TensorDataset(train_samples, train_lbs)


def continuous_labels(classes, labels):
    labels_new = tc.zeros(labels.shape, device=labels.device, dtype=labels.dtype)
    for n in classes:
        labels_new[labels == classes[n]] = n
    return labels_new


def dataset2tensors(dataset):
    tmp = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False)
    samples, labels = next(iter(tmp))
    return samples, labels


def feature_map(samples, which='cossin',
                para=None, norm_p=2):
    if which is None:
        return samples
    which = which.lower()
    if para is None:
        para = dict()
    para_ = {
        'd': 2,  # 特征维数
        'theta': 1,  # cos-sin角度
        'alpha': 2,  # 高斯分布标准差
        'order0': 0  # 展开第0项为几阶
    }
    para = combine_dicts(para_, para)
    if samples.ndimension() == 1:
        samples = samples.reshape(1, 1, -1)
    else:
        samples = samples.reshape(-1, 1, samples[0].numel())
    if which == '1x':
        which = 'power'
        para['d'] = 2
    if (which == 'power') and (para['order0'] == 1) and (para['d'] == 1):
        which = 'reshape'

    if which in ['cossin', 'cos-sin', 'cos_sin']:
        img1 = list()
        for dd in range(1, para['d']+1):
            img1.append(math.sqrt(math.comb(
                para['d']-1, dd-1)) * tc.cos(
                samples * para['theta'] * np.pi / 2
            ) ** (para['d'] - dd) * tc.sin(
                samples * para['theta'] * np.pi / 2
            ) ** (dd - 1))
        img1 = tc.cat(img1, dim=1)
        if norm_p == 1:
            img1 = img1 ** 2
        return img1
    elif which == 'linear':
        return tc.cat([samples, 1 - samples], dim=1)
    elif which == 'gaussian':
        return feature_map_gaussian_discretization(
            samples, norm_p=norm_p, d=para['d'], alpha=para['alpha'])
    elif which in ['square-linear', 'square_linear', 'squarelinear',
                   'normalized-linear', 'normalized_linear', 'normalizedlinear']:
        img1 = tc.cat([tc.sqrt(tc.abs(samples)), tc.sqrt(1 - tc.abs(samples))], dim=1)
        if norm_p == 1:
            img1 = img1 ** 2
        return img1
    elif which in ['one-hot', 'one_hot', 'onehot']:
        return feature_map_one_hot(samples, d=para['d'])
    elif which == 'power':
        if para['order0'] == 0:
            img_list = [tc.ones(
                samples.shape, device=samples.device,
                dtype=samples.dtype)]
            order0, order1 = 1, para['d']
        else:
            img_list = list()
            order0, order1 = \
                para['order0'], para['order0']+para['d']
        for dd in range(order0, order1):
            img_list.append(samples ** dd)
        return tc.cat(img_list, dim=1)
    elif which == 'reshape':
        return samples.reshape(samples.shape[0], 1, -1)
    else:
        print('Error: ' + which + ' is not a valid feature map')


def feature_map_gaussian_discretization(samples, d, alpha=5, norm_p=1):
    # Feature取值在0到1间（含）
    x = tc.linspace(1/d/2, 1-1/d/2, d).to(device=samples.device, dtype=samples.dtype)
    print(x)
    samples_ = samples.reshape(samples.shape[0], 1, -1)
    s_list = list()
    for n in range(d):
        s_list.append(tc.exp(-alpha * (samples_ - x[n]) ** 2))
    s_list = tc.cat(s_list, dim=1)
    norms = s_list.norm(dim=1, p=norm_p)
    s_list = tc.einsum('nab,nb->nab', s_list, 1/norms)
    return s_list


def feature_map_one_hot(samples, d, eps=1e-10):
    # Feature取值在0到1间（含）
    x = tc.linspace(1/d, 1, d).to(device=samples.device, dtype=samples.dtype)
    x[-1] += eps
    samples_ = samples.reshape(samples.shape[0], 1, -1)
    samples1 = tc.zeros(samples_.shape, device=samples.device, dtype=samples.dtype)
    samples1[samples_ <= x[0]] = 1.0
    s_list = [samples1]
    for n in range(1, d):
        samples1 = tc.zeros(samples_.shape, device=samples.device, dtype=samples.dtype)
        samples1[samples_ <= x[n]] = 1.0
        samples1[samples_ <= x[n-1]] = 0.0
        s_list.append(samples1)
    s_list = tc.cat(s_list, dim=1)
    return s_list


def get_batch_from_loader(loader, which, only_sample=False):
    for n, x in enumerate(loader):
        if n == which:
            if only_sample:
                return x[0]
            else:
                return x


def labels_rearrange(labels):
    labels1 = labels.clone()
    numbers = set(list(labels.reshape(-1, ).numpy()))
    numbers = sorted(list(numbers))
    for x in numbers:
        labels1[labels == x] = numbers.index(x)
    return labels1


def load_cifar10(which=10, dataset_path=None, test=True, process=None):
    from torchvision import datasets, transforms
    preprocess = [transforms.ToTensor()]
    if process is None:
        process = dict()
    if 'crop' in process:
        preprocess.append(transforms.CenterCrop(size=process['crop']))
    if 'resize' in process:
        preprocess.append(transforms.Resize(size=process['resize']))
    data_tf = transforms.Compose(preprocess)
    if dataset_path is None:
        paths = ['./Datasets', '../Datasets', '../../Datasets', '../../../Datasets', '../../../../Datasets']
        for x in paths:
            if os.path.isdir(x):
                dataset_path = x
    if dataset_path is None:
        dataset_path = './Datasets'
    test_dataset = None
    if which == 10:
        train_dataset = datasets.CIFAR10(
            root=dataset_path, train=True, transform=data_tf, download=True)
        if test:
            test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=data_tf)
    else:
        train_dataset = datasets.CIFAR100(
            root=dataset_path, train=True, transform=data_tf, download=True)
        if test:
            test_dataset = datasets.FashionMNIST(root=dataset_path, train=False, transform=data_tf)
    return train_dataset, test_dataset


def load_iris(return_dict=False, return_tensor=True, device=None, dtype=tc.float64):
    from sklearn import datasets
    iris = datasets.load_iris()
    if return_tensor:
        device = choose_device(device)
        iris['sample'] = tc.from_numpy(iris['sample']).to(device=device, dtype=dtype)
        iris['target'] = tc.from_numpy(iris['target']).to(device=device, dtype=tc.int64)
    if return_dict:
        return iris
    else:
        samples = iris['sample']
        targets = iris['target']
        return samples, targets


def load_samples_mnist(which='mnist', num=1, pos=None, dataset_path=None, test=True, process=None):
    trainset, testset = load_mnist(
        which=which, dataset_path=dataset_path, test=test, process=process)
    if test:
        samples = dataset2tensors(testset)[0]
    else:
        samples = dataset2tensors(trainset)[0]
    if type(pos) in [tuple, list]:
        assert num == len(pos)
        return samples[pos]
    elif pos in ['random', None]:
        ind = tc.randperm(samples.shape[0])[:num]
        return samples[ind]
    elif pos == 'first':
        return samples[:num]
    elif pos == 'last':
        return samples[samples.shape[0]-num:]
    elif pos == 'pos':
        return samples[num]


def load_mnist(which='mnist', dataset_path=None, test=True, process=None):
    from torchvision import datasets, transforms
    preprocess = [transforms.ToTensor()]
    if process is None:
        process = dict()
    if 'crop' in process:
        preprocess.append(transforms.CenterCrop(size=process['crop']))
    if 'resize' in process:
        preprocess.append(transforms.Resize(size=process['resize']))
    if 'normalize' in process:
        preprocess.append(transforms.Normalize(
            mean=process['normalize'][0], std=process['normalize'][1]))

    data_tf = transforms.Compose(preprocess)
    if dataset_path is None:
        paths = ['./Datasets', '../Datasets', '../../Datasets', '../../../Datasets', '../../../../Datasets']
        for x in paths:
            if os.path.isdir(x):
                dataset_path = x
                break
    if dataset_path is None:
        dataset_path = './Datasets'
    test_dataset = None
    if which.lower() == 'mnist':
        train_dataset = datasets.MNIST(
            root=dataset_path, train=True, transform=data_tf, download=True)
        if test:
            test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=data_tf)
    else:
        train_dataset = datasets.FashionMNIST(
            root=dataset_path, train=True, transform=data_tf, download=True)
        if test:
            test_dataset = datasets.FashionMNIST(root=dataset_path, train=False, transform=data_tf)

    if 'classes' in process:
        if type(process['classes']) is int:
            process['classes'] = list(range(process['classes']))
        train_dataset = choose_classes_dataset(train_dataset, process['classes'])
        if test:
            test_dataset = choose_classes_dataset(test_dataset, process['classes'])
    if 'return_tensor' in process and process['return_tensor']:
        train_samples, train_lbs = dataset2tensors(train_dataset)
        if test:
            test_samples, test_lbs = dataset2tensors(test_dataset)
            return (train_samples, train_lbs), (test_samples, test_lbs)
        else:
            return (train_samples, train_lbs), None
    else:
        return train_dataset, test_dataset


def make_dataloader(dataset, batch_size=None, shuffle=False):
    from torch.utils.data import DataLoader
    if type(dataset) in [list, tuple]:
        # dataset = [samples, labels]
        dataset = TensorDataset(dataset[0], dataset[1])
    if batch_size is None:
        batch_size = dataset.data.shape[0]
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def one_hot_labels(labels, device=None, dtype=None):
    if device is None:
        device = labels.device
    if dtype is None:
        dtype = tc.float64
    labels_value = sorted(list(set(list(labels.reshape(-1, ).numpy()))))
    num_c = len(labels_value)
    labels_v = tc.zeros((labels.shape[0], num_c), device=device, dtype=dtype)
    for n in range(labels.shape[0]):
        labels_v[n, labels_value.index(labels[n])] = 1.0
    return labels_v


def preprocess_image(image, preprocess_means):
    if 'cut' in preprocess_means:
        lx = min(preprocess_means['cut'][0], image.shape[0])
        ly = min(preprocess_means['cut'][1], image.shape[1])
        lx0 = int((image.shape[0] - lx) / 2)
        ly0 = int((image.shape[1] - ly) / 2)
        image = image[lx0:lx0 + lx, ly0:ly0 + ly, :]
    if 'size' in preprocess_means:
        if image.shape[0] > preprocess_means['size'][0] or image.shape[1] > preprocess_means['size'][1]:
            image = cv2.resize(image, preprocess_means['size'])
    return image


def rescale_max_min_simple(samples, maximum=1, minimum=0):
    s = samples.shape
    samples_ = samples - samples.min(dim=0, keepdim=True)[0]
    samples_ = samples_ / samples_.max(dim=0, keepdim=True)[0]
    samples_ = samples_ * (maximum - minimum) + minimum
    return samples_.reshape(s)


def rescale_max_min_sample_wise(samples, maximum=1, minimum=0):
    s = samples.shape
    samples_ = samples.reshape(samples.shape[0], -1)
    samples_ = samples_ - samples_.min(dim=1)[0].repeat(samples_.shape[1], 1).permute(1, 0)
    samples_max = samples_.max(dim=1)[0].repeat(samples_.shape[1], 1).permute(1, 0)
    samples_ = samples_ / samples_max
    samples_ = samples_ * (maximum - minimum) + minimum
    return samples_.reshape(s)


def rescale_max_min_feature_wise(samples, maximum=1, minimum=0):
    s = samples.shape
    samples_ = samples.reshape(samples.shape[0], -1)
    samples_ = samples_ - samples_.min(dim=0)[0].repeat(samples_.shape[0], 1)
    samples_max = samples_.max(dim=0)[0].repeat(samples_.shape[0], 1)
    samples_ = samples_ / samples_max
    samples_ = samples_ * (maximum - minimum) + minimum
    return samples_.reshape(s)


def select_num_samples(samples, labels, num, classes=None):
    if classes is None:
        classes = list(set(labels.numpy()))
    labels_new = list()
    samples_new = list()
    for c in classes:
        samples1 = samples[labels == c]
        if samples1.shape[0] > num:
            ind = tc.randperm(samples1.shape[0])[:num]
            samples1 = samples1[ind]
        samples_new.append(samples1)
        labels_new.append(tc.ones((
            samples1.shape[0], ), device=labels.device, dtype=labels.dtype) * c)
    return tc.cat(samples_new, dim=0), tc.cat(labels_new, dim=0)


def split_time_series(data, length, device=None, dtype=tc.float32):
    """
    利用length长度的时序数据预测第length+1位置的数据
    :param data: 一维时序数据
    :param length: 样本长度
    :param device: 计算设备
    :param dtype: 数据精度
    :return: N * length维的样本矩阵，N维的标签向量
    """
    samples, targets = list(), list()
    device = choose_device(device)
    for n in range(length, data.numel()):
        samples.append(data[n-length:n].clone().reshape(1, -1))
        targets.append(data[n].clone())
    return tc.cat(samples, dim=0).to(
        device=device, dtype=dtype), tc.tensor(targets).to(device=device, dtype=dtype)


def split_dataset_train_test(samples, labels):
    num_c = tc.max(labels.flatten()) + 1
    train_samples, test_samples = list(), list()
    train_labels, test_labels = list(), list()
    for n in range(num_c):
        train_, test_ = split_samples(samples[labels == n])
        train_samples.append(train_)
        test_samples.append(test_)
        train_labels.append(n*tc.ones(train_.shape[0], dtype=tc.int64, device=labels.device))
        test_labels.append(n*tc.ones(test_.shape[0], dtype=tc.int64, device=labels.device))
    train_samples = tc.cat(train_samples, dim=0)
    train_labels = tc.cat(train_labels, dim=0)
    test_samples = tc.cat(test_samples, dim=0)
    test_labels = tc.cat(test_labels, dim=0)
    return train_samples, train_labels, test_samples, test_labels


def split_samples(samples, ratio=0.2, shuffle=True):
    num_train = int(samples.shape[0] * ratio)
    if shuffle:
        rand = tc.randperm(samples.shape[0])
        train_samples = samples[rand[:num_train]]
        test_samples = samples[rand[num_train:]]
    else:
        train_samples = samples[:num_train]
        test_samples = samples[num_train:]
    return train_samples, test_samples

