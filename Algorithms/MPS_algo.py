import os
import copy
import torch as tc
import numpy as np
import Algorithms.wheels_gmps as wf
from torch import nn, optim
from Library import DataFun as df
from Library.BasicFun import print_dict, fprint, choose_device, save, load
from Library.MatrixProductState import \
    generative_MPS, ResMPS_basic, activated_ResMPS


def ResMPS_classifier(train_loader, test_loader,
                      para=None, paraMPS=None):
    if para is None:
        para = dict()
    if paraMPS is None:
        paraMPS = dict()
    para0 = {
        'ResMPS': 'simple',  # ResMPS种类
        'feature_map': 'reshape',  # 特征映射
        'it_time': 200,  # 优化迭代次数
        'lr': 1e-3,  # 学习率
        'dtype': tc.float32,
        'device': None
    }
    # ResMPS参数
    paraMPS0 = {
        'classes': 10,  # 类别数
        'pos_c': 'mid',  # 分类指标位置
        'length': 784,  # MPS长度（特征数）
        'd': 1,  # 特征映射维数
        'chi': 40,  # 虚拟维数
        'bias': False,  # 是否加偏置项
        'bias_fc': False,  # 线性层是否加bias
        'eps': 1e-6,  # 扰动大小
        'dropout': 0.2,  # dropout概率（None为不dropout）
        'activation': None
    }
    para = dict(para0, **para)
    paraMPS = dict(paraMPS0, **paraMPS)
    paraMPS['device'] = choose_device(para['device'])
    paraMPS['dtype'] = para['dtype']

    if para['feature_map'] == 'reshape':
        paraMPS['d'] = 1
    elif para['feature_map'] == 'linear':
        paraMPS['d'] = 2

    if para['ResMPS'].lower() in ['simple', 'basic']:
        mps = ResMPS_basic(para=paraMPS)
    else:
        mps = activated_ResMPS(para=paraMPS)

    print('算法参数为：')
    print_dict(para)
    print('ResMPS模型为：' + mps.name)
    print('ResMPS的超参数为：')
    print_dict(mps.para)

    optimizer = optim.Adam(mps.parameters(), lr=para['lr'])
    criterion = nn.CrossEntropyLoss()

    train_acc, test_acc = list(), list()
    for t in range(para['it_time']):
        print('\n训练epoch: %d' % t)
        mps.train()
        loss_rec, num_c, total = 0.0, 0.0, 0.0
        for nb, (img, lb) in enumerate(train_loader):
            vecs = df.feature_map(img.to(
                device=mps.device, dtype=mps.dtype),
                which=para['feature_map'],
                para={'d': paraMPS['d']})
            lb = lb.to(device=mps.device)
            optimizer.zero_grad()
            y = mps(vecs)
            loss = criterion(y, lb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_rec += loss.item()
            predicted = y.argmax(dim=1)
            total += lb.shape[0]
            num_c += predicted.eq(lb).sum().item()
            if (nb % 20 == 0) or (nb == len(train_loader) - 1):
                print('%i个batch已训练; loss: %g, acc: %g%%' %
                      ((nb+1), loss_rec/(nb+1), 100*num_c/total))
        train_acc.append(num_c/total)

        test_loss, num_c, total = 0.0, 0.0, 0.0
        mps.eval()
        print('测试集:')
        with tc.no_grad():
            for nt, (imgt, lbt) in enumerate(test_loader):
                vecs = df.feature_map(imgt.to(
                    device=mps.device, dtype=mps.dtype),
                    which=para['feature_map'],
                    para={'d': paraMPS['d']})
                lbt = lbt.to(device=mps.device)
                yt = mps(vecs)
                loss = criterion(yt.data, lbt)
                test_loss += loss.item()
                predicted = yt.argmax(dim=1)
                total += lbt.shape[0]
                num_c += predicted.eq(lbt).sum().item()
            print('loss: %g, acc: %g%% '
                  % (test_loss/(nt+1), 100*num_c/total))
            test_acc.append(num_c / total)
    info = {
        'train_acc': tc.tensor(train_acc),
        'test_acc': tc.tensor(test_acc)
    }
    return info, mps


def GMPS_train(samples, tensors=None, para=None, paraMPS=None):
    """
    samples.shape = (num of samples_v, num of features)
    tensors = list of tensors of MPS
    NOTE:
        The number of features equals to the length of MPS
    """

    if para is None:
        para = dict()
    para0 = dict()
    para0['lr'] = 0.1  # learning rate
    para0['lr_decay'] = 0.9  # decaying of learning rate
    para0['sweepTime'] = 100  # sweep time

    para0['isSave'] = True  # whether save MPS
    para0['save_dir'] = './'  # where to save MPS
    para0['save_name'] = None  # saving file name
    para0['save_dt'] = 10  # the frequency to save
    para0['record'] = 'record.log'  # name of log file
    para0['device'] = choose_device()  # cpu or gpu
    para0['dtype'] = tc.float64
    para = dict(para0, **para)

    if paraMPS is None:
        paraMPS = dict()
    paraMPS0 = {
        'length': 784,
        'd': 2,
        'chi': 3,
        'boundary': 'open',
        'feature_map': 'cossin',
        'eps': 1e-14,
        'theta': 1.0,
        'device': para['device'],
        'dtype': para['dtype']
    }
    paraMPS = dict(paraMPS0, **paraMPS)
    samples = df.feature_map(
        samples, paraMPS['feature_map'],
        {'d': paraMPS['d'], 'theta': paraMPS['theta']})
    para['num_samples'], paraMPS['d'], paraMPS['length'] \
        = samples.shape
    if para['save_name'] is None:
        para['save_name'] = wf.gmps_save_name1('', paraMPS)
    paraMPS['save_name'] = para['save_name']

    print('Parameters:')
    print_dict(para)
    print('Parameters of GMPS:')
    print_dict(paraMPS)

    if tensors is not None:
        assert paraMPS['length'] == len(tensors)
        assert paraMPS['d'] == tensors[0].shape[1]
    mps = generative_MPS(tensors=tensors, para=paraMPS)
    mps.input_samples_v(samples)
    mps.center_orthogonalization(
        0, way='qr', dc=-1, normalize=True)

    mps.initialize_vecs_norms()
    nll = [mps.average_nll_from_norms(
        mps.norms, paraMPS['eps'])]
    print('Initially, nll = %g' % nll[0])

    lr = para['lr']
    for t in range(para['sweepTime']):
        for n in range(paraMPS['length']):
            mps.grad_update_MPS_tensor_by_env(lr)
            if n < paraMPS['length']-1:
                mps.move_center_one_step(
                    'right', 'qr', -1, normalize=True)
                mps.update_vecsL_n(n)
            else:
                mps.normalize_central_tensor()
        for n in range(paraMPS['length']-1, -1, -1):
            mps.grad_update_MPS_tensor_by_env(lr)
            if n > 0:
                mps.move_center_one_step(
                    'left', 'qr', -1, normalize=True)
                mps.update_vecsR_n(n)
            else:
                mps.normalize_central_tensor()
        if ((t + 1) % para['save_dt'] == 0) or (
                t == (para['sweepTime'] - 1)):
            save(para['save_dir'], para['save_name'], [mps.tensors, para, paraMPS],
                 ['tensors', 'para', 'paraMPS'])
        mps.update_norms_center()
        nll_ = mps.evaluate_nll_from_norms(average=True)
        print('t = %d, nll = %g' % (t, nll_))
        if nll[-1] - nll_ < 1e-10:
            lr *= para['lr_decay']
            print('Reducing lr = %g' % lr)
        if lr < 1e-8:
            break
        nll.append(nll_)
    if para['sweepTime'] == 0:
        save(para['save_dir'], para['save_name'], [mps.tensors, para, paraMPS],
             ['tensors', 'para', 'paraMPS'])
    info = {'nll': nll, 'gmps_file': os.path.join(para['save_dir'], para['save_name'])}
    return mps, para, info


def GMPS_train_feature_selection(samples, tensors=None, para=None, paraMPS=None):
    """
    samples.shape = (num of samples_v, num of features)
    tensors = list of tensors of MPS
    NOTE:
        The number of features equals to the length of MPS
    """

    if para is None:
        para = dict()
    para0 = dict()
    para0['num_feature'] = 50  # feature selection
    para0['lr'] = 0.1  # learning rate
    para0['lr_decay'] = 0.9  # decaying of learning rate
    para0['sweepTime'] = 0  # sweep time
    para0['sweepTimeNoData'] = 100  # sweep time with no saved data
    para0['sweepTimeFS'] = 100  # sweep time after feature selection

    para0['isSave'] = True  # whether save MPS
    para0['save_dir'] = './'  # where to save MPS
    para0['save_name'] = None  # saving file name
    para0['save_dt'] = 10  # the frequency to save
    para0['record'] = 'record.log'  # name of log file
    para0['device'] = choose_device()  # cpu or gpu
    para0['dtype'] = tc.float64
    para = dict(para0, **para)

    if tensors is None:
        para['sweepTime'] = max(para['sweepTimeNoData'], para['sweepTime'])
    mps, para1, info1 = GMPS_train(samples, tensors, para, paraMPS)
    pos = mps.feature_selection_OEE(para['num_feature'])
    samples = samples[:, pos]

    para1['save_name'] += 'feature_selected_%i' % para1['num_feature']
    para1['sweepTime'] = para['sweepTimeFS']
    mps, para1, info2 = GMPS_train(samples, tensors=None, para=para1, paraMPS=paraMPS)
    save(para1['save_dir'], para1['save_name'], [mps.tensors, para, paraMPS, pos],
         ['tensors', 'para', 'paraMPS', 'pos_fs'])
    info2['pos_fs'] = pos
    return mps, para1, info1, info2


def GMPS_classification(trainset, testset,
                        para=None, paraMPS=None):
    """
    :param trainset: 训练集
    :param testset: 测试集
    :param para: 算法参数
    :param paraMPS: MPS参数
    """

    if para is None:
        para = dict()
    para0 = dict()
    para0['lr'] = 0.1  # learning rate
    para0['lr_decay'] = 0.9  # decaying factor of learning rate
    para0['sweepTime'] = 100  # sweep time
    para0['sweepTimeNoData'] = 50  # sweep time

    para0['isSave'] = True  # whether to save MPS
    para0['save_dir'] = './'  # where to save MPS
    para0['save_name'] = None  # name of the saved file
    para0['save_dt'] = 20  # the frequency to save
    para0['record'] = 'record.log'  # name of log file
    para0['device'] = choose_device()  # cpu or gpu
    para0['dtype'] = tc.float64
    para = dict(para0, **para)

    if paraMPS is None:
        paraMPS = dict()
    paraMPS0 = {
        'length': 784,
        'd': 2,
        'chi': 3,
        'boundary': 'open',
        'feature_map': 'cossin',
        'theta': 1.0,
        'eps': 1e-14,
        'device': para['device'],
        'dtype': para['dtype']
    }
    paraMPS = dict(paraMPS0, **paraMPS)

    samples, labels = df.dataset2tensors(trainset)
    samples_t, labels_t = df.dataset2tensors(testset)
    samples, samples_t = \
        samples.reshape(samples.shape[0], -1), \
        samples_t.reshape(samples_t.shape[0], -1)
    classes = sorted(list(set(list(labels.cpu().numpy()))))

    for digit in range(len(classes)):
        samples_now = samples[labels == classes[digit], :]
        if samples_now.numel() == 0:
            print('Category %i has no samples_v; '
                  'remove this category' % classes[digit])
        else:
            print('Training GMPS for digits '
                   + str(classes[digit]))
            para['save_name'] = wf.gmps_save_name1(
                classes[digit], paraMPS)
            data = load(os.path.join(
                para['save_dir'], para['save_name']),
                ['tensors', 'paraMPS'])
            if type(data) is tuple:
                tensors, paraMPS1 = data
            elif data is not None:  # mps为GMPS类
                tensors, paraMPS1 = data.tensors, data.para
            else:
                tensors, paraMPS1 = None, dict()
                para['sweepTime'] = max(
                    para['sweepTime'], para['sweepTimeNoData'])
            if tensors is not None:
                print('Load existing samples...')
            else:
                print('Train new GMPS ...')
            GMPS_train(samples_now.to(
                device=para['device'], dtype=para['dtype']),
                tensors, para, paraMPS)
            tc.cuda.empty_cache()
    acc_train = wf.acc_from_saved_gmps(
        samples, labels, para, paraMPS)
    acc_test = wf.acc_from_saved_gmps(
        samples_t, labels_t, para, paraMPS)
    print('Train accuracy = %g, test accuracy = %g'
          % (acc_train, acc_test))
    return acc_train, acc_test


def GMPSC_feature_select(trainset, testset,
                         para=None, paraMPS=None):
    """
    :param trainset: 训练集
    :param testset: 测试集
    :param para: 算法参数
    :param paraMPS: MPS参数
    """

    if para is None:
        para = dict()
    para0 = dict()
    para0['num_feature'] = 50  # feature selection
    para0['lr'] = 0.1  # learning rate
    para0['lr_decay'] = 0.9  # decaying factor of learning rate
    para0['sweepTime'] = 100  # sweep time
    para0['sweepTimeNoData'] = 100  # sweep time
    para0['sweepTimeFS'] = 100  # sweep time after feature selection
    para0['calcu_acc_no_select'] = True

    para0['isSave'] = True  # whether to save MPS
    para0['save_dir'] = './'  # where to save MPS
    para0['save_name'] = None  # name of the saved file
    para0['save_dt'] = 20  # the frequency to save
    para0['record'] = 'record.log'  # name of log file
    para0['device'] = choose_device()  # cpu or gpu
    para0['dtype'] = tc.float64
    para = dict(para0, **para)

    if paraMPS is None:
        paraMPS = dict()
    paraMPS0 = {
        'length': 784,
        'd': 2,
        'chi': 3,
        'boundary': 'open',
        'feature_map': 'cossin',
        'theta': 1.0,
        'eps': 1e-14,
        'device': para['device'],
        'dtype': para['dtype']
    }
    paraMPS = dict(paraMPS0, **paraMPS)

    samples, labels = df.dataset2tensors(trainset)
    samples_t, labels_t = df.dataset2tensors(testset)
    samples, samples_t = \
        samples.reshape(samples.shape[0], -1), \
        samples_t.reshape(samples_t.shape[0], -1)
    classes = sorted(list(set(list(labels.cpu().numpy()))))

    gmps_names = list()
    gmps_fs_names = list()
    for digit in range(len(classes)):
        samples_now = samples[labels == classes[digit], :]
        if samples_now.numel() == 0:
            print('Category %i has no samples_v; '
                  'remove this category' % classes[digit])
        else:
            print('Training GMPS for digits '
                   + str(classes[digit]))
            para['save_name'] = wf.gmps_save_name1(
                classes[digit], paraMPS)
            data = load(os.path.join(
                para['save_dir'], para['save_name']),
                ['tensors', 'paraMPS'])
            if type(data) is tuple:
                tensors, paraMPS1 = data
            elif data is not None:  # mps为GMPS类
                tensors, paraMPS1 = data.tensors, data.para
            else:
                tensors, paraMPS1 = None, dict()
                para['sweepTime'] = max(
                    para['sweepTime'], para['sweepTimeNoData'])
            if tensors is not None:
                print('Load existing samples and retrain for %i sweeps...' % para['sweepTime'])
            else:
                print('Train new GMPS ...')
            info1, info2 = GMPS_train_feature_selection(samples_now.to(
                device=para['device'], dtype=para['dtype']), tensors, para, paraMPS)[2:]
            gmps_names.append(info1['gmps_file'])
            gmps_fs_names.append(info2['gmps_file'])
            tc.cuda.empty_cache()
    if para['calcu_acc_no_select']:
        acc_train = wf.acc_saved_gmpsc_by_file_names(samples, labels, gmps_names)
        acc_test = wf.acc_saved_gmpsc_by_file_names(samples_t, labels_t, gmps_names)
        fprint('With all feature selected, train accuracy = %g, test accuracy = %g'
               % (acc_train, acc_test), file='acc.log')
    else:
        acc_train, acc_test = None, None
    acc_train_fs = wf.acc_saved_gmpsc_FS_by_file_names(samples, labels, gmps_fs_names)
    acc_test_fs = wf.acc_saved_gmpsc_FS_by_file_names(samples_t, labels_t, gmps_fs_names)
    fprint('With %i feature selected, train accuracy = %g, test accuracy = %g'
           % (para['num_feature'], acc_train_fs, acc_test_fs), file='acc.log')
    return acc_train, acc_test, acc_train_fs, acc_test_fs


def generate_sample_by_gmps(
        mps, sample=None, order_g=None, paraG=None):
    from Algorithms.wheels_gmps import generate_from_onebody_rho

    if paraG is None:
        paraG = dict()
    para0 = {
        'generate_order': 'ascending',
        'way': 'multi_average',  # single, multi-average
        'num_s': 1
    }
    paraG = dict(para0, **paraG)

    if type(mps) in [tuple, list]:
        # mps = (tensors, paraMPS)
        mps = generative_MPS(mps[0], mps[1])
    mps.clone_tensors()
    mps.correct_device()
    mps.to()

    if order_g is None:
        if paraG['generate_order'] == 'ascending':
            order_g = list(range(len(mps.tensors)))
        elif paraG['generate_order'] == 'descending':
            order_g = list(range(len(mps.tensors) - 1, -1, -1))
        else:
            order_g = copy.deepcopy(paraG['generate_order'])
    order_g = np.array(order_g)

    if sample is None:
        sample = tc.zeros(len(mps.tensors),
                          device=mps.device, dtype=mps.dtype)
    else:
        sample = sample.clone().to(
            device=mps.device, dtype=mps.dtype)
        order_p = list(range(sample.numel()))
        for x in order_g:
            order_p.remove(x.item())
        sample_v = df.feature_map(
            sample[order_p], mps.para['feature_map'],
            para={'d': mps.para['d'],
                  'theta': mps.para['theta']})
        mps.project_multi_qubits(order_p, sample_v[0])

    order_gn = np.argsort(np.argsort(order_g))
    if (paraG['way'] == 'single') or (paraG['num_s'] == 1):
        pos = 0
        while len(order_gn) > 0:
            mps.center_orthogonalization(
                c=order_gn[0], way='qr', normalize=True)
            rho = mps.one_body_RDM(order_gn[0])
            state, vec = generate_from_onebody_rho(rho, mps.para, paraG)
            sample[order_g[pos]] = state
            mps.project_qubit_nt(order_gn[0], vec)
            mps.center = max(0, mps.center-1)

            order_gn[order_gn>order_gn[0]] -= 1
            order_gn = order_gn[1:]
            pos += 1
    else:
        sample_g = tc.zeros(
            sample.shape,
            device=sample.device, dtype=sample.dtype)
        for ns in range(paraG['num_s']):
            sample_ = sample.clone()
            mps1 = generative_MPS(mps.tensors, mps.para)
            mps1.clone_tensors()
            order_gn1 = copy.deepcopy(order_gn)
            pos = 0
            while len(order_gn1) > 0:
                mps1.center_orthogonalization(
                    c=order_gn1[0], way='qr', normalize=True)
                rho = mps1.one_body_RDM(order_gn1[0])
                state, vec = generate_from_onebody_rho(rho, mps.para, paraG)
                sample_[order_g[pos]] = state
                mps1.project_qubit_nt(order_gn1[0], vec)
                mps1.center = max(0, mps1.center - 1)

                order_gn1[order_gn1 > order_gn1[0]] -= 1
                order_gn1 = order_gn1[1:]
                pos += 1
            sample_g += sample_
        sample = sample_g / paraG['num_s']
    return sample



