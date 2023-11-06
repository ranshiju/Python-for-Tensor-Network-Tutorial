import torch as tc
import Library.BasicFun as bf
from numpy import ceil, log2
from torch.nn import MSELoss, NLLLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from Library.DataFun import feature_map, split_time_series
from Library.ADQC import ADQC_LatentGates, QRNN_LatentGates
from Library.QuantumTools import vecs2product_state


def ADQC_classifier(trainloader, testloader, num_classes,
                    length, para=None):
    para0 = dict()  # 默认参数
    para0['n_img'] = 200  # 批次大小
    para0['feature_map'] = 'cossin'  # 特征映射
    para0['lattice'] = 'brick'  # ADQC链接形式（brick或stair）
    para0['depth'] = 4  # ADQC层数
    para0['ini_way'] = 'random'  # 线路初始化策略
    para0['lr'] = 2e-4  # 初始学习率
    para0['it_time'] = 200  # 迭代次数
    para0['print_time'] = 10  # 打印间隔
    para0['eps'] = 1e-12  # 防止无穷大的小数
    para0['device'] = None
    para0['dtype'] = tc.float64

    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数
    para['length'] = length
    para['device'] = bf.choose_device(para['device'])

    qc = ADQC_LatentGates(
        lattice=para['lattice'], num_q=para['length'],
        depth=para['depth'], ini_way=para['ini_way'],
        device=para['device'], dtype=para['dtype'])
    qc.single_state = False  # 切换至多个态演化模式
    optimizer = Adam(qc.parameters(), lr=para['lr'])

    num_qc = int(ceil(log2(num_classes)))

    loss_train_rec = list()
    acc_train = list()
    loss_test_rec = list()
    acc_test = list()
    criteria = NLLLoss()
    for t in range(para['it_time']):
        loss_tmp, num_t, num_c = 0.0, 0, 0
        for n, (samples, lbs) in enumerate(trainloader):
            vecs = feature_map(
                samples, which=para['feature_map'])
            vecs = vecs2product_state(vecs)
            psi1 = qc(vecs)
            psi1 = probabilities_adqc_classifier(
                psi1, num_qc, num_classes)
            loss = criteria(tc.log(psi1), lbs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_tmp += loss.item() * samples.shape[0]
            num_t += samples.shape[0]
            num_c += (psi1.data.argmax(dim=1) == lbs).sum()

        if (t + 1) % para['print_time'] == 0:
            loss_train_rec.append(loss_tmp / num_t)
            acc_train.append(num_c / num_t)
            with tc.no_grad():
                loss_tmp, num_t, num_c = 0.0, 0, 0
                for n, (samples, lbs) in enumerate(testloader):
                    vecs = feature_map(
                        samples, which=para['feature_map'])
                    vecs = vecs2product_state(vecs)
                    psi1 = qc(vecs)
                    psi1 = probabilities_adqc_classifier(
                        psi1, num_qc, num_classes)
                    loss = criteria(tc.log(psi1), lbs)
                    loss_tmp += loss.item() * samples.shape[0]
                    num_t += samples.shape[0]
                    num_c += (psi1.data.argmax(
                        dim=1) == lbs).sum()
                loss_test_rec.append(loss_tmp / num_t)
                acc_test.append(num_c / num_t)
                print('Epoch %i: train loss %g, '
                      'test loss %g \n '
                      'train acc %g, test acc %g' %
                      (t + 1, loss_train_rec[-1],
                       loss_test_rec[-1],
                       acc_train[-1], acc_test[-1]))
    results = dict()
    results['train_loss'] = loss_train_rec
    results['test_loss'] = loss_test_rec
    results['train_acc'] = acc_train
    results['test_acc'] = acc_test
    return qc, results, para


def ADQC_predict_time_series(data, para=None):
    para0 = dict()  # 默认参数
    para0['test_ratio'] = 0.2  # 将部分样本划为测试集
    para0['length'] = 4  # 数据样本维数
    para0['n_img'] = 200  # 批次大小
    para0['feature_map'] = 'cossin'  # 特征映射
    para0['lattice'] = 'brick'  # ADQC链接形式（brick或stair）
    para0['depth'] = 4  # ADQC层数
    para0['ini_way'] = 'random'  # 线路初始化策略
    para0['lr'] = 2e-4  # 初始学习率
    para0['it_time'] = 200  # 迭代次数
    para0['print_time'] = 10  # 打印间隔
    para0['device'] = None
    para0['dtype'] = tc.float64

    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数
    para['device'] = bf.choose_device(para['device'])

    num_train = int(data.numel() * (1-para['test_ratio']))
    trainset, train_lbs = split_time_series(
        data[:num_train], para['length'], para['device'], para['dtype'])
    testset, test_lbs = split_time_series(
        data[num_train-para['length']:], para['length'], para['device'], para['dtype'])
    trainset = feature_map(trainset, which=para['feature_map'])
    testset = feature_map(testset, which=para['feature_map'])
    trainloader = DataLoader(TensorDataset(trainset, train_lbs), batch_size=para['n_img'], shuffle=True)
    testloader = DataLoader(TensorDataset(testset, test_lbs), batch_size=para['n_img'], shuffle=False)

    qc = ADQC_LatentGates(lattice=para['lattice'], num_q=para['length'], depth=para['depth'], ini_way=para['ini_way'],
                          device=para['device'], dtype=para['dtype'])
    qc.single_state = False  # 切换至多个态演化模式
    optimizer = Adam(qc.parameters(), lr=para['lr'])

    loss_train_rec = list()
    loss_test_rec = list()
    for t in range(para['it_time']):
        loss_tmp = 0.0
        for n, (samples, lbs) in enumerate(trainloader):
            psi0 = vecs2product_state(samples)
            psi1 = qc(psi0)
            norms = probability_0_of_qubit_last(psi1)
            loss = MSELoss()(norms, lbs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_tmp += loss.item() * samples.shape[0]

        if (t+1) % para['print_time'] == 0:
            loss_train_rec.append(loss_tmp / train_lbs.numel())
            loss_tmp = 0.0
            with tc.no_grad():
                for n, (samples, lbs) in enumerate(testloader):
                    psi0 = vecs2product_state(samples)
                    psi1 = qc(psi0)
                    norms = probability_0_of_qubit_last(psi1)
                    loss = MSELoss()(norms, lbs)
                    loss_tmp += loss.item() * samples.shape[0]
            loss_test_rec.append(loss_tmp / test_lbs.numel())
            print('Epoch %i: train loss %g, test loss %g' %
                  (t+1, loss_train_rec[-1], loss_test_rec[-1]))

    with tc.no_grad():
        results = dict()
        psi0 = vecs2product_state(trainset)
        psi1 = qc(psi0)
        output = probability_0_of_qubit_last(psi1)
        output = tc.cat([data[:para['length']].to(dtype=output.dtype), output.to(device=data.device)], dim=0)
        results['train_pred'] = output.data
        psi0 = vecs2product_state(testset)
        psi1 = qc(psi0)
        output1 = probability_0_of_qubit_last(psi1)
        results['test_pred'] = output1.data.to(device=data.device)
        results['train_loss'] = loss_train_rec
        results['test_loss'] = loss_test_rec
    return qc, results, para


def QRNN_predict_time_series(data, para=None):
    para0 = dict()  # 默认参数
    para0['test_ratio'] = 0.2  # 将部分样本划为测试集
    para0['length'] = 4  # 数据样本维数
    para0['n_img'] = 200  # 批次大小
    para0['feature_map'] = 'cossin'  # 特征映射
    para0['ancillary_length'] = 4  # 辅助量子位数
    para0['unitary'] = True  # 是否要求局域张量幺正
    para0['lattice'] = None  # ADQC连接形式（brick或stair，None时采样内置默认连接）
    para0['depth'] = 1  # 每个unit的ADQC层数
    para0['ini_way'] = 'random'  # 线路初始化策略
    para0['lr'] = 2e-4  # 初始学习率
    para0['it_time'] = 1000  # 迭代次数
    para0['print_time'] = 10  # 打印间隔
    para0['device'] = None
    para0['dtype'] = tc.float64

    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数
    para['device'] = bf.choose_device(para['device'])

    num_train = int(data.numel() * (1-para['test_ratio']))
    trainset, train_lbs = split_time_series(
        data[:num_train], para['length'], para['device'], para['dtype'])
    testset, test_lbs = split_time_series(
        data[num_train-para['length']:], para['length'], para['device'], para['dtype'])
    trainset = feature_map(trainset, which=para['feature_map'])
    testset = feature_map(testset, which=para['feature_map'])
    trainloader = DataLoader(TensorDataset(trainset, train_lbs), batch_size=para['n_img'], shuffle=True)
    testloader = DataLoader(TensorDataset(testset, test_lbs), batch_size=para['n_img'], shuffle=False)

    if para['lattice'] is None:
        pos = [[m, para['ancillary_length']] for m in range(para['ancillary_length']-1, -1, -1)]
        pos = pos * para['depth']
    else:
        pos = None
    qc = QRNN_LatentGates(pos_one_layer=pos, lattice=para['lattice'],
                          num_ancillary=para['ancillary_length'], ini_way=para['ini_way'],
                          depth=para['depth'], unitary=para['unitary'],
                          device=para['device'], dtype=para['dtype'])
    optimizer = Adam(qc.parameters(), lr=para['lr'])

    loss_train_rec = list()
    loss_test_rec = list()
    for t in range(para['it_time']):
        loss_tmp = 0.0
        for n, (samples, lbs) in enumerate(trainloader):
            norms = qc(samples)
            loss = MSELoss()(norms, lbs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_tmp += loss.item() * samples.shape[0]

        if (t+1) % para['print_time'] == 0:
            loss_train_rec.append(loss_tmp / train_lbs.numel())
            loss_tmp = 0.0
            with tc.no_grad():
                for n, (samples, lbs) in enumerate(testloader):
                    norms = qc(samples)
                    loss = MSELoss()(norms, lbs)
                    loss_tmp += loss.item() * samples.shape[0]
            loss_test_rec.append(loss_tmp / test_lbs.numel())
            print('Epoch %i: train loss %g, test loss %g' %
                  (t+1, loss_train_rec[-1], loss_test_rec[-1]))

    with tc.no_grad():
        results = dict()
        norms = qc(trainset)
        output = tc.cat([data[:para['length']].to(dtype=norms.dtype), norms.to(device=data.device)], dim=0)
        results['train_pred'] = output.data
        norms1 = qc(testset)
        results['test_pred'] = norms1.data.to(device=data.device)
        results['train_loss'] = loss_train_rec
        results['test_loss'] = loss_test_rec
    return qc, results, para


def probability_0_of_qubit_last(states):
    # states.shape = (量子态个数, 2, 2, 2, ...)
    s = states.shape
    states = states.reshape(-1, s[-1])[:, 0].reshape(s[0], -1)
    return tc.einsum('na,na->digit', states, states.conj())


def probabilities_adqc_classifier(psi, num_qc, num_class):
    s = psi.shape
    psi1 = psi.reshape(s[0], -1, 2 ** num_qc)
    psi1 = tc.einsum('nab,nac->nbc', psi1, psi1.conj())
    p = tc.zeros((s[0], num_class),
                 device=psi.device, dtype=psi1.dtype)
    for n in range(num_class):
        p[:, n] = psi1[:, n, n]
    p = tc.einsum('na,digit->na', p, 1/(tc.norm(p, dim=1)+1e-10))
    return p
