import torch as tc
from torch import nn, optim
from Library import DataFun as df
from Library.BasicFun import print_dict
from Library.MatrixProductState import \
    ResMPS_basic, activated_ResMPS


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
        'num_c': 10,  # 类别数
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
    paraMPS['device'] = para['device']
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
        loss_rec, num_c, total = 0, 0, 0
        for nb, (img, lb) in enumerate(train_loader):
            vecs = df.feature_map(img.to(
                device=mps.device, dtype=mps.dtype),
                which=para['feature_map'],
                para={'d': paraMPS['d']})
            lb = lb.to(device=mps.device)
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

        test_loss, num_c, total = 0, 0, 0
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
                loss = criterion(yt, lbt)
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
