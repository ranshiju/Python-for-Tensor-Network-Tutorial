import torch as tc
import Library.BasicFun as bf
from torch import nn
from torch.optim import Adam


class FC2(nn.Module):

    def __init__(self, dim_in, dim_h, dim_out):
        super(FC2, self).__init__()
        self.layer1 = nn.Linear(dim_in, dim_h)
        self.layer2 = nn.Linear(dim_h, dim_out)
        self.activate = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activate(x)
        x = self.layer2(x)
        return x


def FC2_classifier(trainloader, testloader, num_classes,
                   length, para=None):
    para0 = dict()  # 默认参数
    para0['batch_size'] = 200  # 批次大小
    para0['dim_h'] = 4  # 隐藏层维数
    para0['lr'] = 2e-4  # 初始学习率
    para0['it_time'] = 1000  # 迭代次数
    para0['print_time'] = 10  # 打印间隔
    para0['device'] = None
    para0['dtype'] = tc.float64

    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数
    para['dim_in'] = length
    para['dim_out'] = num_classes
    para['device'] = bf.choose_device(para['device'])

    net = FC2(para['dim_in'], para['dim_h'], para['dim_out'])
    net = net.to(device=para['device'], dtype=para['dtype'])
    optimizer = Adam(net.parameters(), lr=para['lr'])

    loss_train_rec = list()
    acc_train = list()
    loss_test_rec = list()
    acc_test = list()
    criteria = nn.CrossEntropyLoss()
    for t in range(para['it_time']):
        loss_tmp, num_t, num_c = 0.0, 0, 0
        for n, (samples, lbs) in enumerate(trainloader):
            y = net(samples)
            loss = criteria(y, lbs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_tmp += loss.item() * samples.shape[0]
            num_t += samples.shape[0]
            num_c += (y.data.argmax(dim=1) == lbs).sum()
        if (t + 1) % para['print_time'] == 0:
            loss_train_rec.append(loss_tmp / num_t)
            acc_train.append(num_c / num_t)
            with tc.no_grad():
                loss_tmp, num_t, num_c = 0.0, 0, 0
                for n, (samples, lbs) in enumerate(testloader):
                    y = net(samples)
                    loss = criteria(y, lbs)
                    loss_tmp += loss.item() * samples.shape[0]
                    num_t += samples.shape[0]
                    num_c += (y.data.argmax(dim=1) == lbs).sum()
                loss_test_rec.append(loss_tmp / num_t)
                acc_test.append(num_c / num_t)
                print('Epoch %i: train loss %g, test loss %g \n train acc %g, test acc %g' %
                      (t + 1, loss_train_rec[-1], loss_test_rec[-1], acc_train[-1], acc_test[-1]))
    results = dict()
    results['train_loss'] = loss_train_rec
    results['test_loss'] = loss_test_rec
    results['train_acc'] = acc_train
    results['test_acc'] = acc_test
    return net, results, para
