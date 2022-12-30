import torch as tc
import Library.BasicFun as bf
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from Library.DataFun import split_time_series


class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer):
        super(RNN, self).__init__()
        # self.n_layer = n_layer
        # self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()


def LSTM_predict_time_series(data, para=None):
    para0 = dict()  # 默认参数
    para0['test_ratio'] = 0.2  # 将部分样本划为测试集
    para0['length'] = 8  # 数据样本维数
    para0['batch_size'] = 200  # 批次大小
    para0['n_layer'] = 4  # LSTM层数
    para0['h_dim'] = 100  # 隐藏层维数
    para0['lr'] = 1e-4  # 初始学习率
    para0['it_time'] = 1000  # 迭代次数
    para0['print_time'] = 10  # 打印间隔
    para0['device'] = None
    para0['dtype'] = tc.float64

    if para is None:
        para = para0
    else:
        para = dict(para0, **para)  # 更新para参数
    para['device'] = bf.choose_device(para['device'])

    num_train = int(data.numel() * (1 - para['test_ratio']))
    trainset, train_lbs = split_time_series(
        data[:num_train], para['length'], para['device'], para['dtype'])
    testset, test_lbs = split_time_series(
        data[num_train - para['length']:], para['length'], para['device'], para['dtype'])
    trainloader = DataLoader(TensorDataset(trainset, train_lbs), batch_size=para['batch_size'], shuffle=True)
    testloader = DataLoader(TensorDataset(testset, test_lbs), batch_size=para['batch_size'], shuffle=False)

    net = RNN(1, para['h_dim'], para['n_layer'])
    net = net.to(device=para['device'], dtype=para['dtype'])
    optimizer = optim.Adam(net.parameters(), lr=para['lr'])

    loss_train_rec = list()
    loss_test_rec = list()
    for t in range(para['it_time']):
        net.train()
        loss_tmp = 0.0
        for nb, (samples, lbs) in enumerate(trainloader):
            norms = net(samples.reshape(samples.shape+(1, )))
            loss = nn.MSELoss()(norms, lbs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_tmp += loss.item() * samples.shape[0]
        if (t+1) % para['print_time'] == 0:
            loss_train_rec.append(loss_tmp / train_lbs.numel())
            with tc.no_grad():
                net.eval()
                loss_tmp = 0.0
                for nb, (samples, lbs) in enumerate(testloader):
                    norms = net(samples.reshape(samples.shape + (1,)))
                    loss = nn.MSELoss()(norms, lbs)
                    loss_tmp += loss.item() * samples.shape[0]
                loss_test_rec.append(loss_tmp / test_lbs.numel())
            print('Epoch %i: train loss %g, test loss %g' %
                  (t+1, loss_train_rec[-1], loss_test_rec[-1]))

    with tc.no_grad():
        results = dict()
        norms = net(trainset.reshape(trainset.shape+(1, )))
        output = tc.cat([data[:para['length']].to(dtype=norms.dtype), norms.to(device=data.device)], dim=0)
        results['train_pred'] = output.data
        norms1 = net(testset.reshape(testset.shape+(1, )))
        results['test_pred'] = norms1.data.to(device=data.device)
        results['train_loss'] = loss_train_rec
        results['test_loss'] = loss_test_rec
    return net, results, para

