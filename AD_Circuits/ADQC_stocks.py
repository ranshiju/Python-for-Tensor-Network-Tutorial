import torch as tc
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from Library.BasicFun import load, choose_device
from Algorithms import ADQC_algo, LSTM_algo


print('设置参数')
# 通用参数
para = {'length': 8,  # 每个样本长度
        'batch_size': 2000,  # batch大小
        'it_time': 1000,  # 总迭代次数
        'dtype': tc.float64,  # 数据精度
        'device': choose_device()}  # 计算设备（cuda优先）
# LSTM参数
para_lstm = {'lr': 1e-3,  # 初始学习率
             'n_layer': 6,  # LSTM层数
             'h_dim': 100}  # 隐藏变量维数
# ADQC参数
para_adqc = {'lr': 5e-3,  # 初始学习率
             'depth': 4}  # ADQC量子门层数
# QRNN参数
para_qrnn = {'lr': 5e-3,  # 初始学习率
             'depth': 4,  # 每个unit的ADQC量子门层数
             'ancillary_length': 2,  # 辅助量子比特个数
             'ini_way': 'random'}  # 初始化为近似单位阵
para_adqc = dict(deepcopy(para), **para_adqc)
para_qrnn = dict(deepcopy(para), **para_qrnn)
para_lstm = dict(deepcopy(para), **para_lstm)

series = load('../data/stocks/000001.SZ.data', names='close')

plt.plot(tc.arange(series.numel()), series)
plt.show()

print('预处理数据')
shift = -series.min() + 0.1 * series.min().abs()
close_diff = series + shift
ratio = close_diff.max() * 1.1
series1 = close_diff / ratio

print('训练QRNN实现预测')
_, results_qrnn, para_qrnn = ADQC_algo.QRNN_predict_time_series(series1.clone(), para_qrnn)
output_qrnn = tc.cat([results_qrnn['train_pred'],
                      results_qrnn['test_pred']], dim=0)
output_qrnn = output_qrnn * ratio - shift

print('训练ADQC实现预测')
_, results_adqc, para_adqc = ADQC_algo.ADQC_predict_time_series(series1.clone(), para_adqc)
output_adqc = tc.cat([results_adqc['train_pred'],
                      results_adqc['test_pred']], dim=0)
output_adqc = output_adqc * ratio - shift

print('训练LSTM实现预测')
_, results_lstm, para_lstm = LSTM_algo.LSTM_predict_time_series(series1.clone(), para_lstm)
output_lstm = tc.cat([results_lstm['train_pred'],
                      results_lstm['test_pred']], dim=0)
output_lstm = output_lstm * ratio - shift
# ===================== 画图程序 =====================
x = tc.arange(series.numel())

num_train = results_qrnn['train_pred'].numel()
output_lstm = output_lstm.cpu().numpy()
output_adqc = output_adqc.cpu().numpy()
output_qrnn = output_qrnn.cpu().numpy()
plt.rcParams['figure.figsize'] = (12, 5)

plt.subplot(3, 1, 1)
plt.title('LSTM')
l1, = plt.plot(x, series)
l2, = plt.plot(x[para['length']:num_train],
               output_lstm[para['length']:num_train],
               marker='o', color='r', markerfacecolor='none',
               markeredgewidth=0.4, linewidth=0)
l3, = plt.plot(x[num_train:], output_lstm[num_train:],
               marker='o', color='b', markerfacecolor='none',
               markeredgewidth=0.4, linewidth=0)
plt.legend([l1, l2, l3],
           ['ground truth', 'train set', 'test set'])

plt.subplot(3, 1, 2)
plt.title('ADQC')
l4, = plt.plot(x, series)
l5, = plt.plot(x[para['length']:num_train],
               output_adqc[para['length']:num_train],
               marker='o', color='r', markerfacecolor='none',
               markeredgewidth=0.4, linewidth=0)
l6, = plt.plot(x[num_train:], output_adqc[num_train:],
               marker='o', color='b', markerfacecolor='none',
               markeredgewidth=0.4, linewidth=0)
plt.legend([l4, l5, l6],
           ['ground truth', 'train set', 'test set'])

plt.subplot(3, 1, 3)
plt.title('QRNN')
l7, = plt.plot(x, series)
l8, = plt.plot(x[para['length']:num_train],
               output_qrnn[para['length']:num_train],
               marker='o', color='r', markerfacecolor='none',
               markeredgewidth=0.4, linewidth=0)
l9, = plt.plot(x[num_train:], output_qrnn[num_train:],
               marker='o', color='b', markerfacecolor='none',
               markeredgewidth=0.4, linewidth=0)
plt.legend([l7, l8, l9],
           ['ground truth', 'train set', 'test set'])
plt.savefig('fig-series.png', dpi=300)
plt.show()

plt.rcParams['figure.figsize'] = (12, 5)
plt.subplot(1, 1, 1)
epochs = (np.arange(len(results_lstm['train_loss'])) + 1
          ) * para_lstm['print_time']
l10, = plt.plot(epochs, np.array(results_lstm['train_loss']),
                linestyle='--', color='r')
l11, = plt.plot(epochs, np.array(results_lstm['test_loss']), color='r')
l12, = plt.plot(epochs, np.array(results_adqc['train_loss']),
                linestyle='--', color='b')
l13, = plt.plot(epochs, np.array(results_adqc['test_loss']), color='b')
l14, = plt.plot(epochs, np.array(results_qrnn['train_loss']),
                linestyle='--', color='g')
l15, = plt.plot(epochs, np.array(results_qrnn['test_loss']), color='g')
plt.legend([l10, l11, l12, l13, l14, l15],
           ['LSTM: train loss', 'LSTM: test loss',
            'ADQC: train loss', 'ADQC: test loss',
            'QRNN: train loss', 'QRNN: test loss'])
plt.savefig('fig-loss.png', dpi=300)
plt.show()
