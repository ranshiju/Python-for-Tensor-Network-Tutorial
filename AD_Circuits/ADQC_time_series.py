import torch as tc
import numpy as np
from matplotlib import pyplot as plt
from Library.BasicFun import choose_device
from Library.MathFun import series_sin_cos
from Algorithms import ADQC_algo, LSTM_algo


print('设置参数')
# 通用参数
para = {'lr': 1e-3,  # 初始学习率
        'length_tot': 500,  # 序列总长度
        'order_g': 10,  # 生成序列的傅里叶阶数
        'length': 8,  # 每个样本长度
        'batch_size': 2000,  # batch大小
        'it_time': 1000,  # 总迭代次数
        'dtype': tc.float64,  # 数据精度
        'device': choose_device()}  # 计算设备（cuda优先）
# LSTM参数
para_lstm = {'n_layer': 6,  # LSTM层数
             'h_dim': 100}  # 隐藏变量维数
# ADQC参数
para_adqc = {'depth': 4}  # ADQC量子门层数
# QRNN参数
para_qrnn = {'depth': 4,  # 每个unit的ADQC量子门层数
             'ancillary_length': 4,  # 辅助量子比特个数
             'ini_way': 'identity'}  # 初始化为近似单位阵
para_adqc = dict(para, **para_adqc)
para_qrnn = dict(para, **para_qrnn)
para_lstm = dict(para, **para_lstm)

print('随机生成一维序列')
x = tc.arange(para['length_tot'])
series = series_sin_cos(x, tc.randn(para['order_g']),
                        tc.randn(para['order_g']))

num_train = int(series.numel() * 0.8)
f_train, = plt.plot(list(range(num_train)), series[:num_train])
f_test, = plt.plot(list(range(num_train-1, series.numel(), 1)),
                   series[num_train-1:], linewidth='3', color='r')
plt.title('Time series to be learnt')
plt.legend([f_train, f_test], ['training set', 'testing set'])
plt.xlabel('t')
plt.ylabel('x')
plt.show()

print('预处理数据')
series1 = series.to(dtype=para_adqc['dtype'],
                    device=para_adqc['device'])
shift = -series1.min() + 0.2 * series1.min().abs()
series1 = series1 + shift
ratio = series1.max() * 1.2
series1 = series1 / ratio

print('训练LSTM实现预测')
_, results_lstm, para_lstm = \
    LSTM_algo.LSTM_predict_time_series(series1, para_lstm)
output_lstm = tc.cat([results_lstm['train_pred'],
                      results_lstm['test_pred']], dim=0)
output_lstm = output_lstm * ratio - shift

print('训练ADQC实现预测')
_, results_adqc, para_adqc = \
    ADQC_algo.ADQC_predict_time_series(series1, para_adqc)
output_adqc = tc.cat([results_adqc['train_pred'],
                      results_adqc['test_pred']], dim=0)
output_adqc = output_adqc * ratio - shift

print('训练QRNN实现预测')
_, results_qrnn, para_qrnn = \
    ADQC_algo.QRNN_predict_time_series(series1, para_qrnn)
output_qrnn = tc.cat([results_qrnn['train_pred'],
                      results_qrnn['test_pred']], dim=0)
output_qrnn = output_qrnn * ratio - shift

# ===================== 画图程序 =====================
num_train = results_adqc['train_pred'].numel()
output_lstm = output_lstm.cpu().numpy()
output_adqc = output_adqc.cpu().numpy()
output_qrnn = output_qrnn.cpu().numpy()

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
plt.show()

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
plt.show()
