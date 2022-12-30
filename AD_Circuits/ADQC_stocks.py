import torch as tc
from Library.BasicFun import load, plot
from Library.MathFun import sign_accuracy
from Algorithms import ADQC_algo as ag


print('设置参数、读取数据')
para = {'lr': 1e-3,
        'length': 8,
        'depth': 4,
        'batch_size': 2000,
        'it_time': 1000}

close0 = load('../data/stocks/000001.SZ.data', names='close')

print('预处理数据')
close_diff0 = tc.diff(close0)
shift = -close_diff0.min() + 0.1 * close_diff0.min().abs()
close_diff = close_diff0 + shift
ratio = close_diff.max() * 1.1
close_diff = close_diff / ratio

print('使用ADQC实现预测')
_, results_adqc = ag.ADQC_predict_time_series(close_diff, para)
results_adqc['train_pred'] = results_adqc['train_pred'] * ratio - shift
results_adqc['test_pred'] = results_adqc['test_pred'] * ratio - shift
acc_train_adqc = sign_accuracy(results_adqc['train_pred'],
                               close_diff0[:results_adqc['train_pred'].shape[0]])
acc_test_adqc = sign_accuracy(results_adqc['test_pred'],
                              close_diff0[results_adqc['train_pred'].shape[0]:])
print('Sign accuracy of ADQC: train = %g, test = %g' % (acc_train_adqc, acc_test_adqc))

output_adqc = tc.cat([results_adqc['train_pred'],
                      results_adqc['test_pred']], dim=0)
data_adqc = close0.clone()
data_adqc[1:] = close0[:-1] + output_adqc

print('使用QRNN实现预测')
para['unitary'] = False
para['ini_way'] = 'identity'
_, results_qrnn = ag.QRNN_predict_time_series(close_diff, para)
results_qrnn['train_pred'] = results_qrnn['train_pred'] * ratio - shift
results_qrnn['test_pred'] = results_qrnn['test_pred'] * ratio - shift
acc_train_qrnn = sign_accuracy(results_qrnn['train_pred'],
                               close_diff0[:results_qrnn['train_pred'].shape[0]])
acc_test_qrnn = sign_accuracy(results_qrnn['test_pred'],
                              close_diff0[results_qrnn['train_pred'].shape[0]:])
print('Sign accuracy of QRNN: train = %g, test = %g' % (acc_train_qrnn, acc_test_qrnn))

output_qrnn = tc.cat([results_qrnn['train_pred'],
                      results_qrnn['test_pred']], dim=0)
data_qrnn = close0.clone()
data_qrnn[1:] = close0[:-1] + output_qrnn

plot(tc.arange(close_diff0.numel()), output_adqc, output_qrnn)
plot(tc.arange(close0.numel()), close0, data_adqc, data_qrnn)
