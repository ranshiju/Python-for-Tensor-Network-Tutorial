import copy, os
import torch as tc
import Library.BasicFun as bf
import Library.MatrixProductState as mps
from Library.DataFun import load_mnist, dataset2tensors
from Algorithms.MPS_algo import GMPS_train
from Algorithms.wheels_gmps import \
    acc_saved_gmpsc_by_file_names, gmps_save_name1


'''
下述示例用到的GMPS数据可从百度云盘下载：
https://pan.baidu.com/s/1LnbcWDRm3YODcSGnc3RZ_A 
提取码: TNML
'''

para = {
    'lr': 0.1,
    'sweepTime': 50,
    'dtype': tc.float64,
    'save_dir': r'../data/GMPS_MNIST/'  # path for GMPS's
}

train_dataset, test_dataset = load_mnist('mnist')
train_img, labels = dataset2tensors(train_dataset)
test_img, labels_t = dataset2tensors(test_dataset)
train_img = train_img.reshape(train_img.shape[0], -1)
test_img = test_img.reshape(test_img.shape[0], -1)

paraMPS, OEE_list, gmps_names = dict(), list(), list()
print('Calculating OEE for each class...')
for digit in range(10):
    para['save_name'] = r'GMPS_chi12_theta1.0_FM_' \
                        r'cossin_digit_%i' % digit
    tensors, paraMPS = bf.load(os.path.join(
        para['save_dir'], para['save_name']),
        ['tensors', 'paraMPS'])
    gmps = mps.generative_MPS(tensors, paraMPS)
    gmps.correct_device()
    ent = gmps.entanglement_ent_onsite()
    OEE_list.append(ent)
    gmps_names.append(os.path.join(
        para['save_dir'], para['save_name']))
bf.show_multiple_images(OEE_list, (2, 5), list(
    range(10)), img_size=(28, 28))

print('Calculating accuracies of GMPSC with all features...')
acc_train = acc_saved_gmpsc_by_file_names(
    train_img.to(device=bf.choose_device(),
                 dtype=paraMPS['dtype']),
    labels.to(device=bf.choose_device()),
    gmps_files=gmps_names, classes=list(range(10)))
acc_test = acc_saved_gmpsc_by_file_names(
    test_img.to(device=bf.choose_device(),
                dtype=paraMPS['dtype']),
    labels_t.to(device=bf.choose_device()),
    gmps_files=gmps_names, classes=list(range(10)))
bf.fprint('Train & test acc = %g, %g' % (
    acc_train.item(), acc_test.item()),
          file='results.log')

OEE_sum = sum(OEE_list)
selected_pos = tc.argsort(OEE_sum, descending=True)
for num in [15, 20, 50, 100, 200, 300, 400, 500, 600]:
    bf.fprint('Selecting %i features and classify with the '
              'RMD of full GMPS ...' % num, file='results.log')
    acc_train = acc_saved_gmpsc_by_file_names(
        train_img, labels,
        gmps_files=gmps_names, pos=selected_pos[:num],
        classes=list(range(10)))
    acc_test = acc_saved_gmpsc_by_file_names(
        test_img, labels_t,
        gmps_files=gmps_names, pos=selected_pos[:num],
        classes=list(range(10)))
    bf.fprint('train & test acc = %g, %g' %
              (acc_train.item(), acc_test.item()),
              file='results.log')

for num in [15, 20, 50, 100, 200, 300, 400, 500, 600]:
    print('Training GMPSC with selected features...')
    para_tmp = copy.deepcopy(para)
    para_tmp['save_dir'] = './data_tmp/'
    paraMPS_tmp = copy.deepcopy(paraMPS)
    gmps_names = list()
    train_tmp, test_tmp = list(), list()
    train_lbs, test_lbs = list(), list()
    for digit in range(10):
        print('Class: %i' % digit)
        samples = train_img[labels == digit][
                  :, selected_pos[:num]]
        para_tmp['save_name'] = gmps_save_name1(
            digit, paraMPS_tmp)
        para_tmp['save_name'] += '_selected%i' % num
        gmps_names.append(os.path.join(
            para_tmp['save_dir'], para_tmp['save_name']))
        GMPS_train(samples, para=para_tmp, paraMPS=paraMPS_tmp)
        train_tmp.append(samples)
        test_tmp.append((test_img[labels_t == digit][
                         :, selected_pos[:num]]).to(
            device=paraMPS['device'], dtype=paraMPS['dtype']))
        train_lbs.append(tc.ones(len(
            train_tmp[-1]), device=paraMPS['device'],
            dtype=tc.int64) * digit)
        test_lbs.append(tc.ones(
            len(test_tmp[-1]), device=paraMPS['device'],
            dtype=tc.int64) * digit)
    acc_train = acc_saved_gmpsc_by_file_names(
        tc.cat(train_tmp, dim=0), tc.cat(train_lbs, dim=0),
        gmps_files=gmps_names, classes=list(range(10)))
    acc_test = acc_saved_gmpsc_by_file_names(
        tc.cat(test_tmp, dim=0), tc.cat(test_lbs, dim=0),
        gmps_files=gmps_names, classes=list(range(10)))
    bf.fprint('With GMPS trained by selected features, '
              'train & test acc (%i features) = %g, %g' % (
        num, acc_train.item(), acc_test.item()), file='results.log')










