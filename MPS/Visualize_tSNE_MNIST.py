import torch as tc
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from Library.BasicFun import combine_dicts
from Library.DataFun import load_mnist, dataset2tensors
from Library.MathFun import metric_matrix_neg_log_cos_sin


def visualize_tsne(samples, labels=None, para=None):
    # 该函数位于/Algorithms/Visualizations.py
    para0 = {
        'perplexity': 30,
        'n_components': 2,
        'init': 'pca',
        'learning_rate': 'auto',
        'metric': 'euclidean',  # or 'precomputed'
        'save_name': 'visual.png'
    }
    para = combine_dicts(para0, para)
    if para['metric'] == 'precomputed':
        para['init'] = 'random'
    if labels is None:
        labels = tc.zeros(samples.shape[0],
                          dtype=tc.int64)

    tsne = TSNE(
        perplexity=para['perplexity'], init=para['init'],
        n_components=para['n_components'],
        learning_rate=para['learning_rate'],
        metric=para['metric'])

    samples1 = tsne.fit_transform(
        samples.reshape(samples.shape[0], -1))
    x_min = samples1[:, 0].min(0)
    x_max = samples1[:, 0].max(0)
    samples1[:, 0] = (samples1[:, 0] - x_min
                      ) / (x_max - x_min)  # 归一化
    y_min = samples1[:, 1].min(0)
    y_max = samples1[:, 1].max(0)
    samples1[:, 1] = (samples1[:, 1] - y_min
                      ) / (y_max - y_min)  # 归一化
    plt.figure(figsize=(8, 8))
    color_list = [plt.cm.tab10(n)
                  for n in range(labels.max() + 1)]
    for i in range(samples1.shape[0]):
        plt.text(samples1[i, 0], samples1[i, 1],
                 str(labels[i].item()),
                 color=color_list[labels[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(para['save_name'])
    plt.show()


dataset = 'mnist'
classes = [0, 1, 2]
num = 3000
perplexity = 20
save_path = '../data/visualizations_4.7/'

# 从训练集随机选出num个样本
train_dataset = load_mnist(
    dataset, process={'classes': classes})[0]
samples, labels = dataset2tensors(train_dataset)
ind = tc.randperm(samples.shape[0])[:num]
samples = samples[ind]
labels = labels[ind]

visualize_tsne(samples.numpy(), labels.numpy(),
               para={'metric': 'euclidean',
                     'perplexity': perplexity,
                     'save_name': save_path+'EU.png'})

dis = metric_matrix_neg_log_cos_sin(samples)
visualize_tsne(dis.numpy(), labels.numpy(),
               para={'metric': 'precomputed',
                     'perplexity': perplexity,
                     'save_name': save_path+'NLF.png'})

for beta in [1.01, 1.05, 1.1, 1.3]:
    dis_resc = beta ** dis
    fname = save_path + 'RLF' + str(beta) + '.png'
    visualize_tsne(dis_resc.numpy(), labels.numpy(),
                   para={'metric': 'precomputed',
                         'perplexity': perplexity,
                         'save_name': fname})
