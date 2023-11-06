import torch as tc
import matplotlib.pyplot as plt
import Library.BasicFun as bf
from sklearn import manifold


def visualize_tsne(samples, labels=None, para=None):
    para0 = {
        'perplexity': 5,
        'n_components': 2,
        'init': 'pca',
        'learning_rate': 'auto',
        'metric': 'euclidean',  # or 'precomputed'
        'save_name': 'visual.png',
        'show': True
    }
    para = bf.combine_dicts(para0, para)
    if para['metric'] == 'precomputed':
        para['init'] = 'random'
    if labels is None:
        labels = tc.zeros(samples.shape[0],
                          dtype=tc.int64)

    tsne = manifold.TSNE(
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
    if para['show']:
        plt.show()





