import torch
import matplotlib

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
import matplotlib.patches as mpatches

matplotlib.use('Agg')

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor, source_classes: torch.Tensor, target_classes: torch.Tensor, 
              filename: str, source_color='r', target_color='b'):
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    domains = np.concatenate([source_classes.numpy(), target_classes.numpy() + 5], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=0).fit_transform(features)

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    colors_all = ['r', 'salmon', 'darkred', 'darkorange', 'gold', 'b', 'purple', 'navy', 'cyan', 'darkcyan']
    labels_all = ['source_bus', 'source_subway', 'source_car', 'source_airplane', 'source_train', 'target_bus', 'target_subway', 'target_car', 'target_airplane', 'target_train']
    colors = []
    labels = []
    for label in np.unique(domains):
        colors.append(colors_all[label])
        labels.append(labels_all[label])
    patches = []
    dictionary = {label: idx for idx, label in enumerate(np.unique(domains))}
    for i in range(len(colors)):
        patches.append(mpatches.Patch(color=colors[i], label=labels[i]))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[dictionary[x] for x in domains], cmap=col.ListedColormap(colors), s=5)
    plt.legend(handles=patches)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)

