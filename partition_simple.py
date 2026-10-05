"""
partition_simple.py — the plain version of "samples per class per client".

The heatmap version (partition_viz.py) is dense. This one is meant to be read at
a glance: one bar per client, coloured by class, IID beside non-IID.

The point it has to make: under IID every client is a scaled copy of the same
distribution, so all ten bars look identical. Under non-IID they do not.

Out: figures/simple_partition_{cicids,unsw}.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ids_loader, unsw_loader
from ids_loader import load_ids_dataset, CLASS_NAMES
from unsw_loader import load_unsw_dataset, CLASS_NAMES_UNSW

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200,
                     'axes.spines.top': False, 'axes.spines.right': False})


def counts(loader, n_classes, alpha, seed=42):
    ids_loader.SEED = unsw_loader.SEED = seed
    concepts = loader(alpha)[0]
    K = len(concepts[0]['client_data'])
    M = np.zeros((K, n_classes), int)
    for c in concepts:
        for k, (_, yk) in enumerate(c['client_data']):
            if len(yk):
                u, n = np.unique(yk, return_counts=True)
                M[k, u] += n
    return M


def plot(name, loader, classes):
    iid, non = counts(loader, len(classes), 100.0), counts(loader, len(classes), 0.5)
    order = np.argsort(-iid.sum(0))
    classes = [classes[i] for i in order]
    iid, non = iid[:, order], non[:, order]
    cols = plt.get_cmap('tab10')(np.linspace(0, 1, 10))[:len(classes)]
    x = np.arange(iid.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6),
                             gridspec_kw={'height_ratios': [1.25, 1]})

    # ── row 1: stacked totals ────────────────────────────────────────────────
    for ax, M, t in [(axes[0, 0], iid, 'IID  (alpha = 100)'),
                     (axes[0, 1], non, 'non-IID  (alpha = 0.5)')]:
        bot = np.zeros(len(x))
        for ci, c in enumerate(classes):
            ax.bar(x, M[:, ci], 0.72, bottom=bot, color=cols[ci],
                   label=c, edgecolor='white', linewidth=0.6)
            bot += M[:, ci]
        ax.set_title(t, fontsize=13, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels([f'client {k}' for k in x],
                                             rotation=45, fontsize=8.5)
        ax.set_ylabel('samples held')
        ax.set_ylim(0, max(iid.sum(1).max(), non.sum(1).max()) * 1.06)
    axes[0, 0].text(0.5, 0.94, 'every client is a copy of the same mix',
                    transform=axes[0, 0].transAxes, ha='center', fontsize=10,
                    style='italic', color='#333333')
    axes[0, 1].text(0.5, 0.94, 'every client is different',
                    transform=axes[0, 1].transAxes, ha='center', fontsize=10,
                    style='italic', color='#B4441F')
    axes[0, 1].legend(fontsize=8, ncol=1, frameon=False,
                      loc='center left', bbox_to_anchor=(1.01, 0.5))

    # ── row 2: the rarest class on its own, absolute counts ─────────────────
    rare = len(classes) - 1
    for ax, M, t, col in [(axes[1, 0], iid, 'IID', '#4C72B0'),
                          (axes[1, 1], non, 'non-IID', '#D85A30')]:
        v = M[:, rare]
        ax.bar(x, v, 0.72, color=col, edgecolor='white', linewidth=0.6)
        for k, val in enumerate(v):
            ax.text(k, val, f'{val}', ha='center', va='bottom', fontsize=8.5)
        ax.set_title(f'rarest class only: {classes[rare]}  —  {t}', fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels([f'c{k}' for k in x], fontsize=9)
        ax.set_ylabel('samples')
        ax.set_ylim(0, max(iid[:, rare].max(), non[:, rare].max()) * 1.25 + 1)

    fig.suptitle(f'{name}: how the same data is split across 10 clients',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    p = f'{FIGS}/simple_partition_{name.split("-")[0].lower()}.png'
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f'[fig] {p}')
    print(f'   {classes[rare]}: IID gives each client '
          f'{iid[:, rare].min()}-{iid[:, rare].max()}; '
          f'non-IID gives {non[:, rare].min()}-{non[:, rare].max()}')


if __name__ == '__main__':
    plot('CICIDS2017', lambda a: load_ids_dataset(n_clients=10, alpha=a,
                                                  max_per_class=5000, verbose=False),
         list(CLASS_NAMES))
    plot('UNSW-NB15', lambda a: load_unsw_dataset(n_clients=10, alpha=a,
                                                  max_per_class=5000, verbose=False),
         list(CLASS_NAMES_UNSW))
