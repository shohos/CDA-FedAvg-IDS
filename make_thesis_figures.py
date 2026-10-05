"""
make_thesis_figures.py -- the result charts of the thesis, drawn from the saved
result files with one colour per arm and per class.

  fig_res_fivearm.png          Figure 5.1  macro-F1 of the arms at alpha = 0.5
  fig_res_heterogeneity.png    Figure 5.2  macro-F1 as clients become more different
  fig_tabfm_frontend_chart.png Figure 5.4  original against TabFM features
  fig_dirichlet_split.png      Figure 4.5  a real CICIDS2017 split (needs the dataset)

    python make_thesis_figures.py [output_dir]      (default: figures/thesis)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CK = os.path.join(HERE, 'checkpoints')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figures', 'thesis')
os.makedirs(OUT, exist_ok=True)


def load(*p):
    return json.load(open(os.path.join(CK, *p)))


A = load('final_v2_R2_analysis.json')               # confirmation run, seeds 600-609
TABFM = {c['dataset']: c for c in load('tabfm_integration', 'tabfm_summary.json')['cells']}

INK, GRID = '#1F2A36', '#E3E8EC'
ARM = {'pooled': '#9CC3D5', 'shuffled': '#3B75AF', 'nomem': '#9AA0A6', 'always': '#1B6F8A',
       'anyone': '#4E9A5B', 'quorum': '#8E5BA8', 'fliids': '#E08A2E'}
CLASS = {'BENIGN': '#9AA0A6', 'BruteForce': '#3B75AF', 'DoS': '#E08A2E', 'WebAttack': '#4E9A5B',
         'DDoS': '#8E5BA8', 'PortScan': '#C44E52', 'Bot': '#1B6F8A'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK, 'text.color': INK, 'axes.spines.top': False,
                     'axes.spines.right': False})
DS = [('cicids', 'CICIDS2017'), ('ids2018', 'CSE-CIC-IDS2018'), ('unsw', 'UNSW-NB15')]


def sc(d, arm, al='0.5'):
    return A[d]['scores'][al][arm]['mean']


def pooled():
    """The pooled reference was not part of the confirmation plan: CICIDS2017 and
    UNSW-NB15 come from the development runs (seeds 42-46), CSE-CIC-IDS2018 from
    the post hoc run on seeds 600-609 (run_pooled_arm.py)."""
    dev = {d: float(np.mean([r['iid_pooled']['macro_f1'] for r in load(f)[d].values()]))
           for d, f in [('cicids', 'cicids_improved.json'), ('unsw', 'unsw_final_q5.json')]}
    runs = load('final_v2', 'R2_ids2018_pooled.json')['runs'].values()
    dev['ids2018'] = float(np.mean([r['iid_pooled']['final']['macro_f1'] for r in runs]))
    return dev


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), facecolor='white')
    plt.close(fig)
    print('wrote', name)


def five_arm():
    p = pooled()
    arms = [('Pooled reference', lambda d: p[d], ARM['pooled'], '////'),
            ('Shuffled stream', lambda d: sc(d, 'iid_stream'), ARM['shuffled'], None),
            ('No memory', lambda d: sc(d, 'nonstat'), ARM['nomem'], None),
            ('Always-adapt rehearsal', lambda d: sc(d, 'rehearsal'), ARM['always'], None),
            ('CDA-FedAvg (any-one rule)', lambda d: sc(d, 'cda_q1'), ARM['anyone'], None),
            ('CDA-FedAvg (quorum of five)', lambda d: sc(d, 'cda'), ARM['quorum'], None)]
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=250)
    w = 0.135
    for j, (lab, f, c, h) in enumerate(arms):
        b = ax.bar([i + (j - 2.5) * w for i in range(3)], [f(d) for d, _ in DS], w * 0.92,
                   color=c, hatch=h, edgecolor='white', linewidth=0.4, label=lab)
        ax.bar_label(b, fmt='%.2f', fontsize=6.3, padding=1.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for _, n in DS], fontsize=9.5)
    ax.set_ylabel('macro-F1 at the end of the stream')
    ax.set_ylim(0, 1.08)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=3)
    fig.tight_layout()
    save(fig, 'fig_res_fivearm.png')


def heterogeneity():
    arms = [('rehearsal', 'Always-adapt rehearsal', ARM['always']), ('cda_q1', 'CDA-FedAvg (any-one rule)', ARM['anyone']),
            ('cda', 'CDA-FedAvg (quorum of five)', ARM['quorum']), ('fliids', 'Adapted FL-IIDS', ARM['fliids'])]
    als = ['100', '1', '0.5', '0.1']
    xl = ['α = 100\nnear-identical', 'α = 1', 'α = 0.5', 'α = 0.1\nvery different']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=250)
    for ax, (d, name) in zip(axes, [('cicids', 'CICIDS2017'), ('unsw', 'UNSW-NB15')]):
        for j, (k, lab, c) in enumerate(arms):
            b = ax.bar([i + (j - 1.5) * 0.2 for i in range(4)], [sc(d, k, al) for al in als], 0.19, color=c, label=lab)
            ax.bar_label(b, fmt='%.2f', fontsize=5.6, padding=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(xl, fontsize=8)
        ax.set_ylim(0, 1.05 if d == 'cicids' else 0.45)
        ax.set_ylabel('macro-F1')
        ax.set_title(name, fontsize=10.5, fontweight='bold', loc='left')
        ax.grid(axis='y', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, fontsize=8, frameon=False, loc='upper center')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 'fig_res_heterogeneity.png')


def tabfm():
    cells = [('cicids', 'rehearsal', 'CICIDS2017\nalways-adapt rehearsal'),
             ('cicids', 'cda', 'CICIDS2017\nCDA-FedAvg (quorum of five)'),
             ('unsw', 'rehearsal', 'UNSW-NB15\nalways-adapt rehearsal'),
             ('unsw', 'cda', 'UNSW-NB15\nCDA-FedAvg (quorum of five)')]
    raw = [TABFM[d]['raw_' + k] for d, k, _ in cells]
    tab = [TABFM[d]['tabfm_' + k] for d, k, _ in cells]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=250)
    y = list(range(4))[::-1]
    ax.barh([v + 0.19 for v in y], raw, 0.36, color=ARM['nomem'], label='Original features')
    ax.barh([v - 0.19 for v in y], tab, 0.36, color=ARM['always'], label='TabFM features')
    for v, r, t in zip(y, raw, tab):
        ax.text(r + 0.012, v + 0.19, f'{r:.3f}', va='center', fontsize=8)
        ax.text(t + 0.012, v - 0.19, f'{t:.3f}', va='center', fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, _, lab in cells], fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('macro-F1 at the end of the stream')
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.42, -0.13), ncol=2)
    fig.tight_layout()
    save(fig, 'fig_tabfm_frontend_chart.png')


def split():
    import partition_simple as PS
    from ids_loader import load_ids_dataset, CLASS_NAMES
    os.chdir(HERE)                                   # the loaders read the data relative to the code
    load_split = lambda a: load_ids_dataset(n_clients=10, alpha=a, max_per_class=5000, verbose=False)
    classes = list(CLASS_NAMES)
    iid, non = PS.counts(load_split, len(classes), 100.0), PS.counts(load_split, len(classes), 0.5)
    order = np.argsort(-iid.sum(0))
    classes = [classes[i] for i in order]
    iid, non = iid[:, order], non[:, order]
    x = np.arange(iid.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), dpi=250, sharey=True)
    for ax, M, t in [(axes[0], iid, 'Near-identical clients (α = 100)'), (axes[1], non, 'Different clients (α = 0.5)')]:
        bot = np.zeros(len(x))
        for ci, c in enumerate(classes):
            ax.bar(x, M[:, ci], 0.72, bottom=bot, color=CLASS[c], label=c, edgecolor='white', linewidth=0.5)
            bot += M[:, ci]
        ax.set_title(t, fontsize=10.5, fontweight='bold', loc='left')
        ax.set_xticks(x)
        ax.set_xticklabels([str(k + 1) for k in x], fontsize=8)
        ax.set_xlabel('client')
        ax.grid(axis='y', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('training rows held')
    axes[1].legend(fontsize=7.8, frameon=False, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    save(fig, 'fig_dirichlet_split.png')


if __name__ == '__main__':
    five_arm()
    heterogeneity()
    tabfm()
    split()
