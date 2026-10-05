"""
make_agg_outputs.py -- thesis tables and figure for the aggregator study
(protocol v3), from checkpoints/agg_analysis.json.

    python make_agg_outputs.py ../thesis_latex
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, 'checkpoints', 'agg_analysis.json')))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'thesis_latex')
DS = [d for d in ('cicids', 'ids2018', 'unsw') if d in A]
NAME = {'cicids': 'CICIDS2017', 'ids2018': 'CSE-CIC-IDS2018', 'unsw': 'UNSW-NB15'}
RULES = ['fedavg', 'fedprox', 'scaffold', 'median']
RN = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'scaffold': 'SCAFFOLD (orig.)', 'median': 'Median'}
# "(orig.)": the SCAFFOLD runs used the original implementation (legacy/aggregators_pre_repair.py),
# whose weighting and random-state defects exclude them from method-level conclusions.
ALPHAS = ['0.1', '0.5', '100']
DASH = '---'


def num(x, digits=4):
    return f"{x:+.{digits}f}".replace('-', '$-$')


def ci(v, digits=4):
    if v is None:
        return DASH
    if v['sd'] == 0 and v['mean'] == 0:
        return '0 (identical)'
    s = f"{num(v['mean'], digits)} $\\pm$ {(v['hi'] - v['lo']) / 2:.{digits}f}"
    return s + '$^{*}$' if v['clear'] else s   # asterisk marks a clear difference


def tab(cols, head, rows):
    return ('\\begin{tabular}{@{}' + cols + '@{}}\n\\toprule\n' + head + ' \\\\\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def block(arm, fmt):
    rows = []
    for d in DS:
        for al in ALPHAS:
            key = f'{al}/{arm}'
            if key not in A[d]['scores']:
                continue
            rows.append(f"{NAME[d] if al == ALPHAS[0] else ''} & {al} & " + fmt(d, key) + ' \\\\')
        rows.append('\\addlinespace')
    return rows[:-1]


def t_scores(arm='rehearsal'):
    def fmt(d, key):
        sc = A[d]['scores'][key]
        return ' & '.join(f"{sc[r]['mean']:.4f} $\\pm$ {sc[r]['sd']:.4f}" if r in sc else DASH for r in RULES)
    head = '\\textbf{Dataset} & $\\alpha$ & ' + ' & '.join(f'\\textbf{{{RN[r]}}}' for r in RULES)
    return tab('llcccc', head, block(arm, fmt))


def t_contrasts(arm='rehearsal'):
    def fmt(d, key):
        vs = A[d]['vs_fedavg'][key]
        return ' & '.join(ci(vs.get(r)) for r in RULES[1:])
    head = '\\textbf{Dataset} & $\\alpha$ & ' + ' & '.join(f'\\textbf{{{RN[r]} $-$ FedAvg}}' for r in RULES[1:])
    return tab('llccc', head, block(arm, fmt))


def fig(out):
    fig, axs = plt.subplots(1, len(DS), figsize=(5.9, 2.3))
    style = {'fedavg': ('#1b6f8a', 'o'), 'fedprox': ('#6b8e23', 's'), 'scaffold': ('#b5462f', '^'),
             'median': ('#7b3294', 'D')}
    from scipy import stats
    for ax, d in zip(np.atleast_1d(axs), DS):
        x = np.arange(len(ALPHAS))
        for r in RULES:
            m, h = [], []
            for al in ALPHAS:
                v = np.array(A[d]['scores'][f'{al}/rehearsal'][r]['per_seed'])
                m.append(v.mean())
                h.append(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
            ax.errorbar(x + (RULES.index(r) - 1.5) * 0.06, m, yerr=h, color=style[r][0], marker=style[r][1],
                        ms=3.5, lw=1.1, capsize=2, label=RN[r])
        ax.set_xticks(x)
        ax.set_xticklabels([f'$\\alpha$={a}' for a in ALPHAS], fontsize=7)
        ax.set_title(NAME[d], fontsize=8)
        ax.grid(axis='y', alpha=.2)
    np.atleast_1d(axs)[0].set_ylabel('macro-F$_1$ (rehearsal arm)', fontsize=7.5)
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(.5, -.06))
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    T = os.path.join(ROOT, 'tables')
    for name, body in [('agg_scores.tex', t_scores()), ('agg_contrasts.tex', t_contrasts()),
                       ('agg_contrasts_B.tex', t_contrasts('nonstat')), ('agg_contrasts_C.tex', t_contrasts('cda'))]:
        open(os.path.join(T, name), 'w', encoding='utf-8', newline='\n').write(body)
    fig(os.path.join(ROOT, 'images', 'fig_aggregators.pdf'))
    print('aggregator tables and figure written')
