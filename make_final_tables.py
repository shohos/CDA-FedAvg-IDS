"""
make_final_tables.py -- writes the Chapter 5 final-evaluation tables as LaTeX
from the analysis files, so the thesis numbers are never typed by hand.

    python make_final_tables.py ../thesis_latex/tables
      reads checkpoints/final_v2_R2_analysis.json (confirmation run, seeds 600-609)
        and checkpoints/final_v2_R1_analysis.json (paired rerun, seeds 47-56, whose
        before_after block compares it with the original implementation)
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, 'checkpoints')
A = json.load(open(os.path.join(CK, 'final_v2_R2_analysis.json')))
R1 = json.load(open(os.path.join(CK, 'final_v2_R1_analysis.json')))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'thesis_latex', 'tables')
DS = [d for d in ('cicids', 'ids2018', 'unsw') if d in A]
NAME = {'cicids': 'CICIDS2017', 'ids2018': 'CSE-CIC-IDS2018', 'unsw': 'UNSW-NB15'}
ARM = {'iid_stream': 'A2\\ IID-stream (matched)', 'nonstat': 'B\\ \\ Non-stationary',
       'rehearsal': 'B2\\ Rehearsal (always adapt)', 'cda_q1': 'C\\ \\ CDA-FedAvg, any one client',
       'cda': 'C\\ \\ CDA-FedAvg, quorum 5', 'fliids': 'F\\ \\ FL-IIDS-derived, raw target',
       'fliids_norm': 'Fn\\ FL-IIDS-derived, normalised target',
       'fliids_ce': 'F0\\ FL-IIDS-derived, plain loss'}
MAIN = ['iid_stream', 'nonstat', 'rehearsal', 'cda_q1', 'cda', 'fliids', 'fliids_norm', 'fliids_ce']
DASH = '---'


def ms(v):
    return f"{v['mean']:.4f} $\\pm$ {v['sd']:.4f}"


def num(x, digits=4):
    """Signed number with a proper minus sign in LaTeX."""
    s = f"{x:+.{digits}f}"
    if float(s) == 0:                       # no signed zero such as -0.0000
        s = f"{0:.{digits}f}"
    return s.replace('-', '$-$')


def ci(v, digits=4):
    """Mean +- half-width of the (symmetric) 95% t-interval; an asterisk when it excludes 0.
    A difference that is exactly zero on every seed (bit-identical runs) prints as 0."""
    if v is None:
        return DASH
    if v['sd'] == 0 and v['mean'] == 0:
        return '0 (identical)'
    s = f"{num(v['mean'], digits)} $\\pm$ {(v['hi'] - v['lo']) / 2:.{digits}f}"
    return s + '$^{*}$' if v['clear'] else s   # asterisk marks a clear difference


def write(name, body):
    with open(os.path.join(OUT, name), 'w', encoding='utf-8', newline='\n') as f:
        f.write(body)


def tab(cols, head, rows):
    return ('\\begin{tabular}{@{}' + cols + '@{}}\n\\toprule\n' + head + ' \\\\\n\\midrule\n'
            + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n')


def dshead(first='Arm'):
    return f'\\textbf{{{first}}} & ' + ' & '.join(f'\\textbf{{{NAME[d]}}}' for d in DS)


def con(d, A, key):
    return A_.get(d, {}).get('contrasts', {}).get(A, {}).get(key)


A_ = A


def t_validation():
    rows = [ARM[a] + ' & ' + ' & '.join(ms(A[d]['scores']['0.5'][a]) if a in A[d]['scores']['0.5'] else DASH
                                        for d in DS) + ' \\\\' for a in MAIN]
    return tab('l' + 'c' * len(DS), dshead(), rows)


def t_contrasts():
    lab = [('B2-B', '0.5', 'Memory: B2 $-$ B'),
           ('C5-B2', '0.5', 'Detector, quorum 5: C $-$ B2'),
           ('C5-B2', '0.1', 'Detector, quorum 5, $\\alpha = 0.1$'),
           ('C1-B2', '0.5', 'Detector, any one client: C $-$ B2'),
           ('B2-A2', '0.5', 'B2 $-$ training-matched reference A2')]
    rows = [l + ' & ' + ' & '.join(ci(con(d, al, k)) if al in A[d]['contrasts'] else DASH for d in DS) + ' \\\\'
            for k, al, l in lab]
    return tab('l' + 'c' * len(DS), dshead('Difference'), rows)


def per_alpha(keys):
    rows = []
    for d in DS:
        for al in ('0.1', '0.5', '1', '100'):
            if al not in A[d]['contrasts']:
                continue
            rows.append(f"{NAME[d]} & {al} & " + ' & '.join(ci(A[d]['contrasts'][al].get(k)) for k in keys) + ' \\\\')
        rows.append('\\addlinespace')
    return rows[:-1]


def t_fliids():
    head = ('\\textbf{Dataset} & $\\alpha$ & \\textbf{F $-$ B2} & \\textbf{Fn $-$ B2} & '
            '\\textbf{F $-$ F0} & \\textbf{Fn $-$ F0}')
    return tab('lccccc', head, per_alpha(['F-B2', 'Fn-B2', 'F-F0', 'Fn-F0']))


def t_lossdecomp():
    head = ('\\textbf{Dataset} & $\\alpha$ & \\textbf{Step size} & \\textbf{Gradient balance} & '
            '\\textbf{Label smoothing (raw)} & \\textbf{Label smoothing (norm.)}')
    return tab('lccccc', head, per_alpha(['F0h-F0', 'Fgb-F0h', 'Fls-F0h', 'Flsn-F0h']))


def t_noniid():
    arms = ['rehearsal', 'cda', 'fliids', 'fliids_norm', 'fliids_ce', 'nonstat']
    ds = [d for d in DS if 'noniid_cost' in A[d]]
    rows = []
    for arm in arms:
        rows.append(ARM[arm] + ' & ' + ' & '.join(ci(A[d]['noniid_cost'][arm].get(al), 3)
                                                  for d in ds for al in ('0.1', '0.5')) + ' \\\\')
    rows.append('\\midrule')
    for arm, lab in (('fliids', 'F\\ \\ extra cost over B2'), ('fliids_norm', 'Fn\\ extra cost over B2')):
        rows.append(lab + ' & ' + ' & '.join(ci(A[d]['noniid_cost_vs_B2'][arm].get(al), 3)
                                             for d in ds for al in ('0.1', '0.5')) + ' \\\\')
    head = (' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{\\textbf{{{NAME[d]}}}}}' for d in ds) + ' \\\\\n'
            + ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(ds))) + '\n'
            + '\\textbf{Arm} & ' + ' & '.join('$\\alpha=0.1$ & $\\alpha=0.5$' for _ in ds))
    return tab('l' + 'cc' * len(ds), head, rows)


def t_forgetting():
    arms = ['nonstat', 'rehearsal', 'cda', 'fliids', 'fliids_norm', 'fliids_ce']
    rows = []
    for d in DS:
        F = A[d]['forgetting']['0.5']
        rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{NAME[d]}}}}} \\\\")
        for arm in arms:
            f = F[arm]
            rows.append(f"\\quad {ARM[arm]} & {f['mean_forgetting_recall']['mean']:.3f} & "
                        f"{f['mean_forgetting_f1']['mean']:.3f} & {f['learned']['mean']:.1f} & "
                        f"{f['learned_then_lost']['mean']:.1f} \\\\")
        rows.append('\\addlinespace')

    def two(a, b):
        return f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}\\textbf{{{a}}}\\\\\\textbf{{{b}}}\\end{{tabular}}"
    head = '\\textbf{Arm} & ' + ' & '.join([two('Forgetting', '(recall)'), two('Forgetting', '($F_1$)'),
                                           two('Classes', 'learned'), two('Learned,', 'then lost')])
    return tab('lcccc', head, rows[:-1])


def t_cost():
    arms = ['iid_stream', 'nonstat', 'rehearsal', 'cda', 'fliids']
    rows = []
    for a in arms:
        cells = []
        for d in DS:
            st = f"{A[d]['cost']['0.5'][a]['mean']:,.0f}".replace(',', '{,}')
            mem = A[d]['memory']['0.5'].get(a)
            cells += [st, f"{mem['mean']:,.0f}".replace(',', '{,}') if mem else DASH]
        rows.append(ARM[a] + ' & ' + ' & '.join(cells) + ' \\\\')
    head = (' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{\\textbf{{{NAME[d]}}}}}' for d in DS) + ' \\\\\n'
            + ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(DS))) + '\n'
            + '\\textbf{Arm} & ' + ' & '.join('Steps & Rows kept' for _ in DS))
    return tab('l' + 'rr' * len(DS), head, rows)


def t_forward():
    arms = ['nonstat', 'rehearsal', 'cda', 'fliids', 'fliids_norm', 'fliids_ce']
    rows = []
    for d in DS:
        fw = A[d]['forward']['0.5']
        rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{NAME[d]}}}}} \\\\")
        for a in arms:
            v = fw[a]
            rows.append(f"\\quad {ARM[a]} & {v['macro_f1']['mean']:.3f} & {v['detection']['mean']:.3f} & "
                        f"{v['false_alarm']['mean']:.3f} \\\\")
        rows.append('\\addlinespace')
    head = '\\textbf{Arm} & \\textbf{\\macroF{}} & \\textbf{Attacks flagged} & \\textbf{Normal flows flagged}'
    return tab('lccc', head, rows[:-1])


def t_beforeafter():
    """Corrected minus original implementation, same seeds (47-56), alpha 0.5. FL-IIDS is
    shown like for like: the original arm used the normalised Eq. 8 target, so it is
    compared with the corrected normalised-target arm Fn."""
    import numpy as np
    sys.path.insert(0, HERE)
    from analyse_final_eval import interval
    leg = json.load(open(os.path.join(CK, 'final_eval_analysis.json')))
    arms = [('iid_stream', 'iid_stream'), ('nonstat', 'nonstat'), ('rehearsal', 'rehearsal'),
            ('cda', 'cda'), ('cda_q1', 'cda_q1'), ('fliids_norm', 'fliids'), ('fliids_ce', 'fliids_ce')]
    rows = []
    for new, old in arms:
        cells = []
        for d in DS:
            a, b = R1[d]['scores']['0.5'].get(new), leg[d]['scores']['0.5'].get(old)
            cells.append(ci(interval(np.array(a['per_seed']) - np.array(b['per_seed'])))
                         if a and b and a['seeds'] == b['seeds'] else DASH)
        lab = 'Fn\\ FL-IIDS-derived (original: F)' if new == 'fliids_norm' else ARM[new]
        rows.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
    return tab('l' + 'c' * len(DS), dshead(), rows)


def t_partition():
    """Table 4.3 from partition_properties.json (regenerated from the real loaders)."""
    P = json.load(open(os.path.join(CK, 'partition_properties.json')))
    rows = []
    for d in ('cicids', 'unsw'):
        for al in ('0.1', '0.5', '1', '100'):
            v = P[d][al]
            r = v['size_ratio_median']
            rs = f"{r:.0f}$\\times$" if r >= 10 else f"{r:.1f}$\\times$"
            rows.append(f"{NAME[d] if al == '0.1' else ''} & {al} & {v['holders_ge1']:.1f} & "
                        f"{v['holders_ge200']:.1f} & {100 * v['top3_share']:.1f}\\% & {rs} \\\\")
        rows.append('\\addlinespace')
    return ('\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n\\textbf{Dataset} & \\alphaD{} & '
            '\\textbf{Clients with} & \\textbf{Clients with} & \\textbf{Top-3} & '
            '\\textbf{Largest $/$} \\\\\n & & \\textbf{$\\ge 1$ of rare class} & '
            '\\textbf{$\\ge 200$ of it} & \\textbf{share} & \\textbf{smallest client} \\\\\n'
            '\\midrule\n' + '\n'.join(rows[:-1]) + '\n\\bottomrule\n\\end{tabular}\n')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [('partition.tex', t_partition), ('final_validation.tex', t_validation),
                     ('final_contrasts.tex', t_contrasts), ('final_fliids.tex', t_fliids),
                     ('final_lossdecomp.tex', t_lossdecomp), ('final_noniid.tex', t_noniid),
                     ('final_forgetting.tex', t_forgetting), ('final_steps.tex', t_cost),
                     ('final_forward.tex', t_forward), ('final_beforeafter.tex', t_beforeafter)]:
        write(name, fn())
    print('tables written to', os.path.abspath(OUT), 'datasets:', DS)
