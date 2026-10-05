"""
run_pooled_arm.py -- arm A (IID-pooled reference) under the final-evaluation settings.

final_eval.py leaves arm A out. This adds it for one dataset without touching
final_eval.py: same repaired implementation, loader, seeds and step counting.
It is a post-hoc reference arm, not part of the plan fixed before the
confirmation run; report it as such.

    python run_pooled_arm.py --dataset ids2018 --alphas 0.5 --seeds 600,601 --out final_v2/R2_ids2018_pooled.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_eval as FE
from ids_loader import CKPT_DIR
import aggregators


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['cicids', 'unsw', 'ids2018'], required=True)
    ap.add_argument('--alphas', default='0.5')
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--max-per-class', type=int, default=5000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    FE.set_impl('repaired', 'raw')
    aggregators.AGG = 'fedavg'
    P, M, ev, C, names = FE.setup(a.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(CKPT_DIR, a.out)
    out = json.load(open(path)) if os.path.exists(path) else {
        'meta': {'dataset': a.dataset, 'arm': 'iid_pooled', 'post_hoc': True,
                 'config': FE.config(a.dataset, 'repaired', P, a.max_per_class),
                 'command': ' '.join(sys.argv), 'started': time.strftime('%Y-%m-%d %H:%M:%S')},
        'classes': names, 'runs': {}}
    for alpha in [float(x) for x in a.alphas.split(',')]:
        for seed in [int(x) for x in a.seeds.split(',')]:
            key = f'{alpha:g}/{seed}'
            if 'iid_pooled' in out['runs'].get(key, {}):
                continue                                    # resumable
            concepts, test = FE.load(a.dataset, alpha, seed, a.max_per_class)
            aggregators.reset()
            torch.manual_seed(seed)
            np.random.seed(seed)
            fn = 'local_train_unsw' if P is FE.PU else 'local_train_ids'
            box, restore = FE.count_steps(P, fn)
            t0 = time.time()
            try:
                if P is FE.PU:
                    _, hist = P.run_iid_fedavg_unsw(concepts, test, device, verbose=False)
                else:
                    _, hist = P.run_iid_fedavg(concepts, test, device, n_classes=C, verbose=False)
            finally:
                restore()
            out['runs'].setdefault(key, {})['iid_pooled'] = {
                'history': hist, 'final': hist[-1], 'grad_steps': box['steps'], 'seconds': round(time.time() - t0, 1)}
            print(f"  {a.dataset} a={alpha:g} s={seed} iid_pooled {time.time() - t0:6.0f}s "
                  f"macroF1={hist[-1]['macro_f1']:.4f} steps={box['steps']}", flush=True)
            with open(path, 'w') as f:
                json.dump(out, f, indent=1)
    print('[DONE]', path, flush=True)


if __name__ == '__main__':
    main()
