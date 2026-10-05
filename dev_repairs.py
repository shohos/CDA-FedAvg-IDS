"""
dev_repairs.py -- development check for the audit repairs, on consumed seeds only.

For the arms the repairs touch, run each repair on its own and all together, so
the before/after accounting can say which repair moved which number.

  legacy   the original implementation
  preproc  + causal preprocessing only
  memory   + memory bound only
  rng      + CUDA RNG isolation only (changes the detector arm only)
  all      every repair (= final_eval.py --impl repaired)

    python dev_repairs.py --dataset cicids --seeds 42,43,44 --out audit_repair_dev/decomp_cicids.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_eval as FE
import ids_loader, unsw_loader, drift_channels, fl_iids
import ids_cda_pipeline as PI
import unsw_cda_pipeline as PU

CONFIGS = {'legacy': set(), 'preproc': {'preproc'}, 'memory': {'memory'}, 'rng': {'rng'},
           'teacher': {'teacher'}, 'all': {'preproc', 'memory', 'rng', 'teacher'}}
ONLY = {'rng': {'cda', 'cda_q1'}, 'teacher': {'fliids', 'fliids_ce_ls'}}   # arms a repair can move


def apply(on):
    FE.set_impl('legacy', 'normalised')
    if 'preproc' in on:
        ids_loader.PREPROC = unsw_loader.PREPROC = 'causal'
    if 'memory' in on:
        PI.MEMORY_BOUND = PU.MEMORY_BOUND = fl_iids.MEMORY_BOUND = True
    if 'rng' in on:
        drift_channels.ISOLATE_CUDA_RNG = True
    if 'teacher' in on:
        fl_iids.TEACHER, fl_iids.LS_TARGET = 'old_only', 'raw'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--arms', default='rehearsal,cda,fliids')
    ap.add_argument('--configs', default=','.join(CONFIGS))
    ap.add_argument('--fliids-lr', type=float, default=0.02, help='as in the final protocol')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    FE.FLIIDS_LR = a.fliids_lr
    for s in map(int, a.seeds.split(',')):
        assert s < 47, 'development uses consumed seeds only (41-46)'
    path = os.path.join(FE.CKPT_DIR, a.out)
    out = json.load(open(path)) if os.path.exists(path) else {'dataset': a.dataset,
                                                               'alpha': a.alpha, 'runs': {}}
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P, _, _, C, _ = FE.setup(a.dataset)
    PI.QUORUM = PU.QUORUM = FE.QUORUM
    for cfg in a.configs.split(','):
        for seed in map(int, a.seeds.split(',')):
            cell = out['runs'].setdefault(f'{cfg}/{seed}', {})
            todo = [x for x in a.arms.split(',') if x not in cell
                    and (cfg not in ONLY or x in ONLY[cfg])]
            if not todo:
                continue
            apply(CONFIGS[cfg])
            concepts, test = FE.load(a.dataset, a.alpha, seed, 5000)
            for arm in todo:
                apply(CONFIGS[cfg])
                r = FE.run_arm(a.dataset, arm, concepts, test, dev, seed)
                cell[arm] = {k: r[k] for k in ('final', 'forward', 'detected', 'seconds')
                             if k in r}
                cell[arm]['memory_rows'] = [h.get('memory_rows') for h in r['history']]
                for k in ('memory_records', 'memory_stored', 'grad_steps'):
                    if k in r:
                        cell[arm][k] = r[k]
                print(f"  {a.dataset} {cfg:<8} s={seed} {arm:<10} "
                      f"macroF1={r['final']['macro_f1']:.4f} ({r['seconds']:.0f}s)", flush=True)
                json.dump(out, open(path, 'w'), indent=1)
    print('[DONE]', path)


if __name__ == '__main__':
    main()
