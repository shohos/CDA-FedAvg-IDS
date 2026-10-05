"""
final_eval.py -- the final evaluation for the thesis: a fixed configuration,
fixed seeds, full per-concept histories, and FL-IIDS-derived arms on identical
streams.

Arms (A = IID-pooled is left out: it is a 19x-budget ceiling already reported)
  iid_stream   A2  compute-matched IID reference (sees all concepts shuffled)
  nonstat      B   ordered stream, no memory
  rehearsal    B2  memory, adapts at every concept boundary (always adapt)
  cda          C   memory + hybrid drift detector, quorum 5
  cda_q1       C   the same with quorum 1 (the source method's any-one rule)
  fliids       FL-IIDS-derived arm, Eq. 9 loss (see fl_iids.py)
  fliids_ce    the same memory and schedule, plain cross-entropy (F0)
  fliids_ce_half / fliids_gb / fliids_ce_ls   loss variants (fl_iids.VARIANTS)

--impl legacy    reproduces the original runs (checkpoints/final_*.json).
--impl repaired  applies the 2026-09-26 audit repairs (AUDIT_REPAIR_PLAN.md):
                 causal preprocessing, CUDA RNG isolation, memory bound L,
                 FL-IIDS storage bound and fidelity switches.

Two kinds of measurement are kept separate:
  retrospective retention  every concept-end evaluation on the pooled test set
                           (the final row is the headline score)
  forward-stream           each arriving concept's held-out rows, scored just
                           before the model trains on that concept

    python final_eval.py --impl repaired --dataset cicids --alphas 0.5 --seeds 41 --out dev.json
"""
import argparse, hashlib, json, os, platform, sys, time
import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_loader, unsw_loader, ids2018_loader, drift_channels, aggregators
import ids_cda_pipeline as PI
import unsw_cda_pipeline as PU
from ids_loader import load_ids_dataset, IDSDNN, evaluate_ids, CKPT_DIR
from unsw_loader import load_unsw_dataset, UNSW_DNN, evaluate_unsw, CLASS_NAMES_UNSW
from ids2018_loader import load_ids2018_dataset, CLASS_NAMES_2018, N_CONCEPTS_2018_DAILY
import fl_iids

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ['iid_stream', 'nonstat', 'rehearsal', 'cda', 'cda_q1', 'fliids', 'fliids_ce']
FLIIDS_MODE = {'fliids': 'fliids', 'fliids_ce': 'ce', 'fliids_ce_half': 'ce_half',
               'fliids_gb': 'gb', 'fliids_ce_ls': 'ce_ls',
               'fliids_norm': 'fliids', 'fliids_ce_ls_norm': 'ce_ls'}
# FL-IIDS's Eq. 8 target is ambiguous and no author code is public, so under
# --impl repaired both stable readings are run as arms (fl_iids.LS_TARGET notes):
# arms ending in _norm use the normalised target, the others the raw one.
DETECTOR, FUSION, QUORUM = 'hybrid', 'or', 5
FLIIDS_LR = None      # None = the shared learning rate
SOURCES = ['final_eval.py', 'ids_loader.py', 'unsw_loader.py', 'ids2018_loader.py',
           'ids_cda_pipeline.py', 'unsw_cda_pipeline.py', 'drift_channels.py',
           'fl_iids.py', 'memory_bound.py', 'aggregators.py']


def set_impl(impl, ls_target):
    """Every switch the audit repairs added, set together."""
    rep = impl == 'repaired'
    drift_channels.ISOLATE_CUDA_RNG = rep
    ids_loader.PREPROC = unsw_loader.PREPROC = 'causal' if rep else 'all_train'
    PI.MEMORY_BOUND = PU.MEMORY_BOUND = fl_iids.MEMORY_BOUND = rep
    fl_iids.TEACHER = 'old_only' if rep else 'full'
    fl_iids.LS_TARGET = ls_target if rep else 'normalised'


def setup(which):
    """Returns (pipeline module, model class, evaluate fn, n_classes, class names)."""
    if which == 'unsw':
        return PU, UNSW_DNN, evaluate_unsw, len(CLASS_NAMES_UNSW), CLASS_NAMES_UNSW
    if which == 'ids2018':
        PI.N_CONCEPTS = N_CONCEPTS_2018_DAILY
        PI.R_PER_CONCEPT = 5
        PI.N_ROUNDS = 5 * N_CONCEPTS_2018_DAILY
        PI.CLASS_NAMES = CLASS_NAMES_2018
        ids_loader.CLASS_NAMES = CLASS_NAMES_2018      # evaluate_ids reads this
        return PI, IDSDNN, evaluate_ids, len(CLASS_NAMES_2018), CLASS_NAMES_2018
    return PI, IDSDNN, evaluate_ids, len(ids_loader.CLASS_NAMES), ids_loader.CLASS_NAMES


def load(which, alpha, seed, max_pc):
    ids_loader.SEED = unsw_loader.SEED = ids2018_loader.SEED = seed
    if which == 'cicids':
        c, t, _, _, _ = load_ids_dataset(n_clients=10, alpha=alpha, max_per_class=max_pc,
                                         test_fraction=0.2, verbose=False)
    elif which == 'unsw':
        c, t, _, _ = load_unsw_dataset(n_clients=10, alpha=alpha, max_per_class=max_pc,
                                       verbose=False)
    else:
        c, t, _, _, _ = load_ids2018_dataset(n_clients=10, alpha=alpha,
                                             max_per_class=max_pc, verbose=False,
                                             mode='daily')
    return c, t


def count_steps(P, fn_name):
    """Wrap P.<fn_name> so every mini-batch step it takes is counted."""
    orig = getattr(P, fn_name)
    box = {'steps': 0}

    def wrapped(global_state, X, y, device, *a, **kw):
        E = kw.get('E') or P.LOCAL_EPOCHS
        B = kw.get('B') or P.BATCH_SIZE
        n = len(X)
        if n >= 2:
            box['steps'] += E * sum(1 for s in range(0, n, B) if min(B, n - s) >= 2)
        return orig(global_state, X, y, device, *a, **kw)
    setattr(P, fn_name, wrapped)
    return box, lambda: setattr(P, fn_name, orig)


def forward_metrics(model_cls, state, X, y, device, n_classes, names):
    """Scores on an arriving concept's held-out rows. Class 0 is benign in every
    dataset, so 'detection' is the share of attack rows flagged as any attack."""
    m = model_cls(n_classes=n_classes).to(device)
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(X).float().to(device)).argmax(1).cpu().numpy()
    present = sorted(int(c) for c in np.unique(y))
    rec = recall_score(y, p, labels=present, average=None, zero_division=0)
    att, ben = y != 0, y == 0
    return {'macro_f1': float(f1_score(y, p, labels=present, average='macro', zero_division=0)),
            'accuracy': float((p == y).mean()),
            'detection': float((p[att] != 0).mean()) if att.any() else None,
            'false_alarm': float((p[ben] != 0).mean()) if ben.any() else None,
            'recall': {names[c]: float(r) for c, r in zip(present, rec)}}


def run_arm(which, arm, concepts, test, device, seed):
    P, M, ev, C, names = setup(which)
    aggregators.reset()                  # SCAFFOLD state never crosses runs
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    info = {}
    fwd = []
    cuda = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    def on_start(i, state):
        X, y = concepts[i]['test']
        with torch.random.fork_rng(devices=cuda):          # scoring must not move any RNG
            fwd.append({'concept': concepts[i]['name'], 'index': i,
                        **forward_metrics(M, state, X, y, device, C, names)})

    if arm.startswith('fliids'):
        shared, target = P.SGD_LR, fl_iids.LS_TARGET
        P.SGD_LR = FLIIDS_LR or shared
        if arm.endswith('_norm'):
            fl_iids.LS_TARGET = 'normalised'
        try:
            _, hist, info = fl_iids.run_fliids(P, M, ev, concepts, test, device, C,
                                               mode=FLIIDS_MODE[arm], on_concept_start=on_start)
        finally:
            P.SGD_LR, fl_iids.LS_TARGET = shared, target
        info['lr'] = FLIIDS_LR or shared
        info['ls_target'] = 'normalised' if arm.endswith('_norm') else target
        detected = None
    else:
        fn = 'local_train_unsw' if P is PU else 'local_train_ids'
        box, restore = count_steps(P, fn)
        P.QUORUM = 1 if arm == 'cda_q1' else QUORUM
        arm = 'cda' if arm == 'cda_q1' else arm
        kw = dict(verbose=False, on_concept_start=on_start)
        try:
            if P is PU:
                runners = {
                    'iid_stream': lambda: PU.run_iid_stream_fedavg_unsw(concepts, test, device, verbose=False),
                    'nonstat': lambda: PU.run_nonstationary_fedavg_unsw(concepts, test, device, **kw),
                    'rehearsal': lambda: PU.run_cda_fedavg_unsw(concepts, test, device, always_adapt=True, **kw),
                    'cda': lambda: PU.run_cda_fedavg_unsw(concepts, test, device,
                                                          detector_mode=DETECTOR, fusion=FUSION, **kw)}
            else:
                runners = {
                    'iid_stream': lambda: PI.run_iid_stream_fedavg(concepts, test, device, n_classes=C, verbose=False),
                    'nonstat': lambda: PI.run_nonstationary_fedavg(concepts, test, device, n_classes=C, **kw),
                    'rehearsal': lambda: PI.run_cda_fedavg_ids(concepts, test, device, n_classes=C,
                                                               always_adapt=True, **kw),
                    'cda': lambda: PI.run_cda_fedavg_ids(concepts, test, device, n_classes=C,
                                                         detector_mode=DETECTOR, fusion=FUSION, **kw)}
            res = runners[arm]()
        finally:
            restore()
            P.QUORUM = QUORUM
        hist = res[1]
        detected = res[2] if len(res) > 2 else None
        info = {'grad_steps': box['steps']}
    clean = [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
             for r in hist]
    return {'history': clean, 'final': clean[-1], 'forward': fwd, 'detected': detected,
            'seconds': round(time.time() - t0, 1), **info}


def concept_classes(concepts, n_classes):
    """Which classes each concept actually contains (all clients together)."""
    out = []
    for c in concepts:
        ys = [y for _, y in c['client_data'] if len(y)]
        present = np.unique(np.concatenate(ys)) if ys else []
        out.append({'name': c['name'], 'classes': [int(v) for v in present if v < n_classes]})
    return out


# ── provenance ────────────────────────────────────────────────────────────────

def _sha(path, cache):
    st = os.stat(path)
    key = f'{path}|{st.st_size}|{int(st.st_mtime)}'
    if key not in cache:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for b in iter(lambda: f.read(1 << 22), b''):
                h.update(b)
        cache[key] = h.hexdigest()
    return {'file': os.path.relpath(path, HERE), 'bytes': st.st_size, 'sha256': cache[key]}


def data_files(which):
    if which == 'unsw':
        d = unsw_loader.DATA_DIR
        return [os.path.join(d, unsw_loader.TRAIN_CSV), os.path.join(d, unsw_loader.TEST_CSV)]
    if which == 'ids2018':
        return sorted({os.path.join(ids2018_loader.DATA_DIR_2018, f)
                       for _, fs, _, _ in ids2018_loader.CONCEPTS_2018_DAILY for f in fs})
    d = ids_loader.DATA_DIR_IMPROVED if ids_loader.DATA_VERSION == 'improved' else ids_loader.DATA_DIR
    base = (ids_loader.CONCEPT_FILES_IMPROVED if ids_loader.DATA_VERSION == 'improved'
            else ids_loader.CONCEPT_FILES)
    return sorted({os.path.join(d, f) for _, fs, _ in base for f in fs})


def config(which, impl, P, max_pc):
    """Everything that changes a result. Resume refuses a file with a different one."""
    import sklearn, scipy, pandas
    cache_path = os.path.join(CKPT_DIR, 'dataset_fingerprints.json')
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    cfg = {
        'impl': impl, 'dataset': which, 'max_per_class': max_pc,
        'preprocessing': ids_loader.PREPROC if which != 'unsw' else unsw_loader.PREPROC,
        'data_version': ids_loader.DATA_VERSION, 'concept_mode': ids_loader.CONCEPT_MODE,
        'norm': ids_loader.NORM, 'optimizer': P.OPTIMIZER, 'lr': P.SGD_LR,
        'local_epochs': P.LOCAL_EPOCHS, 'batch_size': P.BATCH_SIZE, 'clip_norm': P.CLIP_NORM,
        'rounds_per_concept': P.R_PER_CONCEPT, 'n_clients': P.N_CLIENTS,
        'memory': {'L': P.L_CAPACITY, 'per_class_per_concept': P.PER_CLASS,
                   'rule': P.MEMORY_RULE, 'bound': P.MEMORY_BOUND},
        'detector': {'mode': DETECTOR, 'fusion': FUSION, 'quorum': QUORUM,
                     'quorum_mode': P.QUORUM_MODE, 'lambda': P.LAM, 'delta': P.DELTA,
                     'window': P.N_MAX, 'ae_calibration': drift_channels.CALIB,
                     'ae_scale_factor': drift_channels.SCALE_FACTOR,
                     'ae_defaults': 'epochs=20 batch=100 lr=1e-3 holdout=0.2',
                     'isolate_cuda_rng': drift_channels.ISOLATE_CUDA_RNG},
        'aggregation': {'rule': aggregators.AGG,
                        'mu': aggregators.MU if aggregators.AGG == 'fedprox' else None},
        'fliids': {'gamma': [fl_iids.GAMMA_GB, fl_iids.GAMMA_LS], 'lr': FLIIDS_LR or P.SGD_LR,
                   'teacher': fl_iids.TEACHER, 'ls_target': fl_iids.LS_TARGET,
                   'memory_bound': fl_iids.MEMORY_BOUND, 'variants': fl_iids.VARIANTS},
        'code_sha256': {f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest()
                        for f in SOURCES},
        'data_sha256': [_sha(p, cache) for p in data_files(which)],
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'numpy': np.__version__, 'sklearn': sklearn.__version__,
                     'scipy': scipy.__version__, 'pandas': pandas.__version__,
                     'cuda': torch.version.cuda,
                     'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    json.dump(cache, open(cache_path, 'w'), indent=1)
    return json.loads(json.dumps(cfg))        # tuples -> lists, so it compares with a loaded file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', choices=['legacy', 'repaired'], required=True)
    ap.add_argument('--ls-target', choices=['normalised', 'raw'], default='raw',
                    help="FL-IIDS Eq. 8 target under --impl repaired; 'raw' is the "
                         "paper's description, chosen for fidelity (not by score)")
    ap.add_argument('--dataset', choices=['cicids', 'unsw', 'ids2018'], required=True)
    ap.add_argument('--alphas', default='0.1,0.5,1.0,100')
    ap.add_argument('--seeds', default='47,48,49,50,51,52,53,54,55,56')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--lr', type=float, default=None, help='override for every arm')
    ap.add_argument('--fliids-lr', type=float, default=None, help='FL-IIDS arms only')
    ap.add_argument('--max-per-class', type=int, default=5000)
    ap.add_argument('--agg', choices=['fedavg', 'fedprox', 'scaffold', 'median'], default='fedavg',
                    help='server aggregation rule (aggregators.py)')
    ap.add_argument('--mu', type=float, default=None, help='FedProx proximal weight')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    set_impl(a.impl, a.ls_target)
    aggregators.AGG = a.agg
    if a.mu is not None:
        aggregators.MU = a.mu
    if a.lr:
        PI.SGD_LR = PU.SGD_LR = a.lr
    global FLIIDS_LR
    FLIIDS_LR = a.fliids_lr
    PI.QUORUM = PU.QUORUM = QUORUM
    alphas = [float(x) for x in a.alphas.split(',')]
    seeds = [int(x) for x in a.seeds.split(',')]
    arms = [x for x in a.arms.split(',') if x]
    bad = [x for x in arms if x not in ARMS and x not in FLIIDS_MODE]
    if bad:
        sys.exit(f'unknown arms {bad}')
    if a.agg != 'fedavg' and any(x.startswith('fliids') or x == 'iid_stream' for x in arms):
        sys.exit('the aggregator study covers nonstat, rehearsal, cda and cda_q1 only')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P, _, _, C, names = setup(a.dataset)
    cfg = config(a.dataset, a.impl, P, a.max_per_class)
    path = os.path.join(CKPT_DIR, a.out)
    if os.path.exists(path):
        out = json.load(open(path))
        old = out.get('meta', {}).get('config')
        if old != cfg:
            diff = sorted(k for k in set(cfg) | set(old or {}) if (old or {}).get(k) != cfg.get(k))
            sys.exit(f'{a.out} was written with different settings ({diff}); '
                     f'refusing to resume into it')
    else:
        out = {'meta': {'dataset': a.dataset, 'alphas': alphas, 'seeds': seeds, 'arms': arms,
                        'config': cfg, 'command': ' '.join(sys.argv),
                        'started': time.strftime('%Y-%m-%d %H:%M:%S')},
               'classes': names, 'runs': {}}
    print(f"final_eval impl={a.impl} {a.dataset} alphas={alphas} seeds={seeds} arms={arms} "
          f"lr={P.SGD_LR} out={a.out}", flush=True)
    wall = time.time()
    for alpha in alphas:
        for seed in seeds:
            key = f'{alpha:g}/{seed}'
            cell = out['runs'].setdefault(key, {})
            todo = [x for x in arms if x not in cell]      # resumable
            if not todo:
                continue
            concepts, test = load(a.dataset, alpha, seed, a.max_per_class)
            cell['concepts'] = concept_classes(concepts, C)
            for arm in todo:
                r = run_arm(a.dataset, arm, concepts, test, device, seed)
                cell[arm] = r
                print(f"  {a.dataset} a={alpha:g} s={seed} {arm:<15} {r['seconds']:>6.0f}s "
                      f"macroF1={r['final']['macro_f1']:.4f} steps={r.get('grad_steps')}"
                      + (f" drifts={len(r['detected'])}" if r['detected'] is not None else ''),
                      flush=True)
                with open(path, 'w') as f:
                    json.dump(out, f, indent=1)
    print(f"[DONE] {(time.time() - wall) / 60:.1f} min -> {path}", flush=True)


if __name__ == '__main__':
    main()
