"""
null_control.py — how often does the drift detector fire when there is no drift?

Every detector result we have reports detection rate ("4/4 drifts found"). That is
the one metric a detector that fires on everything maximises, and our firing
counts are suspicious: at alpha=100, 9.8 of 10 clients fire on every concept, and
because any single firing triggers adaptation, an always-firing detector is
behaviourally identical to the oracle — which is exactly the result we report.

So: build a stream with NO drift and count firings. Concept 0 (BENIGN / Normal) is
split into 5 pseudo-concepts drawn from one distribution. Phase 0 trains on the
first; the remaining four are streamed exactly as the real pipeline streams real
concepts. Every firing is a false positive.

Reports, per detector mode:
  * false positives  — client-concepts that fired, out of 40 (10 clients x 4)
  * concept-level FP — pseudo-concepts where >=1 client fired (what triggers
                       adaptation in the real system), out of 4
  * MTFA             — mean samples streamed before a false alarm; small values
                       mean the detector fires almost immediately

and the same latency figure on the REAL stream, so true- and false-alarm
behaviour can be compared directly.

Run: python null_control.py [--dataset cicids|unsw|both]
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_loader, unsw_loader
from ids_loader import load_ids_dataset, IDSDNN, CLASS_NAMES, CKPT_DIR, N_FEATURES
from unsw_loader import (load_unsw_dataset, UNSW_DNN, CLASS_NAMES_UNSW,
                         N_FEATURES_UNSW)
import ids_cda_pipeline as P
import unsw_cda_pipeline as U
from drift_channels import HybridDriftDetector, train_ae, recon_scores

SEEDS = [42, 43, 44]
MODES = ['conf', 'recon', 'hybrid']


def null_stream(concepts):
    """Split concept 0 into len(concepts) pseudo-concepts. No drift anywhere."""
    n = len(concepts)
    out = []
    for i in range(n):
        cd = []
        for Xk, yk in concepts[0]['client_data']:
            if len(Xk) == 0:
                cd.append((Xk, yk))
                continue
            idx = np.array_split(np.arange(len(Xk)), n)[i]
            cd.append((Xk[idx], yk[idx]))
        out.append({'name': f'null_{i}', 'description': 'no drift',
                    'client_data': cd})
    return out


def phase0(concepts, mod, model_cls, local_train, device, n_classes):
    """Train the classifier on pseudo-concept 0, exactly as the pipeline does."""
    mem = [mod.LongTermMemory() for _ in range(mod.N_CLIENTS)]
    for k, (Xk, yk) in enumerate(concepts[0]['client_data']):
        if len(Xk) > 0:
            mem[k].collect_and_expand(Xk, yk, concepts[0]['name'])
    gs = model_cls(n_classes=n_classes).to(device).state_dict()
    for _ in range(mod.R_PER_CONCEPT):
        ups = []
        for k in range(mod.N_CLIENTS):
            X_L, y_L = mem[k].get_all()
            if X_L is None or len(X_L) < 2:
                continue
            s, n = local_train(gs, torch.tensor(X_L).float().to(device),
                               torch.tensor(y_L).long().to(device), device)
            ups.append((s, n))
        if ups:
            gs = {k: v.to(device) for k, v in mod.fedavg_aggregate(ups).items()}
    return gs


def stream_and_count(concepts, mod, model_cls, local_train, device, n_classes,
                     n_features, mode, seed):
    """Returns (client_concept_firings, concept_level_firings, latencies, n_streamed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gs = phase0(concepts, mod, model_cls, local_train, device, n_classes)

    model = model_cls(n_classes=n_classes).to(device)
    model.load_state_dict(gs)
    model.eval()

    K = mod.N_CLIENTS
    dets = [HybridDriftDetector(mod.BetaDriftDetector, mode=mode, fusion='or',
                                rng=np.random.default_rng([seed, k]),
                                rng2=np.random.default_rng([seed, 10_000 + k]))
            for k in range(K)]
    aes, scales = [None] * K, [1.0] * K
    if mode in ('recon', 'hybrid'):
        for k, (Xk, _) in enumerate(concepts[0]['client_data']):
            if len(Xk) >= 2:
                aes[k], scales[k] = train_ae(Xk, device, n_features,
                                             seed=seed * 100 + k)

    # warm the windows on phase-0, as the pipeline does
    for k, (Xk, _) in enumerate(concepts[0]['client_data']):
        if len(Xk) == 0:
            continue
        with torch.no_grad():
            cf = model.get_confidence(torch.tensor(Xk).float().to(device))
        rc = recon_scores(aes[k], Xk, scales[k], device) if aes[k] is not None else None
        for i, c in enumerate(cf):
            dets[k].update(float(c), float(rc[i]) if rc is not None else None)

    per_concept, lat, streamed = [], [], 0
    for concept in concepts[1:]:
        fired_idx, sizes = [], [len(Xk) for Xk, _ in concept['client_data']]
        for k, (Xk, _) in enumerate(concept['client_data']):
            if len(Xk) == 0:
                continue
            with torch.no_grad():
                cf = model.get_confidence(torch.tensor(Xk).float().to(device))
            rc = recon_scores(aes[k], Xk, scales[k], device) if aes[k] is not None else None
            for i, c in enumerate(cf):
                streamed += 1
                c_f = float(c)
                r_f = float(rc[i]) if rc is not None else None
                dets[k].update(c_f, r_f)
                if dets[k].should_check(c_f, r_f) and dets[k].detect():
                    dets[k].reset()
                    fired_idx.append(k)
                    lat.append(i + 1)
                    break
        tot = max(sum(sizes), 1)
        per_concept.append({'n': len(fired_idx),
                            'share': sum(sizes[k] for k in fired_idx) / tot})
    # per-concept client counts let every quorum threshold be scored offline
    return per_concept, lat, streamed


def run(which, device, out):
    if which == 'cicids':
        loader = lambda: load_ids_dataset(n_clients=10, alpha=ALPHA,
                                          max_per_class=5000, test_fraction=0.2,
                                          verbose=False)[0]
        mod, cls, lt, nc, nf, ldr = P, IDSDNN, P.local_train_ids, 7, N_FEATURES, ids_loader
    else:
        loader = lambda: load_unsw_dataset(n_clients=10, alpha=ALPHA,
                                           max_per_class=5000, verbose=False)[0]
        mod, cls, lt, nc, nf, ldr = (U, UNSW_DNN, U.local_train_unsw, 10,
                                     N_FEATURES_UNSW, unsw_loader)

    print(f"\n{'='*76}\n{which.upper()} — false alarms on a no-drift stream\n{'='*76}")
    print(f"  {'mode':<8}{'stream':<8}{'clients/cpt':>10}"
          f"{'n>=1':>8}{'n>=3':>7}{'n>=5':>7}{'w>=30%':>9}{'w>=50%':>8}"
          f"{'w>=70%':>8}{'lat':>9}")
    for mode in MODES:
        for label in ('null', 'real'):
            counts, lats, streamed = [], [], []
            for seed in SEEDS:
                ldr.SEED = seed
                concepts = loader()
                stream = null_stream(concepts) if label == 'null' else concepts
                pc, l, n = stream_and_count(stream, mod, cls, lt, device, nc,
                                            nf, mode, seed)
                counts.append(pc); lats += l; streamed.append(n)
            flat = [c for pc in counts for c in pc]
            ns = [c['n'] for c in flat]
            mt = f"{np.mean(lats):.0f}" if lats else "never"
            q = {t: float(np.mean([n >= t for n in ns])) for t in range(1, 11)}
            w = {t: float(np.mean([c['share'] >= t / 100 for c in flat]))
                 for t in (30, 50, 70)}
            print(f"  {mode:<8}{label:<8}{np.mean(ns):>10.2f}"
                  f"{q[1]:>8.0%}{q[3]:>7.0%}{q[5]:>7.0%}"
                  f"{w[30]:>9.0%}{w[50]:>8.0%}{w[70]:>8.0%}{mt:>9}")
            out[f"{which}_{mode}_{label}"] = {
                'clients_per_concept': float(np.mean(ns)),
                'counts': counts, 'quorum_fire_rate': q, 'weight_fire_rate': w,
                'mean_latency': float(np.mean(lats)) if lats else None,
                'mean_streamed': float(np.mean(streamed))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['cicids', 'unsw', 'both'], default='both')
    ap.add_argument('--norm', choices=['batch', 'layer'], default='layer')
    ap.add_argument('--out', default='null_control.json')
    ap.add_argument('--calib', choices=['heldout', 'train'], default='heldout')
    ap.add_argument('--modes', default=','.join(MODES))
    ap.add_argument('--alpha', type=float, default=0.5)
    args = ap.parse_args()
    ids_loader.NORM = unsw_loader.NORM = args.norm
    import drift_channels
    drift_channels.CALIB = args.calib
    globals()['MODES'] = [m.strip() for m in args.modes.split(',') if m.strip()]
    globals()['ALPHA'] = args.alpha
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"alpha={args.alpha} calib={args.calib} norm={args.norm} seeds={SEEDS}  ('real' rows are the true stream, "
          f"for latency comparison)")
    out = {'meta': {'norm': args.norm, 'seeds': SEEDS}}
    for w in (['cicids', 'unsw'] if args.dataset == 'both' else [args.dataset]):
        run(w, device, out)
    with open(os.path.join(CKPT_DIR, args.out), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {os.path.join(CKPT_DIR, args.out)}")
    print("\nA detector that fires on the null stream as readily as on the real")
    print("one is not detecting drift; it is detecting unseen data.")


if __name__ == '__main__':
    main()
