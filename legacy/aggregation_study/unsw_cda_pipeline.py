"""
UNSW-NB15 CDA-FedAvg Pipeline.

Runs three experiments on UNSW-NB15 using Dirichlet non-IID partitioning:
  Experiment A: FedAvg IID           (upper bound)
  Experiment B: FedAvg Non-stationary (catastrophic forgetting baseline)
  Experiment C: CDA-FedAvg UNSW      (main contribution)

Concept drift design (attack kill chain — 5 stages):
  Stage 0: Normal baseline (no attacks)
  Stage 1: Reconnaissance + Analysis  (intelligence gathering)
  Stage 2: Exploits + Fuzzers         (active exploitation)
  Stage 3: DoS + Generic              (service disruption)
  Stage 4: Backdoors + Shellcode + Worms (persistence/malware)

Federated setup:
  10 clients, Dirichlet(α=0.5) non-IID, pre-partitioned train/test split
All experiments use identical SGD (lr=0.01, no momentum, clip=10.0).
"""

import numpy as np
import torch
import torch.nn as nn
import os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unsw_loader      # module import so SEED is read at call time, not frozen
from unsw_loader import (
    UNSW_DNN, load_unsw_dataset, evaluate_unsw, count_parameters_unsw,
    CLASS_NAMES_UNSW, N_CLASSES_UNSW, N_FEATURES_UNSW,
    CKPT_DIR, SEED, N_CONCEPTS_UNSW, UNSW_CONCEPTS
)
from scipy.stats import beta as beta_dist
from drift_channels import HybridDriftDetector, train_ae, recon_scores

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Experiment settings ───────────────────────────────────────────────────────

N_CLIENTS     = 10
ALPHA         = 0.5
MAX_PER_CLASS = 5000
N_ROUNDS      = 25
R_PER_CONCEPT = N_ROUNDS // N_CONCEPTS_UNSW   # = 5 rounds per concept
LOCAL_EPOCHS  = 10
BATCH_SIZE    = 100
SGD_LR        = 0.01
CLIP_NORM     = 10.0
QUICK_TEST    = True    # set False for full run

# CDA paper constants
LAM        = 0.05
DELTA      = 100
N_MAX      = 1000
L_CAPACITY = 1400
PER_CLASS  = L_CAPACITY // (2 * N_CLASSES_UNSW)   # = 70


# ── Algorithm 5: BetaDriftDetector ───────────────────────────────────────────

class BetaDriftDetector:
    """Casado et al. Algorithm 5 — see ids_cda_pipeline for the full note."""

    def __init__(self, window_size=N_MAX, delta=DELTA, lam=LAM, rng=None):
        self.window_size = window_size
        self.delta       = delta
        self.lam         = lam
        self.threshold   = -np.log(lam)
        self.window      = []
        self.rng         = rng if rng is not None else np.random.default_rng()

    def should_check(self, conf):
        """Algorithm 4 line 12: run detection with probability e^{-2q}."""
        return float(np.exp(-2.0 * conf)) >= self.rng.random()

    def update(self, conf):
        self.window.append(float(np.clip(conf, 1e-6, 1-1e-6)))
        if len(self.window) > self.window_size:
            self.window.pop(0)

    @staticmethod
    def _fit_beta(x):
        x = np.clip(x, 1e-6, 1-1e-6)
        mu, var = np.mean(x), np.var(x)
        max_v = mu*(1-mu) - 1e-6
        var   = np.clip(var, 1e-8, max_v if max_v > 1e-8 else 1e-6)
        f     = mu*(1-mu)/var - 1
        return max(mu*f, 1e-6), max((1-mu)*f, 1e-6)

    def _ll(self, x, a, b):
        return float(np.sum(beta_dist.logpdf(np.clip(x, 1e-6, 1-1e-6), a, b)))

    def detect(self, return_score=False):
        """Algorithm 5: LLR accumulated over the recent sub-window only."""
        w = np.array(self.window)
        N = len(w)
        if N < 2*self.delta:
            return (False, 0.0) if return_score else False
        sf = 0.0
        for k in range(self.delta, N - self.delta + 1):
            qb, qa = w[:k], w[k:]
            if np.mean(qa) > (1 - self.lam) * np.mean(qb):
                continue
            ab, bb = self._fit_beta(qb)
            aa, ba = self._fit_beta(qa)
            qa_c = np.clip(qa, 1e-6, 1 - 1e-6)
            sk = float(np.sum(beta_dist.logpdf(qa_c, aa, ba)
                              - beta_dist.logpdf(qa_c, ab, bb)))
            sf = max(sf, sk)
        fired = bool(sf > self.threshold)
        return (fired, sf) if return_score else fired

    def reset(self):
        self.window.clear()

    def __len__(self):
        return len(self.window)


# ── Long-Term Memory ──────────────────────────────────────────────────────────

QUORUM_MODE = 'count'   # 'count' = QUORUM clients must fire (Casado's rule is
                        # QUORUM=1). 'weight' = the clients that fired must
                        # collectively hold >= QUORUM_WEIGHT of the concept's
                        # data. Measured motivation: at alpha=0.1 the top 3
                        # clients hold 79-99% of a new concept's samples, so
                        # requiring 5 of 10 is impossible; at alpha=100 they
                        # hold 32-35%, so 3 firing is weak evidence. Weighting
                        # by n_k — which FedAvg already communicates — adapts to
                        # both without a per-alpha constant.
                        #
                        # TESTED AND REJECTED (null_control.py, alpha=0.1):
                        # weighting is actively harmful under extreme
                        # heterogeneity — false alarms 83% (CICIDS) / 100%
                        # (UNSW) against detection of 33% / 25%, i.e. it fires
                        # MORE on a no-drift stream than on real drift. The same
                        # concentration that lets 3 clients clear 50% on genuine
                        # drift lets 1 client clear it on a false alarm. Keep
                        # 'count'. Retained only so the result is reproducible.
QUORUM_WEIGHT = 0.5
QUORUM = 1   # clients that must fire before adaptation is triggered.
             # 1 = Casado's any-one rule. null_control.py shows the any-one
             # rule destroys the channel's client-level signal: 2.2 clients per
             # concept fire on a no-drift stream vs 7.25 on real drift, so a
             # threshold in between separates them.

OPTIMIZER = 'momentum'  # best measured (opt_sweep.py): momentum@0.01 wins on
                        # both datasets and both norms. 'sgd' = published runs.
                        # was 'sgd'  # 'sgd' | 'momentum' | 'adam' | 'adamw'
MOMENTUM  = 0.9      # used by 'momentum'
                     # SGD(lr=0.01, no momentum) was inherited from Casado's HAR
                     # CNN and carried unexamined into a different architecture,
                     # data regime and normalisation. Note for FL: client
                     # optimiser state is rebuilt every round (a fresh model is
                     # constructed per local_train call), so Adam's moments reset
                     # each round — the known limitation that motivates
                     # server-side adaptivity (Reddi et al., ICLR 2021).


def _make_opt(params, lr):
    if OPTIMIZER == 'momentum':
        return torch.optim.SGD(params, lr=lr, momentum=MOMENTUM)
    if OPTIMIZER == 'adam':
        return torch.optim.Adam(params, lr=lr)
    if OPTIMIZER == 'adamw':
        return torch.optim.AdamW(params, lr=lr)
    return torch.optim.SGD(params, lr=lr)


LOCAL_NORM = False   # FedBN (ICLR 2021): keep normalisation layers local instead
                     # of averaging them. Isolates WHY LayerNorm helped — we
                     # claim running-statistic averaging is the culprit, but
                     # LayerNorm changes two things at once (no running stats
                     # AND per-sample normalisation). BatchNorm+LOCAL_NORM has
                     # running stats but never averages them, so it separates
                     # the two explanations.
_LOCAL_NORM_STATE = {}


def _norm_keys(model):
    ks = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.LayerNorm)):
            ks.update(f"{name}.{p}" for p, _ in mod.named_parameters())
            ks.update(f"{name}.{b}" for b, _ in mod.named_buffers())
    return ks


MEMORY_RULE = 'cap'   # 'cap' (ours, default) | 'floor' (Casado Alg. 6, faithful)


def _floor_prefix(y_c, n_classes, per_class):
    """Shortest prefix in which every present class reaches per_class."""
    present = [c for c in range(n_classes) if np.any(y_c == c)]
    need = {c: min(per_class, int(np.sum(y_c == c))) for c in present}
    have = {c: 0 for c in present}
    for i, yi in enumerate(y_c):
        have[int(yi)] = have.get(int(yi), 0) + 1
        if all(have[c] >= need[c] for c in present):
            return i + 1
    return len(y_c)


MEMORY_BOUND = True   # see ids_cda_pipeline.MEMORY_BOUND (audit F5)

from memory_bound import evict_to_cap
import aggregators           # see ids_cda_pipeline


class LongTermMemory:
    """Algorithm 6 memory — see ids_cda_pipeline for the cap/floor note."""

    def __init__(self, rng=None):
        self.concepts = []
        self.rng = rng

    def collect_and_expand(self, X_c, y_c, name):
        if MEMORY_RULE == 'floor':
            n = _floor_prefix(y_c, N_CLASSES_UNSW, PER_CLASS)
            if n > 0:
                self.concepts.append((X_c[:n], y_c[:n], name))
        else:
            Xs, ys = [], []
            for cls in range(N_CLASSES_UNSW):
                idx = np.where(y_c == cls)[0]
                if len(idx) == 0:
                    continue
                n = min(len(idx), PER_CLASS)
                Xs.append(X_c[idx[:n]])
                ys.append(y_c[idx[:n]])
            if Xs:
                self.concepts.append((np.concatenate(Xs), np.concatenate(ys), name))
        if MEMORY_BOUND:
            self.concepts = evict_to_cap(self.concepts, L_CAPACITY,
                                         self.rng or np.random.default_rng(0))

    def get_all(self):
        if not self.concepts:
            return None, None
        return (np.concatenate([c[0] for c in self.concepts]),
                np.concatenate([c[1] for c in self.concepts]))

    def total_samples(self):
        return sum(len(c[1]) for c in self.concepts)

    def n_concepts(self):
        return len(self.concepts)


# ── Shared FL primitives ──────────────────────────────────────────────────────

def _metrics(per_class, per_recall):
    """Flatten per-class F1 and per-class recall into result-row keys."""
    return {**{f'f1_{c}':  per_class.get(c,  float('nan')) for c in CLASS_NAMES_UNSW},
            **{f'rec_{c}': per_recall.get(c, float('nan')) for c in CLASS_NAMES_UNSW}}


def fedavg_aggregate(updates):
    if aggregators.AGG != 'fedavg':
        return aggregators.aggregate(updates, _weighted_mean)
    return _weighted_mean(updates)


def _weighted_mean(updates):
    total = sum(n for _, n in updates)
    agg   = {}
    for state, n in updates:
        w = n / total
        for k, v in state.items():
            agg[k] = agg.get(k, 0) + w * v.float()
    return agg


def local_train_unsw(global_state_gpu, X_gpu, y_gpu, device, client_id=None,
                     E=None, B=None, lr=None,
                     n_classes=N_CLASSES_UNSW):
    lr = SGD_LR if lr is None else lr
    E  = LOCAL_EPOCHS if E is None else E
    B  = BATCH_SIZE if B is None else B
    model = UNSW_DNN(n_classes=n_classes).to(device)
    if LOCAL_NORM and client_id is not None:
        st = dict(global_state_gpu)
        prev = _LOCAL_NORM_STATE.get(client_id)
        if prev:
            st.update({k: v.to(device) for k, v in prev.items()})
        model.load_state_dict(st)
    else:
        model.load_state_dict(global_state_gpu)
    model.train()
    n = len(X_gpu)
    if n < 2:
        return {k: v.cpu() for k, v in model.state_dict().items()}, n
    opt  = _make_opt(model.parameters(), lr)
    crit = nn.CrossEntropyLoss()
    hook = aggregators.local_hook(model, client_id, X_gpu, y_gpu, crit, B)   # FedProx/SCAFFOLD
    for _ in range(E):
        for start in range(0, n, B):
            Xb, yb = X_gpu[start:start+B], y_gpu[start:start+B]
            if len(Xb) < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            if hook:
                loss = loss + hook.penalty()
            loss.backward()
            if hook:
                hook.correct()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
    if hook:
        hook.finish()
    out = {k: v.cpu() for k, v in model.state_dict().items()}
    if LOCAL_NORM and client_id is not None:
        _LOCAL_NORM_STATE[client_id] = {k: out[k].clone()
                                        for k in _norm_keys(model) if k in out}
    return out, n


# ── Experiment A: FedAvg IID ──────────────────────────────────────────────────

def run_iid_fedavg_unsw(concepts, test_set, device,
                         n_classes=N_CLASSES_UNSW, verbose=True):
    if verbose:
        print("\n── Experiment A: FedAvg IID (UNSW-NB15) ──────────────────")
    rng = np.random.default_rng(unsw_loader.SEED)
    all_X = [[] for _ in range(N_CLIENTS)]
    all_y = [[] for _ in range(N_CLIENTS)]
    for concept in concepts:
        for k, (Xk, yk) in enumerate(concept['client_data']):
            if len(Xk) > 0:
                all_X[k].append(Xk)
                all_y[k].append(yk)

    client_gpu = []
    for k in range(N_CLIENTS):
        if all_X[k]:
            Xk = np.concatenate(all_X[k])
            yk = np.concatenate(all_y[k])
            if len(Xk) < 2:
                client_gpu.append(None)
                continue
            perm = rng.permutation(len(Xk))
            client_gpu.append((
                torch.tensor(Xk[perm]).float().to(device),
                torch.tensor(yk[perm]).long().to(device)
            ))
        else:
            client_gpu.append(None)

    global_state_gpu = UNSW_DNN(n_classes=n_classes).to(device).state_dict()
    global_state_cpu = {k: v.cpu() for k, v in global_state_gpu.items()}
    results = []

    for r in range(1, N_ROUNDS+1):
        updates = []
        for ci, c in [(i, c) for i, c in enumerate(client_gpu) if c is not None]:
            s, n = local_train_unsw(global_state_gpu, c[0], c[1], device, client_id=ci)
            updates.append((s, n))
        if updates:
            global_state_cpu = fedavg_aggregate(updates)
            global_state_gpu = {k: v.to(device) for k, v in global_state_cpu.items()}
        overall, pc, mf1, rc = evaluate_unsw(global_state_cpu, test_set[0], test_set[1], device)
        results.append({'round': r, 'overall': overall, 'macro_f1': mf1,
                        **_metrics(pc, rc)})
        if r % 5 == 0 and verbose:
            print(f"  Round {r:2d}/25 | Acc={overall:.4f} | MacroF1={mf1:.4f}")

    if verbose:
        print(f"  [IID] Final: Acc={results[-1]['overall']:.4f} MacroF1={results[-1]['macro_f1']:.4f}")
    return global_state_cpu, results


# ── Experiment A2: FedAvg IID-stream (compute-matched ceiling) ───────────────

def run_iid_stream_fedavg_unsw(concepts, test_set, device,
                               n_classes=N_CLASSES_UNSW, verbose=True):
    """
    IID ceiling with the same gradient-step budget as the non-stationary
    baseline: pool each client's concepts, shuffle away the ordering, stream in
    N_ROUNDS equal slices. See run_iid_stream_fedavg in ids_cda_pipeline.py.
    """
    if verbose:
        print("\n── Experiment A2: IID-stream (compute-matched, UNSW) ────")

    rng = np.random.default_rng(unsw_loader.SEED)
    client_slices = []
    for k in range(N_CLIENTS):
        Xs = [c['client_data'][k][0] for c in concepts if len(c['client_data'][k][0])]
        ys = [c['client_data'][k][1] for c in concepts if len(c['client_data'][k][1])]
        if not Xs:
            client_slices.append([(None, None)] * N_ROUNDS)
            continue
        Xk, yk = np.concatenate(Xs), np.concatenate(ys)
        perm = rng.permutation(len(Xk))
        Xk, yk = Xk[perm], yk[perm]
        client_slices.append([
            (Xk[sp], yk[sp]) if len(sp) > 0 else (None, None)
            for sp in np.array_split(np.arange(len(Xk)), N_ROUNDS)
        ])

    global_state_gpu = UNSW_DNN(n_classes=n_classes).to(device).state_dict()
    global_state_cpu = {k: v.cpu() for k, v in global_state_gpu.items()}
    results = []

    for r in range(1, N_ROUNDS + 1):
        updates = []
        for ci, k_slices in enumerate(client_slices):
            Xsl, ysl = k_slices[r - 1]
            if Xsl is None or len(Xsl) == 0:
                continue
            s, n = local_train_unsw(
                global_state_gpu,
                torch.tensor(Xsl).float().to(device),
                torch.tensor(ysl).long().to(device), device, client_id=ci
            )
            updates.append((s, n))
        if updates:
            global_state_cpu = fedavg_aggregate(updates)
            global_state_gpu = {k: v.to(device) for k, v in global_state_cpu.items()}
        overall, pc, mf1, rc = evaluate_unsw(global_state_cpu, test_set[0],
                                             test_set[1], device)
        results.append({'round': r, 'overall': overall, 'macro_f1': mf1,
                        **_metrics(pc, rc)})
        if r % 5 == 0 and verbose:
            print(f"  Round {r:2d}/25 | Acc={overall:.4f} | MacroF1={mf1:.4f}")

    if verbose:
        print(f"  [IID-stream] Final: Acc={results[-1]['overall']:.4f} "
              f"MacroF1={results[-1]['macro_f1']:.4f}")
    return global_state_cpu, results


# ── Experiment B: FedAvg Non-Stationary ──────────────────────────────────────

def run_nonstationary_fedavg_unsw(concepts, test_set, device,
                                   n_classes=N_CLASSES_UNSW, verbose=True,
                                   on_concept_start=None):
    if verbose:
        print("\n── Experiment B: Non-Stationary FedAvg (UNSW-NB15) ──────")
    global_state_gpu = UNSW_DNN(n_classes=n_classes).to(device).state_dict()
    global_state_cpu = {k: v.cpu() for k, v in global_state_gpu.items()}
    results, total_r = [], 0

    for ci_, concept in enumerate(concepts):
        if on_concept_start and ci_ >= 1:
            on_concept_start(ci_, global_state_cpu)
        slices = []
        for Xk, yk in concept['client_data']:
            if len(Xk) == 0:
                slices.append([(None,None)]*R_PER_CONCEPT)
                continue
            splits = np.array_split(np.arange(len(Xk)), R_PER_CONCEPT)
            slices.append([(Xk[sp], yk[sp]) if len(sp)>0 else (None,None)
                           for sp in splits])

        for ri in range(R_PER_CONCEPT):
            updates = []
            for ci, k_slices in enumerate(slices):
                Xsl, ysl = k_slices[ri]
                if Xsl is None or len(Xsl) == 0:
                    continue
                s, n = local_train_unsw(
                    global_state_gpu,
                    torch.tensor(Xsl).float().to(device),
                    torch.tensor(ysl).long().to(device), device, client_id=ci
                )
                updates.append((s, n))
            if updates:
                global_state_cpu = fedavg_aggregate(updates)
                global_state_gpu = {k: v.to(device) for k, v in global_state_cpu.items()}
            total_r += 1

        overall, pc, mf1, rc = evaluate_unsw(global_state_cpu, test_set[0], test_set[1], device)
        results.append({'event': concept['name'], 'round': total_r,
                        'overall': overall, 'macro_f1': mf1,
                        **_metrics(pc, rc)})
        if verbose:
            print(f"  After {concept['name']:15s} (r={total_r:2d}) | "
                  f"Acc={overall:.4f} | MacroF1={mf1:.4f}")

    if verbose:
        print(f"  [Non-stat] Final: Acc={results[-1]['overall']:.4f} MacroF1={results[-1]['macro_f1']:.4f}")
    return global_state_cpu, results


# ── Experiment C: CDA-FedAvg UNSW ────────────────────────────────────────────

def run_cda_fedavg_unsw(concepts, test_set, device,
                         n_classes=N_CLASSES_UNSW, verbose=True,
                         always_adapt=False, detector_mode='conf', fusion='or',
                         on_concept_start=None):
    if verbose:
        print("\n── Experiment C: CDA-FedAvg UNSW-NB15 ──────────────────")
        print(f"  λ={LAM} Δ={DELTA} Nmax={N_MAX} L={L_CAPACITY} PER_CLASS={PER_CLASS}")

    memories  = [LongTermMemory(rng=np.random.default_rng([unsw_loader.SEED, 20_000 + k]))
                 for k in range(N_CLIENTS)]
    detectors = [HybridDriftDetector(
                     BetaDriftDetector, mode=detector_mode, fusion=fusion,
                     rng=np.random.default_rng([unsw_loader.SEED, k]),
                     rng2=np.random.default_rng([unsw_loader.SEED, 10_000 + k]))
                 for k in range(N_CLIENTS)]
    aes = [None] * N_CLIENTS
    scales = [1.0] * N_CLIENTS

    global_state_gpu = UNSW_DNN(n_classes=n_classes).to(device).state_dict()
    global_state_cpu = {k: v.cpu() for k, v in global_state_gpu.items()}
    results, total_rounds, detected = [], 0, []

    # Phase 0: initial Normal baseline training
    normal_concept = concepts[0]
    if verbose:
        print(f"\n  [P0] {normal_concept['name']}: collect Normal → {R_PER_CONCEPT} rounds")

    for k, (Xk, yk) in enumerate(normal_concept['client_data']):
        if len(Xk) > 0:
            memories[k].collect_and_expand(Xk, yk, normal_concept['name'])

    for r in range(1, R_PER_CONCEPT+1):
        updates = []
        for k in range(N_CLIENTS):
            X_L, y_L = memories[k].get_all()
            if X_L is None or len(X_L) < 2:
                continue
            s, n = local_train_unsw(
                global_state_gpu,
                torch.tensor(X_L).float().to(device),
                torch.tensor(y_L).long().to(device), device, client_id=k
            )
            updates.append((s, n))
        if updates:
            global_state_cpu = fedavg_aggregate(updates)
            global_state_gpu = {k: v.to(device) for k, v in global_state_cpu.items()}
        total_rounds += 1

    overall, pc, mf1, rc = evaluate_unsw(global_state_cpu, test_set[0], test_set[1], device)
    results.append({'event': 'Normal_init', 'round': total_rounds,
                    'overall': overall, 'macro_f1': mf1,
                    'memory_rows': [m.total_samples() for m in memories],
                    **_metrics(pc, rc)})
    if verbose:
        print(f"    [Normal trained] r={total_rounds:2d} | Acc={overall:.4f} | MacroF1={mf1:.4f}")

    # Warm up detector windows with Normal confidence scores
    warm_model = UNSW_DNN(n_classes=n_classes).to(device)
    warm_model.load_state_dict(global_state_gpu)
    warm_model.eval()
    if detector_mode in ('recon', 'hybrid'):
        for k, (Xk, yk) in enumerate(normal_concept['client_data']):
            if len(Xk) < 2:
                continue
            aes[k], scales[k] = train_ae(Xk, device, N_FEATURES_UNSW,
                                         seed=unsw_loader.SEED * 100 + k)

    for k, (Xk, yk) in enumerate(normal_concept['client_data']):
        if len(Xk) == 0:
            continue
        with torch.no_grad():
            confs = warm_model.get_confidence(torch.tensor(Xk).float().to(device))
        rec = (recon_scores(aes[k], Xk, scales[k], device)
               if aes[k] is not None else None)
        for i, conf in enumerate(confs):
            detectors[k].update(float(conf),
                                float(rec[i]) if rec is not None else None)

    # Phases 1-4: stream subsequent concepts, detect drift, adapt
    for ci_, concept in enumerate(concepts[1:], start=1):
        name, desc = concept['name'], concept['description']
        if on_concept_start:
            on_concept_start(ci_, global_state_cpu)
        stream_model = UNSW_DNN(n_classes=n_classes).to(device)
        stream_model.load_state_dict(global_state_gpu)
        stream_model.eval()

        clients_detected = 0
        channels = []
        fired_n, total_n = 0, sum(len(Xk) for Xk, _ in concept['client_data'])
        if always_adapt:
            # oracle detector — isolates the buffer's contribution from the
            # detector's (see run_cda_fedavg_ids)
            clients_detected = N_CLIENTS
        else:
            for k, (Xk, yk) in enumerate(concept['client_data']):
                if len(Xk) == 0:
                    continue
                with torch.no_grad():
                    confs = stream_model.get_confidence(torch.tensor(Xk).float().to(device))
                rec = (recon_scores(aes[k], Xk, scales[k], device)
                       if aes[k] is not None else None)
                for i, conf in enumerate(confs):
                    conf_f = float(conf)
                    rec_f = float(rec[i]) if rec is not None else None
                    detectors[k].update(conf_f, rec_f)
                    if detectors[k].should_check(conf_f, rec_f):
                        if detectors[k].detect():
                            channels.append(detectors[k].fired_by[-1])
                            detectors[k].reset()
                            clients_detected += 1
                            fired_n += len(Xk)
                            break

        fired_share = fired_n / max(total_n, 1)
        triggered = (always_adapt
                     or (clients_detected >= QUORUM if QUORUM_MODE == 'count'
                         else fired_share >= QUORUM_WEIGHT))
        if triggered:
            detected.append({
                'concept': name, 'clients': clients_detected,
                'channels': {c: channels.count(c)
                             for c in ('conf', 'recon', 'both') if channels.count(c)}})
            if verbose:
                print(f"\n  [DRIFT] {name:15s}: {clients_detected}/{N_CLIENTS} clients "
                      f"→ {R_PER_CONCEPT} rounds | {desc}")

            for r in range(1, R_PER_CONCEPT+1):
                updates = []
                for k, (Xk, yk) in enumerate(concept['client_data']):
                    if r == 1 and len(Xk) > 0:
                        memories[k].collect_and_expand(Xk, yk, name)
                    X_L, y_L = memories[k].get_all()
                    if X_L is None or len(X_L) < 2:
                        continue
                    s, n = local_train_unsw(
                        global_state_gpu,
                        torch.tensor(X_L).float().to(device),
                        torch.tensor(y_L).long().to(device), device, client_id=k
                    )
                    updates.append((s, n))
                if updates:
                    global_state_cpu = fedavg_aggregate(updates)
                    global_state_gpu = {k: v.to(device) for k, v in global_state_cpu.items()}
                total_rounds += 1

            overall, pc, mf1, rc = evaluate_unsw(global_state_cpu, test_set[0], test_set[1], device)
            results.append({'event': name, 'round': total_rounds,
                            'overall': overall, 'macro_f1': mf1,
                            'memory_rows': [m.total_samples() for m in memories],
                            **_metrics(pc, rc)})
            detected[-1]['memory_client0'] = memories[0].total_samples()
            if verbose:
                print(f"    [After {name:12s}] r={total_rounds:2d} | "
                      f"Acc={overall:.4f} | MacroF1={mf1:.4f} | "
                      f"memory={memories[0].total_samples()} ({memories[0].n_concepts()} concepts)")
        else:
            if verbose:
                print(f"\n  [NO DRIFT] {name}: 0/{N_CLIENTS} clients fired")

    if verbose:
        print(f"\n  [Summary] Rounds={total_rounds} | Detected={detected}")
    return global_state_cpu, results, detected


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_pc = 500 if QUICK_TEST else MAX_PER_CLASS

    print(f"{'='*60}")
    print("UNSW-NB15 CDA-FedAvg Extension")
    print(f"{'='*60}")
    print(f"Device: {device} | α={ALPHA} | Rounds={N_ROUNDS}")
    print(f"{'QUICK_TEST' if QUICK_TEST else 'Full'}: max_per_class={max_pc}")
    count_parameters_unsw()

    concepts, test_set, scaler, feat_cols = load_unsw_dataset(
        n_clients=N_CLIENTS, alpha=ALPHA,
        max_per_class=max_pc, verbose=True
    )
    os.makedirs(CKPT_DIR, exist_ok=True)

    t0 = time.time()
    _, res_iid = run_iid_fedavg_unsw(concepts, test_set, device)
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    _, res_ns = run_nonstationary_fedavg_unsw(concepts, test_set, device)
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    _, res_cda, detected = run_cda_fedavg_unsw(concepts, test_set, device)
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")

    # Summary
    print(f"\n{'='*65}")
    print("UNSW-NB15 RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Experiment':<28} {'Accuracy':>10} {'MacroF1':>10}")
    print(f"  {'─'*50}")
    for label, res in [('IID FedAvg', res_iid), ('Non-stationary', res_ns), ('CDA-FedAvg', res_cda)]:
        print(f"  {label:<28} {res[-1]['overall']:>10.4f} {res[-1]['macro_f1']:>10.4f}")
    delta_acc = res_cda[-1]['overall'] - res_ns[-1]['overall']
    delta_f1  = res_cda[-1]['macro_f1'] - res_ns[-1]['macro_f1']
    print(f"  {'Recovery vs Non-stat':<28} {delta_acc:>+10.4f} {delta_f1:>+10.4f}")

    print(f"\n── Per-class F1 (CDA-FedAvg) ──")
    print(f"  {'Class':<16} {'F1':>8}")
    minority = {'Backdoors', 'Shellcode', 'Worms', 'Analysis'}
    for cls in CLASS_NAMES_UNSW:
        val  = res_cda[-1].get(f'f1_{cls}', float('nan'))
        flag = '  ← minority' if cls in minority else ''
        print(f"  {cls:<16} {val:>8.4f}{flag}")

    # Save
    save_path = os.path.join(CKPT_DIR, 'unsw_results.json')
    with open(save_path, 'w') as f:
        json.dump({'iid': res_iid, 'nonstat': res_ns, 'cda': res_cda,
                   'detected': detected, 'alpha': ALPHA, 'quick_test': QUICK_TEST},
                  f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and x != x) else x)
    print(f"\n[SAVED] {save_path}")


if __name__ == '__main__':
    main()
