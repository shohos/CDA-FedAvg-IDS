"""
test_repairs.py -- checks for the 2026-09-26 audit repairs (AUDIT_REPAIR_PLAN.md).

    python test_repairs.py            fast checks, no datasets needed
    python test_repairs.py --data     also the dataset checks (causal preprocessing,
                                      and C with every boundary triggered == B2)
"""
import sys
import numpy as np
import torch

import drift_channels as DC
import fl_iids as FI
import ids_cda_pipeline as PI
from memory_bound import evict_to_cap


def rng_state():
    cuda = [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    return torch.get_rng_state().clone(), cuda


def test_train_ae_leaves_rng_alone():
    X = np.random.default_rng(0).normal(size=(50, 6)).astype(np.float32)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DC.ISOLATE_CUDA_RNG = True
    torch.manual_seed(123)
    cpu0, cuda0 = rng_state()
    DC.train_ae(X, dev, 6, seed=7, epochs=0)
    cpu1, cuda1 = rng_state()
    assert torch.equal(cpu0, cpu1)
    assert all(torch.equal(a, b) for a, b in zip(cuda0, cuda1)), 'CUDA RNG moved'
    if torch.cuda.is_available():            # the original behaviour, documented
        DC.ISOLATE_CUDA_RNG = False
        torch.manual_seed(123)
        _, c0 = rng_state()
        DC.train_ae(X, dev, 6, seed=7, epochs=0)
        _, c1 = rng_state()
        assert not all(torch.equal(a, b) for a, b in zip(c0, c1)), 'legacy leak not reproduced'
        DC.ISOLATE_CUDA_RNG = True


def test_rehearsal_memory_is_bounded():
    """The audit's counterexample: twenty benign concepts, L = 1400."""
    sizes = {}
    for bound in (True, False):
        PI.MEMORY_BOUND = bound
        m = PI.LongTermMemory(rng=np.random.default_rng(0))
        for t in range(20):
            m.collect_and_expand(np.full((300, 4), t, np.float32), np.zeros(300, np.int64), f'c{t}')
            if bound:
                assert m.total_samples() <= PI.L_CAPACITY
        sizes[bound] = m.total_samples()
    PI.MEMORY_BOUND = True
    assert sizes[True] == PI.L_CAPACITY and sizes[False] == 20 * PI.PER_CLASS, sizes


def test_fliids_stores_only_its_quota():
    mem = FI.DynamicExampleMemory()
    for t in range(3):
        mem.groups.append({'task': t, 'X': np.zeros((24, 3)), 'y': np.full(24, t),
                           'size': 24, 'kept': 24})
    mem.rebalance(12, discard=False)
    assert mem.total() == 12 and mem.stored() == 72          # the audit's diagnostic
    mem.rebalance(12, discard=True)
    assert mem.total() == 12 and mem.stored() == 12


def test_fliids_loss_switches():
    torch.manual_seed(0)
    logits, teacher = torch.randn(8, 5), torch.randn(8, 5)
    y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 3])
    old = torch.tensor([True, True, False, False, False])
    # legacy settings give the original formula exactly
    FI.TEACHER, FI.LS_TARGET = 'full', 'normalised'
    tgt = torch.nn.functional.one_hot(y, 5).float()
    tgt[:, old] = torch.softmax(teacher, 1)[:, old]
    tgt = tgt / tgt.sum(1, keepdim=True)
    ref = torch.nn.functional.kl_div(torch.log_softmax(logits, 1), tgt, reduction='batchmean')
    got = FI.fliids_loss(logits, y, old, teacher, 'ce', 0.0, 1.0)
    assert torch.allclose(got, ref), (got, ref)
    # repaired teacher: a distribution over the old classes only
    FI.TEACHER, FI.LS_TARGET = 'old_only', 'raw'
    tp = torch.softmax(teacher.masked_fill(~old, float('-inf')), 1)
    assert torch.allclose(tp[:, old].sum(1), torch.ones(8)) and (tp[:, ~old] == 0).all()
    v = FI.fliids_loss(logits, y, old, teacher, 'gb', 0.5, 0.5)
    assert torch.isfinite(v)
    # the printed direction KL(student || target) is infinite on these targets
    t_raw = torch.nn.functional.one_hot(y, 5).float()
    t_raw[:, old] = tp[:, old]
    ps = torch.softmax(logits, 1)
    assert torch.isinf((ps * (ps.log() - t_raw.log())).sum(1)).all()
    FI.TEACHER, FI.LS_TARGET = 'old_only', 'normalised'


def data_checks():
    import ids_loader, unsw_loader
    # 1. causal preprocessing: the scaler is the first concept's training rows' own
    ids_loader.PREPROC = 'causal'
    ids_loader.SEED = 41
    concepts, test, ctests, scaler, _ = ids_loader.load_ids_dataset(
        n_clients=10, alpha=0.5, max_per_class=300, verbose=False)
    first = np.concatenate([X for X, _ in concepts[0]['client_data'] if len(X)])
    assert np.allclose(first.mean(0), 0, atol=1e-4), 'first concept not centred by its scaler'
    assert all(np.isfinite(X).all() for c in concepts for X, _ in c['client_data'])
    assert all('test' in c for c in concepts)
    unsw_loader.PREPROC = 'causal'
    unsw_loader.SEED = 41
    uc, ut, _, _ = unsw_loader.load_unsw_dataset(n_clients=10, alpha=0.5, max_per_class=300,
                                                 verbose=False)
    u0 = np.concatenate([X for X, _ in uc[0]['client_data'] if len(X)])
    assert np.allclose(u0.mean(0), 0, atol=1e-4)
    # 2. C with every boundary triggered is B2, bit for bit, once the RNG is isolated
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finals = {}
    for iso in (True, False):
        DC.ISOLATE_CUDA_RNG = iso
        for mode in ('conf', 'hybrid'):
            torch.manual_seed(41)
            np.random.seed(41)
            st, hist, _ = PI.run_cda_fedavg_ids(concepts, test, dev, verbose=False,
                                                always_adapt=True, detector_mode=mode)
            finals[(iso, mode)] = (hist[-1]['macro_f1'], st)
    b2, c = finals[(True, 'conf')], finals[(True, 'hybrid')]
    assert b2[0] == c[0] and all(torch.equal(b2[1][k], c[1][k]) for k in b2[1]), \
        'C (all triggered) differs from B2'
    print(f'  all-triggered C == B2: {c[0]:.6f}; legacy RNG: B2 {finals[(False, "conf")][0]:.6f} '
          f'vs C {finals[(False, "hybrid")][0]:.6f}')
    DC.ISOLATE_CUDA_RNG = True


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for t in tests:
        t()
        print('ok', t.__name__)
    if '--data' in sys.argv:
        data_checks()
        print('ok data checks')
