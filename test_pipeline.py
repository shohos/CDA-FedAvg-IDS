"""
Self-checks for the fixed components. Run: python test_pipeline.py

Covers the three things changed on 2026-08-13:
  1. BetaDriftDetector now implements Casado Alg. 5 (not the pooled-vs-split GLR)
  2. each detector owns its RNG (no coupling through the global NumPy stream)
  3. LongTermMemory supports the paper's 'floor' rule alongside our 'cap' rule
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ids_cda_pipeline as P
import unsw_cda_pipeline as U
from detector_faithfulness import detect_paper


def _win(rng, a1, b1, a2=None, b2=None, n=1000):
    if a2 is None:
        return rng.beta(a1, b1, n)
    return np.concatenate([rng.beta(a1, b1, n // 2), rng.beta(a2, b2, n // 2)])


def test_matches_algorithm5():
    """detect() must agree with a literal transcription of Alg. 5."""
    rng = np.random.default_rng(1)
    cases = [(8, 2, None, None), (2, 2, None, None), (8, 2, 2, 8), (8, 2, 7, 2.6)]
    for a1, b1, a2, b2 in cases:
        w = _win(rng, a1, b1, a2, b2)
        d = P.BetaDriftDetector()
        for q in w:
            d.update(q)
        ours, s_ours = d.detect(return_score=True)
        ref, s_ref = detect_paper(np.clip(w, 1e-6, 1 - 1e-6))
        assert ours == ref, f"fire mismatch on Beta({a1},{b1})->({a2},{b2})"
        assert abs(s_ours - s_ref) < 1e-6 * max(1.0, abs(s_ref)), \
            f"score mismatch {s_ours} vs {s_ref}"
    print("  alg5 transcription      OK  detect() == Algorithm 5, 4 window types")


def test_detects_real_drift_only():
    rng = np.random.default_rng(2)
    d = P.BetaDriftDetector()
    for q in _win(rng, 8, 2):                      # no drift, high confidence
        d.update(q)
    assert not d.detect(), "fired on a no-drift high-confidence window"

    d = P.BetaDriftDetector()
    for q in _win(rng, 8, 2, 2, 8):                # confidence collapses
        d.update(q)
    assert d.detect(), "missed an unambiguous drift"
    print("  drift sensitivity       OK  silent on no-drift, fires on real drift")


def test_per_client_rng_isolated():
    """Same seed reproduces; different seeds differ; global RNG is irrelevant."""
    def stream(seed, k):
        d = P.BetaDriftDetector(rng=np.random.default_rng([seed, k]))
        return [d.should_check(0.7) for _ in range(200)]

    assert stream(42, 0) == stream(42, 0), "same seed did not reproduce"
    assert stream(42, 0) != stream(42, 1), "two clients share a stream"
    assert stream(42, 0) != stream(43, 0), "seed has no effect"

    # perturbing the global NumPy stream must not change anything
    a = stream(42, 3)
    np.random.seed(999); [np.random.random() for _ in range(50)]
    assert stream(42, 3) == a, "detector still depends on the global RNG"
    print("  rng isolation           OK  per-client, reproducible, global-independent")


def test_memory_cap_vs_floor():
    rng = np.random.default_rng(3)
    n_cls = P.N_CLASSES
    # imbalanced concept: class 0 dominant, class 6 rare
    counts = [4000, 800, 600, 400, 300, 200, 40]
    y = np.concatenate([np.full(c, i) for i, c in enumerate(counts)]).astype(np.int64)
    perm = rng.permutation(len(y))
    y = y[perm]
    X = rng.normal(size=(len(y), P.N_FEATURES)).astype(np.float32)

    try:
        P.MEMORY_RULE = 'cap'
        m = P.LongTermMemory(); m.collect_and_expand(X, y, 'c')
        _, y_cap = m.get_all()

        P.MEMORY_RULE = 'floor'
        m = P.LongTermMemory(); m.collect_and_expand(X, y, 'c')
        _, y_floor = m.get_all()
    finally:
        P.MEMORY_RULE = 'cap'

    per = P.PER_CLASS
    for c in range(n_cls):
        assert (y_cap == c).sum() <= per, f"cap exceeded for class {c}"
    # floor: every class reaches min(PER_CLASS, supply)
    for c in range(n_cls):
        assert (y_floor == c).sum() >= min(per, counts[c]), \
            f"floor not met for class {c}"
    assert len(y_floor) > len(y_cap), "floor should collect more than cap here"
    print(f"  memory cap vs floor     OK  cap={len(y_cap)} floor={len(y_floor)} "
          f"(L={P.L_CAPACITY}, per-class={per})")


def test_unsw_detector_agrees():
    """The two pipelines must not drift apart."""
    rng = np.random.default_rng(4)
    w = _win(rng, 8, 2, 3, 5)
    a = P.BetaDriftDetector(); b = U.BetaDriftDetector()
    for q in w:
        a.update(q); b.update(q)
    fa, sa = a.detect(return_score=True)
    fb, sb = b.detect(return_score=True)
    # different PER_CLASS/N_CLASSES but identical detector maths
    assert fa == fb and abs(sa - sb) < 1e-9, "ids and unsw detectors disagree"
    print("  cross-pipeline parity   OK  ids and unsw detectors identical")




def test_norm_swap_is_parameter_matched():
    """LayerNorm must be a drop-in: same trainable params, no running stats."""
    import torch
    import ids_loader, unsw_loader
    for cls, nf, nc, mod in [(ids_loader.IDSDNN, ids_loader.N_FEATURES,
                              ids_loader.N_CLASSES, ids_loader),
                             (unsw_loader.UNSW_DNN, unsw_loader.N_FEATURES_UNSW,
                              unsw_loader.N_CLASSES_UNSW, unsw_loader)]:
        # the default tracks the best measured configuration, not the baseline;
        # what must hold is that both settings are constructible and matched
        assert mod.NORM in ('batch', 'layer')
        bn = cls(n_classes=nc, norm='batch')
        ln = cls(n_classes=nc, norm='layer')
        nb = sum(p.numel() for p in bn.parameters() if p.requires_grad)
        nl = sum(p.numel() for p in ln.parameters() if p.requires_grad)
        assert nb == nl, f"param mismatch {nb} vs {nl}"
        # BatchNorm carries running stats that FedAvg averages; LayerNorm must not
        assert any('running_mean' in k for k in bn.state_dict()), "no BN buffers?"
        assert not any('running_mean' in k for k in ln.state_dict()), \
            "LayerNorm model still carries running statistics"
        ln.eval()
        assert ln(torch.randn(1, nf)).shape == (1, nc)
    print("  norm swap               OK  parameter-matched, no running stats "
          "under LayerNorm")

def test_recon_channel_separates_novelty():
    """The property the whole channel depends on: novel data scores lower."""
    import torch
    from drift_channels import train_ae, recon_scores
    dev = torch.device('cpu')
    rng = np.random.default_rng(7)
    nf = 20
    train = rng.normal(0, 1, size=(2000, nf)).astype(np.float32)
    ae, scale = train_ae(train, dev, nf, epochs=30, seed=1)

    same = recon_scores(ae, rng.normal(0, 1, (800, nf)).astype(np.float32), scale, dev)
    novel = recon_scores(ae, rng.normal(4, 1, (800, nf)).astype(np.float32), scale, dev)

    assert np.all((same > 0) & (same <= 1.0)), "scores must lie in (0,1]"
    assert same.mean() > novel.mean() * 3, (
        f"insufficient separation: in-dist {same.mean():.4f} vs novel {novel.mean():.4f}")
    # and the gate the detector actually applies must open on that shift
    assert novel.mean() <= 0.95 * same.mean(), "gate would stay shut on novel data"
    print(f"  recon separates novelty OK  in-dist {same.mean():.3f} vs "
          f"novel {novel.mean():.3g} (ratio {novel.mean()/same.mean():.3g})")


def test_train_ae_is_reproducible_from_its_seed():
    """Same seed, same data -> same model, whatever the ambient torch RNG is.

    Regression test for a real defect (found in an external audit, 2026-09-17):
    `seed` controlled only the shuffling generator, while module construction and
    Xavier init drew from the global RNG, so two calls with seed=7 differed by up
    to 0.77 in initial parameters. Zero epochs isolates initialisation.
    """
    import torch
    from drift_channels import train_ae
    X = np.random.default_rng(3).normal(size=(40, 6)).astype(np.float32)
    torch.manual_seed(1)
    a, _ = train_ae(X, torch.device('cpu'), 6, seed=7, epochs=0)
    torch.manual_seed(999)
    b, _ = train_ae(X, torch.device('cpu'), 6, seed=7, epochs=0)
    gap = max(float((p - q).abs().max()) for p, q in zip(a.parameters(), b.parameters()))
    assert gap == 0.0, f"same seed gave different weights (max diff {gap:.5f})"
    c, _ = train_ae(X, torch.device('cpu'), 6, seed=8, epochs=0)
    assert max(float((p - q).abs().max())
               for p, q in zip(a.parameters(), c.parameters())) > 0, \
        "different seeds must give different weights"
    print("  train_ae reproducible from seed OK  (and seeds differ)")


def test_conf_mode_is_unchanged():
    """detector_mode='conf' must delegate exactly to the bare detector."""
    from drift_channels import HybridDriftDetector
    rng = np.random.default_rng(8)
    w = _win(rng, 8, 2, 2, 8)

    bare = P.BetaDriftDetector(rng=np.random.default_rng([42, 0]))
    hyb = HybridDriftDetector(P.BetaDriftDetector, mode='conf',
                              rng=np.random.default_rng([42, 0]),
                              rng2=np.random.default_rng([42, 1]))
    for q in w:
        bare.update(q); hyb.update(q, None)
    assert bare.detect() == hyb.detect(), "conf mode diverges from bare detector"
    # trigger stream must match too, or seeded runs would not reproduce
    a = [bare.should_check(0.8) for _ in range(100)]
    b = [hyb.should_check(0.8, None) for _ in range(100)]
    assert a == b, "conf-mode trigger stream diverges"
    print("  conf mode unchanged     OK  delegates exactly to BetaDriftDetector")


def test_fusion_rules():
    """OR fires if either channel fires; AND needs both."""
    from drift_channels import HybridDriftDetector
    rng = np.random.default_rng(9)
    drift = _win(rng, 8, 2, 2, 8)      # fires
    flat = _win(rng, 8, 2)             # does not fire

    for fusion, expect in (('or', True), ('and', False)):
        h = HybridDriftDetector(P.BetaDriftDetector, mode='hybrid', fusion=fusion,
                                rng=np.random.default_rng(0),
                                rng2=np.random.default_rng(1))
        for c, r in zip(drift, flat):   # conf drifts, recon does not
            h.update(float(c), float(r))
        assert h.detect() is expect, f"{fusion} fusion gave the wrong answer"
    print("  fusion rules            OK  OR fires on one channel, AND needs both")


if __name__ == '__main__':
    print("pipeline self-checks")
    test_matches_algorithm5()
    test_detects_real_drift_only()
    test_per_client_rng_isolated()
    test_memory_cap_vs_floor()
    test_unsw_detector_agrees()
    test_norm_swap_is_parameter_matched()
    test_recon_channel_separates_novelty()
    test_train_ae_is_reproducible_from_its_seed()
    test_conf_mode_is_unchanged()
    test_fusion_rules()
    print("all checks passed")
