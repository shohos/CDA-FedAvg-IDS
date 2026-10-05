"""
drift_channels.py — a drift signal that does not read the classifier's confidence.

Why
---
`confidence_diagnostic.py` measured that mean confidence on *misclassified*
new-concept samples equals or exceeds the overall mean: the model is often more
confident when it is wrong. Detection needs a 5% fall in mean confidence
(`mean(recent) <= 0.95 * mean(older)`), so a concept the model confidently
mislabels is invisible to the confidence channel.

The LayerNorm run made this decisive: with a better-behaved model the confidence
channel fired 0 of 4 drifts in all 10 runs, because a model that is stable under
distribution shift is also a model whose confidence does not move. The
confidence-based detector depends on the classifier being badly behaved.

A reconstruction channel does not depend on the classifier at all. A small
autoencoder trained on phase-0 traffic reports how unlike phase-0 a flow looks,
whether or not the classifier has an opinion about it.

Design
------
Following the plan in the gap analysis, with the mapping chosen so the existing
Beta machinery applies unchanged:

  1. per client, train a small AE on that client's phase-0 data only
  2. per streamed sample, e = ||x - AE(x)||^2
  3. s = exp(-e / scale), scale = median phase-0 error. Phase-0-like traffic
     scores near exp(-1) ~ 0.37; novel traffic scores near 0. Low score = novel,
     which mirrors the semantics of max-softmax confidence, so the same
     downward-shift gate and the same Beta-CUSUM apply with no changes.
  4. run a second detector on the s stream and fuse.

The AE is local, never averaged. Drift detection in Algorithm 4 is a per-client
decision, and each client's notion of "normal" is its own; federating the AE
would add a communication round to answer a local question.

`scale` is the calibration knob. The median is a starting point, not a law —
`SCALE_FACTOR` multiplies it, and on real deployments it is the first thing to
tune.
"""
import numpy as np
import torch
import torch.nn as nn

CALIB = 'heldout'     # 'heldout' | 'train' — module default for train_ae
ISOLATE_CUDA_RNG = True   # False reproduces the original runs. torch.manual_seed
                          # reseeds CUDA too, and fork_rng(devices=[]) restored only
                          # the CPU stream, so fitting an AE changed the classifier's
                          # later GPU dropout draws (audit F3, 2026-09-26).
SCALE_FACTOR = 1.0    # ponytail: median-based scale, tune if the channel is
                      # too eager (raise) or too deaf (lower)


class ReconAE(nn.Module):
    """n_features -> 32 -> 8 -> 32 -> n_features. ~5.7k params against the
    classifier's 62.7k, so it cannot dominate the system it advises."""

    def __init__(self, n_features, hidden=32, latent=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, n_features))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(X, device, n_features, epochs=20, batch=100, lr=1e-3, seed=0,
             calib=None, holdout=0.2):
    """
    Fit an AE on one client's phase-0 data. Returns (model, scale).

    calib='heldout' (default) fits on part of phase-0 and calibrates `scale` on
    the rest. calib='train' calibrates on the fitting data itself.

    Why this matters: an AE always reconstructs its own training data better than
    unseen data. With calib='train', `scale` is set from that optimistic error,
    so *any* held-out traffic — including same-distribution traffic — scores
    lower and the detector's 5% gate opens on ordinary generalisation error
    rather than on drift. null_control.py measured that directly: with
    calib='train' the reconstruction channel raised a false alarm on 3.7 of 4
    no-drift concepts, and alarmed sooner on no-drift data (108 samples) than on
    real drift (220). Calibrating on held-out data removes that bias.
    """
    calib = calib or CALIB
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    # `seed` used to control only the permutation generator `g`, while module
    # construction and Xavier init drew from the GLOBAL torch RNG: the same seed
    # gave different models depending on ambient RNG state (measured maximum
    # initial parameter difference 0.77). fork_rng seeds initialisation locally
    # and restores the caller's stream, so a run is reproducible from its seed.
    cuda = (list(range(torch.cuda.device_count()))
            if ISOLATE_CUDA_RNG and torch.cuda.is_available() else [])
    with torch.random.fork_rng(devices=cuda):
        torch.manual_seed(int(seed))
        ae = ReconAE(n_features).to(device)
        for p in ae.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)
    n = len(X)
    if n < 2:
        return ae, 1.0

    Xg = torch.as_tensor(X).float().to(device)
    if calib == 'heldout' and n >= 10:
        perm0 = torch.randperm(n, generator=g)
        n_cal = max(2, int(n * holdout))
        X_cal = Xg[perm0[:n_cal].to(device)]
        X_fit = Xg[perm0[n_cal:].to(device)]
    else:
        X_fit = X_cal = Xg

    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    lossf = nn.MSELoss()
    m = len(X_fit)
    for _ in range(epochs):
        perm = torch.randperm(m, generator=g).to(device)
        for s in range(0, m, batch):
            xb = X_fit[perm[s:s + batch]]
            if len(xb) < 2:
                continue
            opt.zero_grad()
            loss = lossf(ae(xb), xb)
            loss.backward()
            opt.step()
    ae.eval()
    with torch.no_grad():
        err = ((ae(X_cal) - X_cal) ** 2).mean(dim=1)
    scale = float(err.median().item()) * SCALE_FACTOR
    return ae, max(scale, 1e-8)


def recon_scores(ae, X, scale, device):
    """Bounded novelty score in (0, 1]; low = novel. Mirrors confidence."""
    if ae is None or len(X) == 0:
        return np.zeros(len(X), dtype=np.float32)
    ae.eval()
    with torch.no_grad():
        Xg = torch.as_tensor(X).float().to(device)
        err = ((ae(Xg) - Xg) ** 2).mean(dim=1)
        s = torch.exp(-err / scale)
    return s.cpu().numpy()


class HybridDriftDetector:
    """
    Two independent Beta-CUSUM channels with configurable fusion.

    mode:
      'conf'   confidence only — the published behaviour
      'recon'  reconstruction only
      'hybrid' both, fused by `fusion`

    fusion:
      'or'   fire if either channel fires (sensitive; the plan's default)
      'and'  fire only if both agree (conservative; the documented variant)

    Records which channel fired each time, so per-channel attribution can be
    reported rather than guessed at.
    """

    def __init__(self, detector_cls, mode='hybrid', fusion='or', rng=None,
                 rng2=None, **kw):
        self.mode = mode
        self.fusion = fusion
        self.conf_det = detector_cls(rng=rng, **kw)
        self.recon_det = detector_cls(rng=rng2, **kw)
        self.fired_by = []

    def update(self, conf, recon=None):
        if self.mode in ('conf', 'hybrid'):
            self.conf_det.update(conf)
        if self.mode in ('recon', 'hybrid') and recon is not None:
            self.recon_det.update(recon)

    def should_check(self, conf, recon=None):
        """Either channel may request a detection pass, using its own score."""
        a = (self.mode in ('conf', 'hybrid')
             and self.conf_det.should_check(conf))
        b = (self.mode in ('recon', 'hybrid') and recon is not None
             and self.recon_det.should_check(recon))
        return a or b

    def detect(self):
        c = self.conf_det.detect() if self.mode in ('conf', 'hybrid') else False
        r = self.recon_det.detect() if self.mode in ('recon', 'hybrid') else False
        fired = (c and r) if (self.mode == 'hybrid' and self.fusion == 'and') \
            else (c or r)
        if fired:
            self.fired_by.append('both' if (c and r)
                                 else ('conf' if c else 'recon'))
        return fired

    def reset(self):
        self.conf_det.reset()
        self.recon_det.reset()

    def channel_counts(self):
        return {k: self.fired_by.count(k) for k in ('conf', 'recon', 'both')}
