"""
detector_faithfulness.py — does our BetaDriftDetector implement Casado's Algorithm 5?

Casado et al. (Multimedia Tools and Applications 81(3), Alg. 5, line 12) accumulate,
over the RECENT sub-window only, the log-ratio of the two fitted alternatives:

    s_k = Σ_{i=k+1..N} log[ f(q_i | α_a, β_a) / f(q_i | α_b, β_b) ]

Our BetaDriftDetector.detect() instead computes a pooled-vs-split generalised
likelihood ratio over the WHOLE window:

    s_k = ll(Q_a | θ_a) + ll(Q_b | θ_b) − ll(Q | θ_pooled)

The second is a nested-model comparison: the split model has 4 free parameters
against the pooled model's 2, so it can never fit worse and s_k ≥ 0 always.
Asymptotically 2·s_k ~ χ²(2) under the null, giving P(s_k > 3.0) = e^{-3} ≈ 5%
per call — and detect() is called hundreds of times per concept.

Casado's statistic compares two fixed alternatives on the same data, so under no
drift its expectation is negative and clearing +3.0 requires real separation.

This script measures both on identical windows. No training, no GPU.
"""
import numpy as np
from scipy.stats import beta as beta_dist

LAM, DELTA, N_MAX = 0.05, 100, 1000
TH = -np.log(LAM)


def fit_beta(x):
    """Method of moments — identical to both implementations."""
    x = np.clip(x, 1e-6, 1 - 1e-6)
    mu, var = np.mean(x), np.var(x)
    max_v = mu * (1 - mu) - 1e-6
    var = np.clip(var, 1e-8, max_v if max_v > 1e-8 else 1e-6)
    f = mu * (1 - mu) / var - 1
    return max(mu * f, 1e-6), max((1 - mu) * f, 1e-6)


def _ll(x, a, b):
    return float(np.sum(beta_dist.logpdf(np.clip(x, 1e-6, 1 - 1e-6), a, b)))


def detect_ours(w, lam=LAM, delta=DELTA):
    """Exactly what ids_cda_pipeline.BetaDriftDetector.detect() does."""
    w = np.asarray(w)
    if len(w) < 2 * delta:
        return False, 0.0
    a0, b0 = fit_beta(w)
    ll0 = _ll(w, a0, b0)
    best = -np.inf
    for k in range(delta, len(w) - delta):
        wa, wb = w[k:], w[:k]
        if np.mean(wa) > (1 - lam) * np.mean(wb):
            continue
        aa, ba = fit_beta(wa)
        ab, bb = fit_beta(wb)
        best = max(best, _ll(wa, aa, ba) + _ll(wb, ab, bb) - ll0)
    s = best if best > -np.inf else 0.0
    return s > TH, s


def detect_paper(w, lam=LAM, delta=DELTA):
    """Casado et al. Algorithm 5, as written."""
    w = np.asarray(w)
    N = len(w)
    if N < 2 * delta:
        return False, 0.0
    sf = 0.0
    for k in range(delta, N - delta + 1):
        qb, qa = w[:k], w[k:]
        if np.mean(qa) > (1 - lam) * np.mean(qb):
            continue
        ab, bb = fit_beta(qb)
        aa, ba = fit_beta(qa)
        qa_c = np.clip(qa, 1e-6, 1 - 1e-6)
        sk = float(np.sum(beta_dist.logpdf(qa_c, aa, ba)
                          - beta_dist.logpdf(qa_c, ab, bb)))
        sf = max(sf, sk)
    return sf > TH, sf


def gate_ever_opens(w, lam=LAM, delta=DELTA):
    """Did any split pass the 'recent mean dropped 5%' gate at all?"""
    w = np.asarray(w)
    for k in range(delta, len(w) - delta):
        if np.mean(w[k:]) <= (1 - lam) * np.mean(w[:k]):
            return True
    return False


def trial(name, sampler, n_trials, rng):
    ours = paper = gated = 0
    s_ours, s_paper = [], []
    for _ in range(n_trials):
        w = sampler(rng)
        g = gate_ever_opens(w)
        gated += g
        do, so = detect_ours(w)
        dp, sp = detect_paper(w)
        ours += do
        paper += dp
        s_ours.append(so)
        s_paper.append(sp)
    print(f"  {name:<34} gate opens {gated/n_trials:>5.0%} | "
          f"ours fires {ours/n_trials:>5.0%} (median s={np.median(s_ours):>9.1f}) | "
          f"paper fires {paper/n_trials:>5.0%} (median s={np.median(s_paper):>7.1f})")
    return ours / n_trials, paper / n_trials


def main():
    rng = np.random.default_rng(0)
    n = 60
    print(f"Threshold Th = -log({LAM}) = {TH:.3f}, N={N_MAX}, Δ={DELTA}, "
          f"{n} trials each\n")

    print("NO DRIFT — one distribution throughout (any firing is a false positive)")
    trial("high conf, Beta(8,2)", lambda r: r.beta(8, 2, N_MAX), n, rng)
    trial("very high conf, Beta(50,2)", lambda r: r.beta(50, 2, N_MAX), n, rng)
    trial("saturated, Beta(200,1)", lambda r: r.beta(200, 1, N_MAX), n, rng)
    trial("mid conf, Beta(2,2)", lambda r: r.beta(2, 2, N_MAX), n, rng)

    print("\nREAL DRIFT — confidence collapses halfway (both should fire)")
    trial("Beta(8,2) -> Beta(2,8)",
          lambda r: np.concatenate([r.beta(8, 2, 500), r.beta(2, 8, 500)]), n, rng)
    trial("Beta(8,2) -> Beta(4,4)",
          lambda r: np.concatenate([r.beta(8, 2, 500), r.beta(4, 4, 500)]), n, rng)

    print("\nMILD DRIFT — small mean drop, the ambiguous regime")
    trial("Beta(8,2) -> Beta(7,2.6)",
          lambda r: np.concatenate([r.beta(8, 2, 500), r.beta(7, 2.6, 500)]), n, rng)

    print("\nInterpretation: a statistic that fires on no-drift windows is not")
    print("detecting drift, it is detecting that two halves of any sample differ.")


if __name__ == '__main__':
    main()
