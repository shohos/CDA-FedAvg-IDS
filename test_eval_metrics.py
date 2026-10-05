"""
Self-check for the per-class metric fix.

The invariant that the old code violated: macro-F1 is the mean of per-class F1
by definition, so if the per-class dict really holds F1, its mean must equal the
reported macro_f1. The old dict held per-class recall, so it did not.

Run: python test_eval_metrics.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ids_loader import IDSDNN, evaluate_ids, N_CLASSES, N_FEATURES, CLASS_NAMES
from unsw_loader import (UNSW_DNN, evaluate_unsw, N_CLASSES_UNSW,
                         N_FEATURES_UNSW, CLASS_NAMES_UNSW)


def check(name, evaluate, model_cls, n_classes, n_features, class_names):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device('cpu')

    n = 400
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    # every class present, deliberately imbalanced so F1 != recall
    y = np.concatenate([np.full(n - (n_classes - 1) * 10, 0),
                        np.repeat(np.arange(1, n_classes), 10)]).astype(np.int64)

    state = model_cls(n_classes=n_classes).state_dict()
    overall, per_f1, macro_f1, per_rec = evaluate(state, X, y, device)

    f1_vals = [per_f1[c] for c in class_names]
    assert abs(float(np.mean(f1_vals)) - macro_f1) < 1e-9, \
        f"{name}: mean(per-class F1)={np.mean(f1_vals):.6f} != macro_f1={macro_f1:.6f}"

    # F1 and recall must actually differ somewhere, otherwise the "fix" would
    # still be returning recall under a new name
    diffs = [c for c in class_names if abs(per_f1[c] - per_rec[c]) > 1e-9]
    assert diffs, f"{name}: per-class F1 identical to per-class recall everywhere"
    assert 0.0 <= overall <= 1.0

    c = diffs[0]
    print(f"  {name:8s} OK | macroF1={macro_f1:.4f} == mean(per-class F1) | "
          f"differs from recall on {len(diffs)}/{n_classes} classes "
          f"(e.g. {c}: F1={per_f1[c]:.4f} recall={per_rec[c]:.4f})")


if __name__ == '__main__':
    print("per-class metric self-check")
    check('CICIDS', evaluate_ids, IDSDNN, N_CLASSES, N_FEATURES, CLASS_NAMES)
    check('UNSW', evaluate_unsw, UNSW_DNN, N_CLASSES_UNSW, N_FEATURES_UNSW,
          CLASS_NAMES_UNSW)
    print("all checks passed")
