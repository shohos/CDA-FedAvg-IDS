"""
audit_exposure.py — how much training does each configuration actually get?

HANDOFF §1.1 claims the three configurations "share identical data, backbone,
optimizer, and round count" and that "the only thing that varies is how data is
presented and whether memory exists".

This script counts, without training anything, the number of mini-batch gradient
steps each configuration performs, from the real client shard sizes. If the
counts differ, the comparison is not attributable to drift/memory alone.

Run: python audit_exposure.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ids_loader import load_ids_dataset, N_CLASSES
from ids_cda_pipeline import (N_CLIENTS, N_ROUNDS, R_PER_CONCEPT, LOCAL_EPOCHS,
                              BATCH_SIZE, PER_CLASS)

ALPHA = 0.5


def n_batches(n, B=BATCH_SIZE):
    """Mini-batches local_train_ids actually runs: skips n<2 and final batch of 1."""
    if n < 2:
        return 0
    return sum(1 for s in range(0, n, B) if len(range(s, min(s + B, n))) >= 2)


def main():
    concepts, _, _, _, _ = load_ids_dataset(
        n_clients=N_CLIENTS, alpha=ALPHA, max_per_class=5000,
        test_fraction=0.2, verbose=False)

    shards = [[len(Xk) for Xk, _ in c['client_data']] for c in concepts]
    pooled = [sum(shards[ci][k] for ci in range(len(concepts)))
              for k in range(N_CLIENTS)]
    total = sum(pooled)

    # ── A: IID — every round, every client trains on its whole pooled shard
    iid = N_ROUNDS * LOCAL_EPOCHS * sum(n_batches(n) for n in pooled)

    # ── B: non-stationary — each round is one 1/R slice of one concept
    ns = 0
    for ci, _ in enumerate(concepts):
        for k in range(N_CLIENTS):
            for sl in np.array_split(np.arange(shards[ci][k]), R_PER_CONCEPT):
                ns += LOCAL_EPOCHS * n_batches(len(sl))

    # ── C: CDA — every round trains on the whole accumulated buffer.
    # Buffer after a concept = min(PER_CLASS, count) summed over classes seen.
    cda, buffers = 0, []
    for k in range(N_CLIENTS):
        buf = 0
        for ci, c in enumerate(concepts):
            yk = c['client_data'][k][1]
            if len(yk):
                buf += sum(min(int((yk == cls).sum()), PER_CLASS)
                           for cls in range(N_CLASSES) if (yk == cls).any())
            # R rounds of training on the buffer as it stands after this concept
            cda += R_PER_CONCEPT * LOCAL_EPOCHS * n_batches(buf)
        buffers.append(buf)

    print(f"CICIDS2017, α={ALPHA}, {N_CLIENTS} clients, {N_ROUNDS} rounds, "
          f"E={LOCAL_EPOCHS}, B={BATCH_SIZE}")
    print(f"training rows: {total:,} | pooled per client: "
          f"{min(pooled):,}–{max(pooled):,}")
    print(f"final CDA buffer per client: {min(buffers)}–{max(buffers)} "
          f"(L_CAPACITY=1400)\n")

    print(f"{'config':<28}{'grad steps':>13}{'vs non-stat':>13}")
    print("-" * 54)
    for label, v in [("A  IID FedAvg", iid),
                     ("B  Non-stationary FedAvg", ns),
                     ("C  CDA-FedAvg", cda)]:
        print(f"{label:<28}{v:>13,}{v/ns:>12.1f}x")

    print(f"\nunique rows each config trains on (all rounds):")
    print(f"  IID           {total:>9,}  seen {N_ROUNDS}x")
    print(f"  Non-stat      {total:>9,}  seen  1x")
    print(f"  CDA           {sum(buffers):>9,}  ({sum(buffers)/total:.1%} of the data, "
          f"re-seen every round)")


if __name__ == '__main__':
    main()
