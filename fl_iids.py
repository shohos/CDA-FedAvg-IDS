"""
fl_iids.py -- FL-IIDS (Jin et al., Future Generation Computer Systems 151,
2024, pp. 57-70) as an extra arm on the SAME streams as the five-arm design.
Written for objective O5: a matched comparison against FL-IIDS under non-IID
partitioning.

What is implemented, from Section 3 of the paper:

  * Dynamic example memory (Eq. 3). Every (task, class) group a client has
    finished gets memory in proportion to how many samples of it the client
    saw, so the memory follows the real class imbalance instead of splitting
    equally. Exemplars inside a group are chosen by iCaRL herding (Eq. 2) on
    the model's penultimate features.
  * Class gradient balance loss (Eqs. 4-7). Each sample's cross-entropy is
    weighted by |p_y - 1| divided by the mean of that quantity over the
    batch's new-class (or old-class) samples.
  * Sample label smoothing loss (Eq. 8). The one-hot label's old-class entries
    are replaced by the old model's probabilities, and the student is pulled to
    that soft label with KL divergence.
  * Local loss L = 0.5 * L_GB + 0.5 * L_LS (Eq. 9, gamma values of Sec. 4.2).

What is NOT implemented, and why:

  * The relay client that rebuilds representative samples from gradients to
    pick the "best old model" (Sec. 3.3). The paper's own ablation (Table 6)
    puts its effect at 68.764% vs 68.748% final accuracy on UNSW-NB15 and
    99.620% vs 99.505% on CICIDS2018. We use the paper's "without Cr" variant:
    each client keeps, as its old model, the global model from the previous
    task that scores best on its own exemplar memory.

Protocol alignment (identical for every arm, so only the method differs):
full participation of all 10 clients, the same Dirichlet partition, model,
optimiser, learning rate, epochs, batch size and concept schedule as the other
arms. A concept is a task, and a class is "old" for a client once that client
has finished a task containing it. New task data is streamed in the same five
slices as the non-stationary arm; every round a client trains on its slice plus
its exemplar memory. The memory budget follows the rehearsal arm's memory at
the same point (capped at L when MEMORY_BOUND). The two therefore differ in
WHICH records are kept and HOW they are trained on, not in how many are
active. It is an FL-IIDS-derived implementation without the relay client, not a
reproduction; AUDIT_REPAIR_PLAN.md lists every departure from the paper.

`mode` picks the loss (VARIANTS). 'ce' keeps everything above but trains with
plain cross-entropy. 'ce_half', 'gb' and 'ce_ls' separate the loss scale, the
gradient-balance term and the label-smoothing term.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GAMMA_GB = 0.5
GAMMA_LS = 0.5

# Switches added after the 2026-09-26 audit (AUDIT_REPAIR_PLAN.md has the full
# equation-to-code matrix). The first value listed reproduces the original runs.
MEMORY_BOUND = True      # False: the budget grew with the rehearsal arm's per-concept
                         # cap, and herding prefixes beyond each group's quota stayed
                         # stored. True: budget min(L, that size), surplus rows dropped.
TEACHER = 'old_only'     # 'full': the old model's full softmax, sliced to the old
                         # classes. 'old_only': softmax over the old classes' logits
                         # only, the fixed-head analogue of the paper's old model,
                         # which has just the K_old outputs.
LS_TARGET = 'raw'        # 'normalised': the smoothed label rescaled to sum to 1.
                         # 'raw': as the paper describes it, the one-hot label with
                         # its old-class entries replaced (it sums to 1 or 2).
                         # 'raw' is used for fidelity, decided before any score.
# Eq. 8 is printed as KL(student || smoothed label). The smoothed label has a zero
# wherever a class is neither old nor the sample's own, and there KL in that
# direction is infinite, so the distillation direction KL(label || student) is the
# only computable reading. It is kept and disclosed rather than reversed.

# mode -> (classification term, its weight, label-smoothing weight)
VARIANTS = {'fliids':  ('gb', GAMMA_GB, GAMMA_LS),  # Eq. 9 as published
            'ce':      ('ce', 1.0, 0.0),            # plain cross-entropy (F0)
            'ce_half': ('ce', 0.5, 0.0),            # F0 at Eq. 9's loss scale
            'gb':      ('gb', GAMMA_GB, 0.0),       # Eq. 9 without L_LS
            'ce_ls':   ('ce', 0.5, GAMMA_LS)}       # Eq. 9 with CE in place of L_GB


def _features(model, X):
    """Penultimate-layer features, L2-normalised (iCaRL)."""
    with torch.no_grad():
        f = model.net[:-1](X)
    return F.normalize(f, dim=1)


def herding_order(model, X, k):
    """iCaRL herding (FL-IIDS Eq. 2): up to k indices, most representative first."""
    k = min(k, len(X))
    if k <= 0:
        return np.zeros(0, dtype=int)
    f = _features(model, X)
    mu = f.mean(0)
    chosen, running = [], torch.zeros_like(mu)
    avail = torch.ones(len(X), dtype=torch.bool, device=f.device)
    for j in range(1, k + 1):
        d = ((mu - (running + f) / j) ** 2).sum(1)
        d[~avail] = float('inf')
        i = int(torch.argmin(d))
        chosen.append(i)
        avail[i] = False
        running += f[i]
    return np.array(chosen, dtype=int)


def allocate(sizes, m):
    """FL-IIDS Eq. 3: slots proportional to group size, total m, capped by size."""
    sizes = np.asarray(sizes, dtype=float)
    if m <= 0 or sizes.sum() == 0:
        return np.zeros(len(sizes), dtype=int)
    raw = m * sizes / sizes.sum()
    q = np.minimum(np.floor(raw).astype(int), sizes.astype(int))
    left = int(m - q.sum())
    for i in np.argsort(-(raw - np.floor(raw))):      # largest remainder first
        if left <= 0:
            break
        if q[i] < sizes[i]:
            q[i] += 1
            left -= 1
    return q


class DynamicExampleMemory:
    """One client's memory: groups keyed by (task, class), herding order kept."""

    def __init__(self):
        self.groups = []     # dicts: X, y (herding-ordered), size, quota

    def seen_classes(self):
        return {int(g['y'][0]) for g in self.groups if len(g['y'])}

    def add_task(self, model, X, y, task, n_classes, device, k_max):
        # A group stores a herding prefix of the budget at the time it is added.
        # Quotas usually shrink as tasks accumulate, but not always: while the
        # budget is still growing, a later quota can exceed what was stored. It is
        # then capped at the stored prefix and those slots stay empty.
        for c in range(n_classes):
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            Xc = torch.tensor(X[idx]).float().to(device)
            order = herding_order(model, Xc, k_max)
            self.groups.append({'task': task, 'X': X[idx][order], 'y': y[idx][order],
                                'size': len(idx), 'kept': len(order)})

    def rebalance(self, m, discard=False):
        q = allocate([g['size'] for g in self.groups], m)
        for g, qi in zip(self.groups, q):
            g['quota'] = int(min(qi, g['kept']))
            if discard:                  # store only what is used
                g['X'], g['y'] = g['X'][:g['quota']], g['y'][:g['quota']]
                g['kept'] = g['quota']

    def stored(self):
        return sum(len(g['y']) for g in self.groups)

    def get(self):
        parts = [(g['X'][:g['quota']], g['y'][:g['quota']])
                 for g in self.groups if g.get('quota', 0) > 0]
        if not parts:
            return None, None
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def total(self):
        return sum(g.get('quota', 0) for g in self.groups)


def rehearsal_memory_size(per_task_counts, per_class):
    """How many records the rehearsal (B2) arm's cap rule keeps after these tasks."""
    return int(sum(min(n, per_class) for counts in per_task_counts for n in counts))


def fliids_loss(logits, y, old_mask_classes, teacher_logits, cls='gb',
                w_cls=GAMMA_GB, w_ls=GAMMA_LS):
    """
    w_cls * (L_GB or CE) + w_ls * L_LS for one mini-batch (Eq. 9 by default).
    old_mask_classes: bool tensor [C], True for classes old to this client.
    teacher_logits: [B, C] logits of the old model, or None in the first task.
    The L_GB weights are computed without gradient (the paper does not say).
    """
    ce = F.cross_entropy(logits, y, reduction='none')
    if cls == 'gb':
        with torch.no_grad():
            p_y = torch.softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
            g = (1.0 - p_y)                               # |p_y - 1|, Eq. 4
            is_old = old_mask_classes[y]
            g_new = g[~is_old].mean() if (~is_old).any() else g.new_tensor(1.0)
            g_old = g[is_old].mean() if is_old.any() else g.new_tensor(1.0)
            g_m = torch.where(is_old, g_old, g_new).clamp_min(1e-8)
            w = g / g_m                                    # Eq. 7 weight
        loss = w_cls * (w * ce).mean()
    else:
        loss = w_cls * ce.mean()
    if w_ls and teacher_logits is not None and old_mask_classes.any():
        if TEACHER == 'old_only':
            tp = torch.softmax(teacher_logits.masked_fill(~old_mask_classes, float('-inf')), 1)
        else:
            tp = torch.softmax(teacher_logits, 1)
        target = F.one_hot(y, logits.shape[1]).float()
        target[:, old_mask_classes] = tp[:, old_mask_classes]
        if LS_TARGET == 'normalised':
            target = target / target.sum(1, keepdim=True).clamp_min(1e-8)
        # Eq. 8 in the distillation direction KL(target || student); see the note
        # at the top. kl_div treats target entries of 0 as contributing 0.
        loss = loss + w_ls * F.kl_div(F.log_softmax(logits, 1), target,
                                      reduction='batchmean')
    return loss


def local_train_fliids(P, model_cls, global_state, X, y, device, old_classes,
                       teacher_state, mode, n_classes, counter):
    model = model_cls(n_classes=n_classes).to(device)
    model.load_state_dict(global_state)
    model.train()
    cls, w_cls, w_ls = VARIANTS[mode]
    teacher = None
    if teacher_state is not None and w_ls > 0:
        teacher = model_cls(n_classes=n_classes).to(device)
        teacher.load_state_dict(teacher_state)
        teacher.eval()
    n = len(X)
    if n < 2:
        return {k: v.cpu() for k, v in model.state_dict().items()}, n
    old_mask = torch.zeros(n_classes, dtype=torch.bool, device=device)
    for c in old_classes:
        old_mask[c] = True
    opt = P._make_opt(model.parameters(), P.SGD_LR)
    B = P.BATCH_SIZE
    for _ in range(P.LOCAL_EPOCHS):
        for s in range(0, n, B):
            Xb, yb = X[s:s + B], y[s:s + B]
            if len(Xb) < 2:
                continue
            opt.zero_grad()
            logits = model(Xb)
            if mode == 'ce':                 # kept as in the original runs
                loss = F.cross_entropy(logits, yb)
            else:
                tl = None
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(Xb)
                loss = fliids_loss(logits, yb, old_mask, tl, cls, w_cls, w_ls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), P.CLIP_NORM)
            opt.step()
            counter['steps'] += 1
    return {k: v.cpu() for k, v in model.state_dict().items()}, n


def _accuracy(model_cls, state, X, y, device, n_classes):
    m = model_cls(n_classes=n_classes).to(device)
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(X).float().to(device)).argmax(1).cpu().numpy()
    return float((p == y).mean())


def run_fliids(P, model_cls, evaluate, concepts, test_set, device, n_classes,
               mode='fliids', verbose=False, on_concept_start=None):
    """
    P: the pipeline module (ids_cda_pipeline or unsw_cda_pipeline) supplying the
    shared training constants and FedAvg. Returns (state, results, info) like the
    other arms; results has one row per concept. mode: a key of VARIANTS.
    on_concept_start(i, state): called before task i >= 1 (forward-stream eval).
    """
    K, R = P.N_CLIENTS, P.R_PER_CONCEPT
    per_class = P.PER_CLASS
    memories = [DynamicExampleMemory() for _ in range(K)]
    teachers = [None] * K
    task_counts = [[] for _ in range(K)]      # per client, per task: class counts
    counter = {'steps': 0}
    gstate = {k: v for k, v in model_cls(n_classes=n_classes).to(device).state_dict().items()}
    results, total_r, mem_sizes, mem_stored = [], 0, [], []

    for t, concept in enumerate(concepts):
        if on_concept_start and t >= 1:
            on_concept_start(t, {k: v.cpu() for k, v in gstate.items()})
        slices, olds = [], []
        for k, (Xk, yk) in enumerate(concept['client_data']):
            olds.append(sorted(memories[k].seen_classes()))
            if len(Xk) == 0:
                slices.append([(None, None)] * R)
                continue
            sp = np.array_split(np.arange(len(Xk)), R)
            slices.append([(Xk[s], yk[s]) if len(s) else (None, None) for s in sp])

        task_states = []
        for r in range(R):
            updates = []
            for k in range(K):
                Xs, ys = slices[k][r]
                Xm, ym = memories[k].get()
                parts = [(a, b) for a, b in ((Xs, ys), (Xm, ym)) if a is not None and len(a)]
                if not parts:
                    continue
                Xl = np.concatenate([a for a, _ in parts])
                yl = np.concatenate([b for _, b in parts])
                perm = np.random.permutation(len(Xl))
                Xg = torch.tensor(Xl[perm]).float().to(device)
                yg = torch.tensor(yl[perm]).long().to(device)
                st, n = local_train_fliids(P, model_cls, gstate, Xg, yg, device,
                                           olds[k], teachers[k], mode, n_classes, counter)
                updates.append((st, n))
            if updates:
                cpu = P.fedavg_aggregate(updates)
                gstate = {k: v.to(device) for k, v in cpu.items()}
            task_states.append({k: v.detach().clone() for k, v in gstate.items()})
            total_r += 1

        # task end: store exemplars, rebalance to the rehearsal arm's record count,
        # and let each client keep its best global model of this task as teacher
        gmodel = model_cls(n_classes=n_classes).to(device)
        gmodel.load_state_dict(gstate)
        gmodel.eval()
        for k, (Xk, yk) in enumerate(concept['client_data']):
            if len(Xk):
                task_counts[k].append([int((yk == c).sum()) for c in range(n_classes)])
            m_k = rehearsal_memory_size(task_counts[k], per_class)
            if MEMORY_BOUND:
                m_k = min(m_k, P.L_CAPACITY)
            if len(Xk):
                memories[k].add_task(gmodel, Xk, yk, t, n_classes, device, m_k)
            memories[k].rebalance(m_k, discard=MEMORY_BOUND)
            Xm, ym = memories[k].get()
            if Xm is not None and len(Xm):
                accs = [_accuracy(model_cls, s, Xm, ym, device, n_classes) for s in task_states]
                teachers[k] = task_states[int(np.argmax(accs))]
        mem_sizes.append([memories[k].total() for k in range(K)])
        mem_stored.append([memories[k].stored() for k in range(K)])

        overall, per_cls, macro_f1, per_rec = evaluate(
            {k: v.cpu() for k, v in gstate.items()}, test_set[0], test_set[1], device)
        results.append({'event': concept['name'], 'round': total_r, 'overall': overall,
                        'macro_f1': macro_f1, **P._metrics(per_cls, per_rec)})
        if verbose:
            print(f"  [{mode}] after {concept['name']:<14} macroF1={macro_f1:.4f} "
                  f"memory={sum(mem_sizes[-1])}", flush=True)
    return ({k: v.cpu() for k, v in gstate.items()}, results,
            {'grad_steps': counter['steps'], 'memory_records': mem_sizes,
             'memory_stored': mem_stored})
