# TabFM integration development protocol, 28 September 2026

This is a new exploratory integration study, not an amendment to the completed
600-609 confirmation. No old campaign or pooled-probe score is reused as an
integration result. The original main results remain immutable.

Design: repaired benchmark loaders, original ordered concepts, ten clients,
5000 rows/class cap, causal initial-concept preprocessing, bounded rehearsal,
identical labels/cohorts within each paired run, FedAvg, original training
schedule. Development seeds 42-46; alpha 0.1 and 0.5. First smoke: seed 42,
alpha .5, UNSW. Extend after correctness and runtime checks. Report completed
cells and missing cells explicitly; no stopping based on a favorable score.

Integration: frozen TabFM front end, with its ICL prediction stage removed
from device residency, followed by the existing federated trainable MLP.
One fixed context of up to 64 benign initial-training rows from the largest
initial client is shared once. Context is never refreshed. This assumption is
an explicit exception to raw-data locality, not a privacy-preserving protocol.
No test labels, later labels, or later-fitted transformation is used.

For a capacity-matched first implementation, project 2048 frozen coordinates
through a fixed Gaussian projection to the dataset's original input dimension.
The projection seed is fixed at 20260928 and never selected on outcomes.
Standardize projected features on initial benign reference embeddings only.
The raw baseline keeps its original causal preprocessing. Head architecture,
optimizer and initial parameter draws are identical across feature conditions.
Memory selection is label/index based. The representation may change detector
alarms and therefore future memory contents and training exposure: effects are
system-level effects, not a pure feature causal effect.

Four paired arms: raw always-adapt, raw CDA quorum5, TabFM always-adapt,
TabFM CDA quorum5. The always-adapt contrast separates feature utility from
detector-policy interaction. Record macro-F1 retention, all concept histories,
forward scores, class forgetting, actual optimizer steps, detector events,
feature-extraction time, resident backbone/head size, peak CUDA memory, and
context broadcast payload. No claim of training-free learning: the head trains.

Verify before scoring: stable context identity, no evaluation context rows,
row-order and batching invariance, determinism, finite embeddings and projection,
matching raw/transformed label vectors, and input dimensions. Record source and
checkpoint hashes. A failing check halts the experiment, not merely a warning.

Report paired seed estimates and intervals for completed development cells.
These are exploratory; no confirmatory superiority statement or retroactive
fresh-seed claim. A new frozen protocol and independently reserved seeds are
required for a confirmatory integration claim. RQ4's original question about
adaptation without training remains only partially addressed by this design.

Pre-score engineering amendment: the BF16 frontend failed batching invariance on initial training rows (maximum absolute difference 0.3046875). No accuracy result was inspected. Float32 loading is tested instead, and the failure log is retained. The loader now allocates only frontend weights from the fixed local checkpoint using a meta-device skeleton, avoiding full-model CPU memory pressure.

Execution note (28 September, continuation): the moderate-heterogeneity
screen (alpha 0.5) is completed first on UNSW-NB15 and CICIDS2017, five
paired development seeds each. The planned alpha 0.1 extension and 2018
integration are not part of this first completed screen and must not be
represented as measured. Two independent dataset processes may overlap;
wall-clock timing from these throughput runs is descriptive and cannot
support an isolated latency/speed comparison. Parameter bytes, context
payloads and optimizer-step counts do not depend on that overlap.
