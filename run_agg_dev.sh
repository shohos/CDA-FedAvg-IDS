#!/usr/bin/env bash
# Aggregator study, development (consumed seeds 42-44). Usage: bash run_agg_dev.sh cicids|unsw
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
DS=$1
C="--impl repaired --dataset $DS --alphas 0.1,0.5,100 --seeds 42,43,44 --arms rehearsal --fliids-lr 0.02"
for MU in 0.001 0.01 0.1 1.0; do
  python final_eval.py $C --agg fedprox --mu $MU --out agg_dev/fedprox_mu${MU}_$DS.json
done
for AG in fedavg scaffold median; do
  python final_eval.py $C --agg $AG --out agg_dev/${AG}_$DS.json
done
echo "[AGG DEV $DS DONE]"
