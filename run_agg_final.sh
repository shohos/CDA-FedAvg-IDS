#!/usr/bin/env bash
# Protocol v3 (aggregator study). Usage: bash run_agg_final.sh <rule> <dataset> [<dataset> ...]
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
RULE=$1; shift
MU=""; [ "$RULE" = "fedprox" ] && MU="--mu 0.001"
for DS in "$@"; do
  python final_eval.py --impl repaired --dataset $DS --alphas 0.1,0.5,100 \
    --seeds 800,801,802,803,804,805,806,807,808,809 --arms nonstat,rehearsal,cda \
    --fliids-lr 0.02 --agg $RULE $MU --out agg/${RULE}_$DS.json
done
echo "[AGG $RULE DONE]"
