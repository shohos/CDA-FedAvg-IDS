#!/usr/bin/env bash
# Download the ten CSE-CIC-IDS2018 CICFlowMeter day files (about 6.5 GB) into
# CICIDS2018/ next to this script. Files already present are skipped.
set -e
cd "$(dirname "$0")"
mkdir -p CICIDS2018 && cd CICIDS2018
B="https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms"
for f in Wednesday-14-02-2018 Thursday-15-02-2018 Friday-16-02-2018 Thuesday-20-02-2018 \
         Wednesday-21-02-2018 Thursday-22-02-2018 Friday-23-02-2018 Wednesday-28-02-2018 \
         Thursday-01-03-2018 Friday-02-03-2018; do
  name="${f}_TrafficForML_CICFlowMeter.csv"
  [ -s "$name" ] || { echo "downloading $name"; curl -fsS -o "$name" "$B/$name"; }
done
echo "CSE-CIC-IDS2018 ready in $(pwd)"
