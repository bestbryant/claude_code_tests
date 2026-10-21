#!/bin/bash
# scripts/push_retry.sh <kernel dir>: retry kaggle kernels push until it reports success (slot limits, 429, 5xx, network)
for i in $(seq 1 200); do
  out=$(kaggle kernels push -p "$1" 2>&1)
  echo "$(date -u +%H:%M) $(echo "$out" | tail -1)"
  echo "$out" | grep -qi "successfully pushed" && exit 0
  sleep 120
done
echo "FAILED to push $1"; exit 1
