#!/bin/bash
# CI dry run on the Mac: 20 train / 20 val tiles, 1 epoch, batch 1, with a simulated preemption.
#   bash tests/dry_run.sh configs/p5_head.yaml      (about 5-10 min on the M2 Air per config)
# Checks: preempt -> exit 75; --resume auto finishes the epoch -> exit 0; again -> "already finished", exit 0;
# the export loads through src/detector.py.
set -u
CONFIG=${1:?config}
PY=${PY:-python}
NAME=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$CONFIG")
OUT=$(PYTHONPATH=src $PY -c "from utils import under_root; print(under_root('runs/$NAME-dryrun'))")   # $DR_ROOT/runs/
fail() { echo "DRY RUN FAILED: $*"; exit 1; }
case "$OUT" in */runs/*-dryrun) ;; *) fail "unexpected output dir '$OUT'";; esac   # guard before rm -rf
echo "output: $OUT"
rm -rf "$OUT"
quiet() { grep -vE 'Redirects|Loading weights|HF_TOKEN'; }

echo "== 1: simulated preemption after 3 steps"
$PY src/train.py --config "$CONFIG" --dry-run --simulate-preempt 3 2>&1 | quiet
[ "${PIPESTATUS[0]}" = 75 ] || fail "expected exit 75"

echo "== 2: --resume auto"
$PY src/train.py --config "$CONFIG" --dry-run --resume auto 2>&1 | quiet
[ "${PIPESTATUS[0]}" = 0 ] || fail "resume did not finish"

echo "== 3: --resume auto on a finished run"
$PY src/train.py --config "$CONFIG" --dry-run --resume auto 2>&1 | quiet | tee /dev/stderr | grep -q "already finished" \
  || fail "finished run was not recognised"

echo "== 4: export loads through the detector hook"
ADAPTER=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['export_best'])" "$OUT/run.json")
PYTHONPATH=src $PY -c "
import sys; from detector import Detector
d = Detector(adapter=sys.argv[1]); print('adapter_info:', d.adapter_info)" "$ADAPTER" 2>&1 | quiet
[ "${PIPESTATUS[0]}" = 0 ] || fail "detector could not load $ADAPTER"

echo "== log.csv"
$PY -c "
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    print('epoch', r['epoch'], 'train_loss', r['train_loss'][:6] or '-', 'val_loss', r['val_loss'][:6],
          'val_AP50', r['val_AP50'], 'best', r['best'])" "$OUT/log.csv"
echo "DRY RUN PASSED: $CONFIG"
