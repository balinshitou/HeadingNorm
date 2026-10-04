#!/bin/zsh
# Wait for the E3 training run to finish, then evaluate, analyse and rebuild the figures.
cd "$(dirname "$0")/.." || exit 1
PY=./.venv/bin/python

while pgrep -f "run_e3_train.py" > /dev/null; do
  sleep 60
done
echo "[chain] training finished at $(date)"

HN_DEVICE=mps $PY tools/e3_evaluate.py --seeds 4 || { echo "[chain] evaluate failed"; exit 1; }
echo "[chain] evaluation finished at $(date)"

$PY tools/e3_analyze.py || { echo "[chain] analyze failed"; exit 1; }
$PY tools/e1_repeatability_table.py || echo "[chain] repeatability table failed"
$PY tools/mst_figures.py || echo "[chain] figures failed"
echo "[chain] done at $(date)"
