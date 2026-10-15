#!/usr/bin/env bash
set -e
echo "========================================================"
echo "[RetinaGuard] Step 07: Cross-Domain & Clinical Evaluation"
echo "========================================================"
python -m src.evaluation.cross_domain_eval
python -m src.evaluation.benchmark
echo "Cross-domain evaluation complete."
