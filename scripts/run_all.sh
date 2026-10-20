#!/usr/bin/env bash
set -e
echo "===================================================================="
echo " RetinaGuard End-to-End Automated Pipeline Execution"
echo "===================================================================="

python scripts/run_pipeline.py --stage all

echo "===================================================================="
echo " RetinaGuard Pipeline Execution Completed Successfully!"
echo "===================================================================="
