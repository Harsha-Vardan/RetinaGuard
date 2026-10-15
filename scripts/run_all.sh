#!/usr/bin/env bash
set -e
echo "===================================================================="
echo " RetinaGuard End-to-End Automated Pipeline Execution"
echo "===================================================================="

bash scripts/00_check_env.sh
bash scripts/03_smoke_test.sh
bash scripts/07_evaluate.sh
bash scripts/08_make_tables_figures.sh
bash scripts/10_build_slides.sh

echo "===================================================================="
echo " RetinaGuard Pipeline Execution Completed Successfully!"
echo "===================================================================="
