#!/usr/bin/env bash
set -e
echo "========================================================"
echo "[RetinaGuard] Step 08: Generate Scientific Figures & Tables"
echo "========================================================"
python -m src.explainability.visualize
echo "Figure and table generation complete."
