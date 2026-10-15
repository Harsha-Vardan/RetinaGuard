#!/usr/bin/env bash
set -e
echo "========================================================"
echo "[RetinaGuard] Step 02: Image Preprocessing & FOV Cropping"
echo "========================================================"
python -m src.data.preprocessing
echo "Preprocessing verification complete."
