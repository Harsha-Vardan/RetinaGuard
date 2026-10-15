#!/usr/bin/env bash
set -e
echo "========================================================"
echo "[RetinaGuard] Step 10: Generate PowerPoint Presentation"
echo "========================================================"
python slides/build_pptx.py
echo "Slide deck build complete: slides/deck.pptx"
