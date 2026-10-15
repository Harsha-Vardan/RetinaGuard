@echo off
echo ========================================================
echo [RetinaGuard] Step 01: Build Master Metadata & Harmonize
echo ========================================================
python -m src.data.dataset --build-metadata
echo Metadata build complete.
