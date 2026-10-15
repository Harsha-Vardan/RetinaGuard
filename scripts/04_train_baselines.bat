@echo off
echo ========================================================
echo [RetinaGuard] Step 04: Train Baseline Model
echo ========================================================
python -m src.training.train --config configs/baseline.yaml
echo Baseline training complete.
