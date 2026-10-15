@echo off
echo ========================================================
echo [RetinaGuard] Step 05: Train RetinaGuard Multi-Task LG-CRL
echo ========================================================
python -m src.training.train_retinaguard --config configs/lgcrl.yaml
echo RetinaGuard training complete.
