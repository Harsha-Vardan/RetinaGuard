#!/usr/bin/env bash
set -e
echo "========================================================"
echo "[RetinaGuard] Step 00: Environment & Hardware Validation"
echo "========================================================"
python -c "import torch, torchvision, sklearn, pptx; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (AMD64)'); print('Environment Check Passed!')"
