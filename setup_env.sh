#!/bin/bash
# setup_env.sh
# Run this inside your MambaVision Docker container to unify dependencies.

set -e

echo "Updating environment for Fusion-mamba..."

# 1. Basic requirements (glue and ReID standard)
pip install yacs opencv-python tqdm einops==0.8.1

# 2. Upgrade timm to latest (needed by MambaVision)
pip install timm==1.0.15

# 3. Ensure mamba-ssm and causal-conv1d (should be there, but verify)
# If missing, it will attempt to install, but compile might take time.
# In a MambaVision image, these should already be optimized.
pip install mamba-ssm==2.2.4 causal-conv1d==1.4.0

# 4. Check torch version (MambaVision usually needs 2.1+)
python -c "import torch; print(f'Torch version: {torch.__version__}'); assert torch.cuda.is_available(), 'CUDA not available!'"

echo "Environment integration complete!"
