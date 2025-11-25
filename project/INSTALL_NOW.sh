#!/bin/bash

# Emergency Install Script - Get Everything Working in 10 Minutes
# Run this to install all required packages

set -e  # Exit on error

echo "=========================================="
echo "INSTALLING ALL REQUIRED PACKAGES"
echo "=========================================="

# Step 1: PyTorch (CPU version - fast install)
echo ""
echo "[1/7] Installing PyTorch (CPU version)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 2: PyTorch Geometric
echo ""
echo "[2/7] Installing PyTorch Geometric..."
pip3 install torch-geometric

# Step 3: Chronos Foundation Model
echo ""
echo "[3/7] Installing Chronos..."
pip3 install chronos-forecasting

# Step 4: Tigramite (PCMCI)
echo ""
echo "[4/7] Installing Tigramite..."
pip3 install tigramite

# Step 5: Core dependencies
echo ""
echo "[5/7] Installing core dependencies..."
pip3 install numpy pandas scikit-learn scipy networkx

# Step 6: Visualization
echo ""
echo "[6/7] Installing visualization tools..."
pip3 install matplotlib seaborn plotly

# Step 7: Other requirements
echo ""
echo "[7/7] Installing remaining dependencies..."
pip3 install tqdm pyyaml loguru transformers accelerate

echo ""
echo "=========================================="
echo "✅ ALL PACKAGES INSTALLED!"
echo "=========================================="

echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import torch_geometric; print(f'✅ PyTorch Geometric: {torch_geometric.__version__}')"
python3 -c "import chronos; print(f'✅ Chronos: OK')"
python3 -c "import tigramite; print(f'✅ Tigramite: OK')"
python3 -c "import numpy; import pandas; import networkx; print(f'✅ Core packages: OK')"

echo ""
echo "=========================================="
echo "🎉 READY TO TEST!"
echo "=========================================="
echo ""
echo "Next step: Run this command:"
echo "  cd /home/user/fault-detection-microservices/project"
echo "  python3 scripts/test_encoders.py --n_cases 1"
echo ""
