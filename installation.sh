#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Activate your virtual environment before running this script.
# For CUDA 12.4: TORCH_WHEEL_INDEX_URL=https://download.pytorch.org/whl/cu124 \
#              PYG_WHEEL_URL=https://data.pyg.org/whl/torch-2.5.0+cu124.html bash installation.sh
python -m pip install 'torch==2.5.0' --index-url "${TORCH_WHEEL_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
python -m pip install -r requirements.txt
python -m pip install --only-binary=:all: --no-index \
  --find-links "${PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.5.0+cpu.html}" \
  torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 pyg-lib==0.4.0
python -m pip check
