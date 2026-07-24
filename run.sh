#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TOKENIZERS_PARALLELISM=false
nvidia-smi || true
python3 -m pip install -q --no-cache-dir "peft>=0.14" "accelerate>=0.30" || \
  python3 -m pip install -q --no-cache-dir --break-system-packages "peft>=0.14" "accelerate>=0.30"
python3 -c "import torch, transformers, vllm, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__, 'peft', peft.__version__)"
python3 -u src/main.py
