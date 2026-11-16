#!/bin/bash
# Weak scale-only augmentation (sd 0.05) LITE on small UCR datasets (laptop GPU, WSL Ubuntu-22.04).
# Usage from Windows:  wsl -d Ubuntu-22.04 -- bash /mnt/c/TimeSeries/Time_Series/LITE-main/scripts/run_gpu_scale005_small.sh
# Resumable: rerun the same command and finished runs are skipped.
# "augment" here = weak scale only (amplitude scaling sd 0.05, no jitter, no shift).
NV=$(~/lite-gpu/bin/python -c "import nvidia, os; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=$(ls -d "$NV"/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
~/lite-gpu/bin/python -u run_ablation.py --datasets Beef OliveOil ShapeletSim --seeds 0 1 2 3 4 \
    --epochs 1500 --lr-patience 50 --conditions augment \
    --aug-scale 0.05 --aug-jitter 0 --aug-shift 0 \
    --output-dir results_ablation/gpu_scale005_small --verbose 0 2>&1 \
    | grep --line-buffered -v "NUMA\|numa_node\|TensorRT\|step$\|━" >> results_ablation_gpu_scale005_small.log
