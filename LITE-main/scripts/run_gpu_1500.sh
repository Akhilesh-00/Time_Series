#!/bin/bash
# Long-epoch LITE ablation on the laptop GPU (run inside WSL Ubuntu-22.04).
# Usage from Windows:  wsl -d Ubuntu-22.04 -- bash /mnt/c/TimeSeries/Time_Series/LITE-main/scripts/run_gpu_1500.sh
# Resumable: rerun the same command and finished runs are skipped.
NV=$(~/lite-gpu/bin/python -c "import nvidia, os; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=$(ls -d "$NV"/*/lib | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
~/lite-gpu/bin/python -u run_ablation.py --datasets SemgHandMovementCh2 --seeds 0 1 2 3 4 \
    --epochs 1500 --patience 100 --lr-patience 50 \
    --conditions original validation validation_refit first_difference validation_and_difference \
    --output-dir results_ablation/gpu_1500ep/semg --verbose 0 2>&1 \
    | grep --line-buffered -v "NUMA\|numa_node\|TensorRT\|step$\|━" >> results_ablation_gpu_1500ep.log
