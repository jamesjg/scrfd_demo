#!/usr/bin/env bash

GPU=0
GROUP=scrfd
TASK=scrfd_34g

#CUDA_VISIBLE_DEVICES="$GPU" python -u tools/benchmark_vga.py ./configs/"$GROUP"/"$TASK".py ./work_dirs/"$TASK"/latest.pth #--cpu
CUDA_VISIBLE_DEVICES="$GPU" python -u tools/test_widerface.py ./configs/"$GROUP"/"$TASK".py ./work_dirs/"$TASK"/model.pth --mode 0 --out wouts --save-preds
