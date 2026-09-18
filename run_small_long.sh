#!/bin/bash
# Small: урт сургалт + хүчтэй shaping — салангид (log: results/train_small_long.log)
cd /c/Users/Acer/Desktop/RL/GenPent
PYTHONIOENCODING=utf-8 ../GenPent-venv/Scripts/python training/train_maskppo.py \
    --scenario small --timesteps 400000 --seed 42 --shaping 100 \
    --run-name MaskPPO_small_s42_sh100_long > results/train_small_long.log 2>&1
echo "DONE" >> results/train_small_long.log
