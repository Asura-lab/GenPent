#!/bin/bash
# Small scenario сургалт — салангид ажиллуулна (log: results/train_small.log)
cd /c/Users/Acer/Desktop/RL/GenPent
PYTHONIOENCODING=utf-8 ../GenPent-venv/Scripts/python training/train_maskppo.py \
    --scenario small --timesteps 150000 --seed 42 \
    --run-name MaskPPO_small_s42 > results/train_small.log 2>&1
echo "DONE" >> results/train_small.log
