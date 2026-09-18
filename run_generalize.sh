#!/bin/bash
# Бүрэн generalization туршилт — салангид ажиллуулна (log: results/gen_full.log)
cd /c/Users/Acer/Desktop/RL/GenPent
PYTHONIOENCODING=utf-8 ../GenPent-venv/Scripts/python evaluation/generalize.py \
    --timesteps 200000 \
    --out results/generalization_full.csv > results/gen_full.log 2>&1
echo "DONE" >> results/gen_full.log
