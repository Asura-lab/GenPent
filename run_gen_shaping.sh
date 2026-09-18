#!/bin/bash
# Generalization: shaping-тэй дахин туршилт — салангид (log: results/gen_shaping.log)
cd /c/Users/Acer/Desktop/RL/GenPent
PYTHONIOENCODING=utf-8 ../GenPent-venv/Scripts/python evaluation/generalize.py \
    --timesteps 300000 \
    --out results/generalization_shaping.csv > results/gen_shaping.log 2>&1
echo "DONE" >> results/gen_shaping.log
