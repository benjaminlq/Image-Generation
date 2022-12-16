#!/usr/bin/env bash

echo "Start Running Training Bash Script"

for HIDDEN in 2 32 64 128
do
    for MODEL in "BaseVAE" "DeepVAE" "ConvVAE"
        do
            python3 ./src/train.py --model $MODEL --hidden $HIDDEN -e 20
        done
done