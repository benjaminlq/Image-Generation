#!/usr/bin/env bash

echo "Start Running Training Bash Script"

for DATA in "fmnist" "cifar10"
    do
        for HIDDEN in 2 32 64 128
            do
                for MODEL in "BaseVAE" "DeepVAE" "ConvVAE"
                    do
                        echo "Train $MODEL model on $DATA dataset using hidden size $HIDDEN"
                        python3 ./src/train.py --model $MODEL --hidden $HIDDEN -e 25 --dataset $DATA
                    done
            done
    done