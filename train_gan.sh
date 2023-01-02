#!/usr/bin/env bash

echo "Start Running Training Bash Script"

for DATA in "mnist" "fmnist" "cifar10"
    do
        for HIDDEN in 2 16 64
            do
                echo "Train model on $DATA dataset using hidden size $HIDDEN for $1 epochs"
                python3 ./src/train_gan.py --hidden $HIDDEN -e $1 -bs 64 -ls mse --dataset $DATA
                python3 ./src/train_gan.py --hidden $HIDDEN -e $1 -bs 64 -ls mse --dataset $DATA -c
            done
    done

# ./train_gan.sh 75