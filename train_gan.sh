#!/usr/bin/env bash

echo "Start Running Training Bash Script"
gen="ConvGAN"
disc="ConvDiscriminator"

for DATA in "mnist" "fmnist" "cifar10"
    do
        for HIDDEN in 2 16 64
            do
                echo "Train generator $gen and discriminator $disc on $DATA dataset using hidden size $HIDDEN for $1 epochs"
                python3 ./src/train_gan.py --hidden $HIDDEN -e $1 -bs 64 -ls mse --dataset $DATA --generator $gen --discriminator $disc
            done
    done

# ./train_gan.sh 75