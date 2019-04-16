#!/usr/bin/env bash

for fold_id in 0 1 2 3
do
    python train.py -m UNet11 \
                    -j 12 \
                    -b 16 \
                    -n 40 \
                    --f $fold_id
done