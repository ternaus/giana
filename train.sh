#!/usr/bin/env bash

python train.py -c configs/dataset.json \
                --jaccard_weight 0.3 \
                --fold_id 0 \
                --device_ids 0,1 \
                -j 0