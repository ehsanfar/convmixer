#!/bin/bash
# NUM_PROC=$1
# shift
python3 train.py /home/etanfar/Documents/DATA/ --train-split /home/etanfar/Documents/DATA/np-DATA/HistFigsClass8-rgb-train --val-split /home/etanfar/Documents/DATA/np-DATA/HistFigsClass8-rgb-train --model convmixer_1536_20 -b 16 -j 10 --opt adamw --epochs 5 --sched onecycle --amp --input-size 3 224 224 --lr 0.01 --aa rand-m9-mstd0.5-inc1 --cutmix 0.5 --mixup 0.5 --reprob 0.25 --remode pixel --num-classes 8 --warmup-epochs 0 --opt-eps=1e-3 --clip-grad 1.0

python3 train.py /home/etanfar/Documents/DATA/ --train-split /home/etanfar/Documents/DATA/np-DATA/HistFigsClass8-rgb-train --val-split /home/etanfar/Documents/DATA/np-DATA/HistFigsClass8-rgb-train --model convmixer_1536_20 -b 32 -j 10 --opt adamw --epochs 5 --sched onecycle --amp --input-size 3 300 300 --lr 0.01 --aa rand-m9-mstd0.5-inc1 --remode pixel --num-classes 8 --warmup-epochs 0 --opt-eps=1e-3 --clip-grad 1.0