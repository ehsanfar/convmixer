#!/bin/bash
# NUM_PROC=$1
# shift
python3 train.py /home/etanfar/Documents/DATA/ --train-split /home/etanfar/Documents/DATA/np-DATA/HistFigsClass8-rgb-train --val-split /home/etanfar/Documents/DATA/np-DATA/HistFigsClass8-rgb-val --model convmixer_1536_20 -b 16 -j 10 --opt adamw --epochs 5 --sched onecycle --amp --input-size 3 224 224 --lr 0.01 --aa rand-m9-mstd0.5-inc1 --cutmix 0.5 --mixup 0.5 --reprob 0.25 --remode pixel --num-classes 8 --warmup-epochs 0 --opt-eps=1e-3 --clip-grad 1.0

python3 train.py /home/etanfar/Documents/DATA/ --train-split /home/etanfar/Documents/DATA/np-DATA/HistFigsClass8-rgb-train --val-split /home/etanfar/Documents/DATA/np-DATA/HistFigsClass8-rgb-val --model convmixer_1536_20 -b 32 -j 10 --opt adamw --epochs 5 --sched onecycle --amp --input-size 3 300 300 --lr 0.01 --aa rand-m9-mstd0.5-inc1 --remode pixel --num-classes 8 --warmup-epochs 0 --opt-eps=1e-3 --clip-grad 1.0

# python3 validate.py output/train/20230925-223110-convmixer_1536_20-274/model_best.pth.tar --model convmixer_1536_20 --num-classes 8 -b 32 -j 10 --input-size 3 274 274 --amp --split /home/etanfar/Documents/DATA/np-DATA/HistFigsClass8-rgb-val --num-classes 8 

python3 validate.py /home/etanfar/Documents/DATA/ --val-split /home/etanfar/Documents/DATA/np-DATA/HistFigsClass8-rgb-val --model convmixer_1536_20 --num-classes 8 -b 32 -j 10 --input-size 3 274 274 --num-classes 8 


python3 validate.py --model convmixer_1536_20 --checkpoint 20230927-215813-convmixer_1536_20-574 -b 8 -j 10 --input-size 3 574 574 --num-classes 8

