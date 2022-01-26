#!/bin/sh
./train.py --epochs 50 --deterministic  --compress sorting_schedule.yaml --model simplesortingnetbnbb --dataset sortingbb --qat-policy qat_policy_sorting.yaml --param-hist --enable-tensorboard --batch-size 64 --regression --optimizer Adam --lr 0.006 --device MAX78000 "$@"
